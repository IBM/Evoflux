from __future__ import annotations

import json
import sys, os, shlex
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from ..benchmark_adapter import BenchmarkAdapter
from ..dsl import ToolSpec, ToolRegistry, GraphTemplate, Node, Action, PromptParams
from ..executor import ExecResult, MetricBreakdown
from ..compiler import CompiledWorkflow


"""
mcp_router/adapters/mcpbench.py
--------------------------------
BenchmarkAdapter for Accenture MCP-Bench.
https://github.com/Accenture/mcp-bench

build_templates() uses TemplatePlanner (mcp_router/template_planner.py) to ask
an LLM to propose semantically grounded workflow templates based on a sample of
the actual task corpus and the full tool-description catalog.

If no credentials are configured, or the LLM call fails, build_templates()
falls back to the original ad-hoc server enumeration automatically.
"""


_DEFAULT_TASK_FILES = {
    "single":  "tasks/mcpbench_tasks_single_runner_format.json",
    "2server": "tasks/mcpbench_tasks_multi_2server_runner_format.json",
    "3server": "tasks/mcpbench_tasks_multi_3server_runner_format.json",
}

_SERVER_TOOL_API_KEY= ["NPS_API_KEY","NASA_API_KEY","HF_TOKEN","GOOGLE_MAPS_API_KEY","NCI_API_KEY"]
# ── LLMProvider bridge for mcp-bench's LLMJudge ──────────────────────────────

# Short aliases → full Bedrock cross-region inference profile IDs.
_BEDROCK_MODEL_ALIASES: Dict[str, str] = {
    "claude-sonnet-4":   "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4":     "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4.5":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4.6":   "us.anthropic.claude-opus-4-6-v1:0",
    "claude-sonnet-4.6": "us.anthropic.claude-sonnet-4-6-v1:0",
}

_VALID_PROVIDERS = {"bedrock", "openrouter", "azure", "base_url"}


class _MCPBenchLLMProvider:
    """
    Thin adapter that implements the LLMProvider protocol expected by
    mcp-bench's LLMJudge / TaskEvaluator.

    Protocol (async):
        async get_completion(system_prompt, user_prompt, max_tokens) -> str
        clean_and_parse_json(raw_json) -> Any

    Supported providers
    -------------------
    "openrouter"  — OpenRouter chat-completions endpoint (requires openrouter_api_key)
    "azure"       — Azure OpenAI (requires azure_api_key + azure_endpoint)
    "bedrock"/"aws" — AWS Bedrock Converse API (uses boto3 credential chain, no key arg needed)
    """

    def __init__(
        self,
        model: str,
        provider: str = "openrouter",
        openrouter_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        provider = provider.lower().strip()
        if provider == "aws":
            provider = "bedrock"
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"[_MCPBenchLLMProvider] Unknown provider {provider!r}. "
                f"Choose from {sorted(_VALID_PROVIDERS)}."
            )
        self.provider           = provider
        self.model              = model
        self.openrouter_api_key = openrouter_api_key
        self.azure_api_key      = azure_api_key
        self.azure_endpoint     = azure_endpoint
        self.base_url           = base_url
        self.api_key            = api_key
        self._bedrock_client    = None  # lazy-init

    # ── public protocol ───────────────────────────────────────────────────────

    async def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
    ) -> str:
        """Dispatch async; the actual call runs in a thread so it never blocks the loop."""
        import asyncio
        return await asyncio.to_thread(
            self._sync_completion, system_prompt, user_prompt, max_tokens
        )

    def clean_and_parse_json(self, raw_json: str) -> Any:
        """Strip markdown fences and parse JSON — mirrors mcp-bench's own helper."""
        import re
        text = raw_json.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
        text = re.sub(r",\s*([}\]])",      r"\1", text)  # trailing commas
        return json.loads(text.strip())

    # ── internal dispatch ─────────────────────────────────────────────────────

    def _sync_completion(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if self.provider == "bedrock":
            return self._call_bedrock(system_prompt, user_prompt, max_tokens)
        elif self.provider == "openrouter":
            return self._call_openrouter(system_prompt, user_prompt, max_tokens)
        elif self.provider == "azure":
            return self._call_azure(system_prompt, user_prompt, max_tokens)
        elif self.provider == "base_url":
            return self._call_base_url(system_prompt, user_prompt, max_tokens)
        else:
            raise RuntimeError(
                f"[_MCPBenchLLMProvider] No handler for provider {self.provider!r}."
            )

    # ── Bedrock ───────────────────────────────────────────────────────────────

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            import boto3
            self._bedrock_client = boto3.client(service_name="bedrock-runtime")
        return self._bedrock_client

    def _call_bedrock(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        client   = self._get_bedrock_client()
        model_id = _BEDROCK_MODEL_ALIASES.get(self.model, self.model)
        response = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        text = ""
        for block in response["output"]["message"].get("content", []):
            if "text" in block:
                text += block["text"]
        return text

    # ── OpenRouter ────────────────────────────────────────────────────────────

    def _call_openrouter(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import requests
        if not self.openrouter_api_key:
            raise RuntimeError("[_MCPBenchLLMProvider] openrouter_api_key not set.")
        model = self.model if "/" in self.model else f"openai/{self.model}"
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
            json={
                "model":      model,
                "messages":   [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── Azure OpenAI ──────────────────────────────────────────────────────────

    def _call_azure(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import requests
        if not self.azure_api_key or not self.azure_endpoint:
            raise RuntimeError(
                "[_MCPBenchLLMProvider] azure_api_key and azure_endpoint are both required."
            )
        url = (
            f"{self.azure_endpoint}/openai/deployments/{self.model}"
            "/chat/completions?api-version=2024-02-01"
        )
        resp = requests.post(
            url,
            headers={"api-key": self.azure_api_key},
            json={
                "messages":   [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ── Generic OpenAI-compatible base_url (local Ollama / vLLM / LM Studio) ──

    def _call_base_url(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import requests
        if not self.base_url:
            raise RuntimeError(
                "[_MCPBenchLLMProvider] base_url must be set when provider='base_url'."
            )
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model":      self.model,
                "messages":   [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class MCPBenchAdapter(BenchmarkAdapter):
    def __init__(
        self,
        mcpbench_root: str = ".",
        # ── Query planner ─────────────────────────────────────────────────────
        provider: str = "openrouter",
        # Open-source via OpenRouter: "meta-llama/llama-3.3-70b-instruct"
        # Local Ollama:               "llama3.3:70b"  (+template_base_url)
        # AWS Bedrock:                "claude-sonnet-4" (short alias) or full ID
        template_model: Optional[str] = None,
        # OpenAI-compatible base URL for the planner only (e.g. Ollama)
        template_base_url: Optional[str] = None,
        template_api_key: Optional[str] = None,
        # ── LLM Judge ────────────────────────────────────────────────────────
        # Defaults to the same provider/model as the query planner when omitted.
        judge_provider: Optional[str] = None,
        judge_model: str = "o4-mini",
        # ── Shared credentials ────────────────────────────────────────────────
        openrouter_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        # ── Local / base_url credentials (Ollama, vLLM, LM Studio) ───────────
        local_base_url: Optional[str] = None,
        local_api_key: Optional[str] = None,
        # ── Misc ──────────────────────────────────────────────────────────────
        task_complexity: str = "all",
        timeout_s: float = 60.0,
        max_servers_per_template: int = 3,
        template_task_sample_n: int = 20,
        enable_judge_stability: bool = True,
    ):
        self.enable_judge_stability = enable_judge_stability
        self.root               = os.path.abspath(mcpbench_root)
        self.openrouter_api_key = openrouter_api_key
        self.azure_api_key      = azure_api_key
        self.azure_endpoint     = azure_endpoint
        self.local_base_url     = local_base_url
        self.local_api_key      = local_api_key
        self.task_complexity    = task_complexity
        self.timeout_s          = timeout_s
        self.max_servers_per_template = max_servers_per_template

        # Query-planner provider
        self.provider = provider

        self.template_model         = template_model or judge_model
        self.template_base_url      = template_base_url
        self.template_api_key       = template_api_key
        self.template_task_sample_n = template_task_sample_n

        # Judge — defaults to same provider/model as the query planner
        self.judge_provider = (judge_provider or provider).lower().strip()
        if self.judge_provider == "aws":
            self.judge_provider = "bedrock"
        self.judge_model = judge_model
        

        self._server_commands: Optional[Dict[str, Any]] = None
        self._tool_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._mcp_sessions: Dict[str, Any] = {}   # populated by _get_server_manager()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def map_server_to_config(self, servers_info: Dict[Any]) -> Optional[Dict[str, Any]]:
        """Map all single server name to actual server configuration.
        
        Server name should be the local server name (e.g., "National Parks", "DEX Paprika")
        Multi-server combinations should be handled by the caller.
        """
        
        # Direct lookup for local servers
        actual_server_info = {}
        for server_name, server_config in servers_info.items():
            cmd_parts = server_config.get('cmd', '').split(" ")
            
            # Use cwd path directly from commands.json
            cwd_path = server_config.get('cwd', '')
            
            if cwd_path.startswith('../'):
                # Handle relative path, convert to absolute path
                actual_cwd = cwd_path.replace("..",  f"{self.root}/mcp_servers")
            else:
                actual_cwd = cwd_path
            #print(actual_cwd)
            # Build environment variables
            env = {}
            for env_var in server_config.get('env', []):
                if env_var in _SERVER_TOOL_API_KEY:
                    env[env_var] = os.getenv(env_var)
            
            # Build base configuration
            server_config['name'] = server_name
            server_config['command'] = cmd_parts
            server_config['env'] = env
            server_config['cwd'] = actual_cwd    

            # Add HTTP configuration if this is an HTTP server
            if server_config.get('transport') == 'http':
                server_config['transport'] = 'http'
                server_config['port'] = server_config.get('port',None) 
                server_config['endpoint'] = server_config.get('endpoint', '/mcp')
            
            actual_server_info[server_name] = server_config 

        return actual_server_info        
    
    def _load_server_commands(self) -> Dict[str, Any]:
        """
        Merge commands.json (launch config) with mcp_servers_info.json (tool specs).

        commands.json  →  {server: {cmd, cwd, env, …}}          (how to start)
        mcp_servers_info.json  →  {servers: {server: {tools: {…}}}}  (what it exposes)

        After merging, every server entry has a "tools" key:
            {tool_name: {name, description, input_schema, output_schema}}
        which is what _discover_tools() and build_registry() consume.
        """
        if self._server_commands is not None:
            return self._server_commands

        cmds_path       = os.path.join(self.root, "mcp_servers", "commands.json")
        cmds_tools_path = os.path.join(self.root, "mcp_servers_info.json")

        if not os.path.exists(cmds_path) or not os.path.exists(cmds_tools_path):
            # Offline / CI: return an empty dict so build_registry() falls
            # through to its stub population and no further I/O is attempted.
            print(
                f"[mcpbench] commands.json or mcp_servers_info.json not found under "
                f"{self.root!r} — using offline stubs.",
                file=sys.stderr,
            )
            self._server_commands = {}
            return self._server_commands

        with open(cmds_path) as f:
            self._server_commands = json.load(f)

        with open(cmds_tools_path) as f:
            server_info = json.load(f)["servers"]

        # Graft tool specs from mcp_servers_info.json into the commands dict
        # so that _discover_tools() has a single source of truth.
        for server, details in server_info.items():
            if server in self._server_commands:
                self._server_commands[server]["tools"] = details["tools"]
            else:
                # Server appears in info but not in commands — add a stub entry
                # so tool discovery still works (call_tool will fail gracefully).
                self._server_commands[server] = {"tools": details["tools"]}
        self._server_commands = self.map_server_to_config(self._server_commands)
        return self._server_commands 

    def _discover_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Return {server_name: {tool_name: spec_dict}} from the merged config.

        spec_dict shape (mirrors mcp_servers_info.json):
            {
                "name":          str,
                "description":   str,
                "input_schema":  dict,   # JSON Schema
                "output_schema": dict,   # JSON Schema (may be absent)
            }
        """

        cmds = self._load_server_commands()
        tool_map: Dict[str, List[str]] = {}
        for server_name, cfg in cmds.items():
            # MCP-Bench commands.json lists tools under cfg["tools"] when
            # the static cache is present, or under cfg["tool_names"].
            tool_map[server_name] = cfg["tools"]
        self._tool_map = tool_map
        return self._tool_map 

    def _get_server_manager(self):
        """
        Return the live PersistentMultiServerManager, creating it on first call.

        The manager is stored in self._mcp_sessions["manager"].  On first access
        it is initialised synchronously by running asyncio event loop to call
        manager.initialize_servers(), which spawns all stdio subprocesses and
        populates manager.all_tools.

        The flat all_tools dict (shape: {tool_name: {"server": …, "description":
        …, "input_schema": …}}) is also cached in _mcp_sessions["all_tools"] so
        that call_tool() can look up server routing without touching the manager.

        If the MCP SDK or PersistentMultiServerManager is unavailable (offline /
        CI) the method returns None and call_tool() falls through to the stub.
        """
        if "manager" in self._mcp_sessions:
            return self._mcp_sessions["manager"]

        from ..mcp_modules.server_manager_persistent import (
            PersistentMultiServerManager as MultiServerManager,
        )

        # Build the config list that MultiServerManager expects.
        # TaskExecutor receives a MultiServerManager that was already initialised
        # by the benchmark runner — we replicate that initialisation here.
        cmds = self._load_server_commands()

        # Load API keys from mcp_servers/api_key into environment before starting
        key_file = os.path.join(self.root, "mcp_servers", "api_key")
        if os.path.exists(key_file):
            with open(key_file) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()

        manager = MultiServerManager(
            server_configs=cmds,
        )
        
        import asyncio
        import threading

        _loop  = asyncio.new_event_loop()
        _ready = threading.Event()
        _exc   = [None]

        def _run_loop():
            asyncio.set_event_loop(_loop)
            async def _init():
                try:
                    await manager.connect_all_servers()
                except Exception as e:
                    _exc[0] = e
                finally:
                    _ready.set()
            _loop.run_until_complete(_init())
            _loop.run_forever()   # keep loop alive for all call_tool calls

        _thread = threading.Thread(target=_run_loop, daemon=True, name="mcpbench-loop")
        _thread.start()
        _ready.wait(timeout=120)

        if _exc[0]:
            print(f"[mcpbench] manager.connect_all_servers() failed: {_exc[0]}", file=sys.stderr)
            return None

        # Cache manager, its loop, and the flat tool map
        self._mcp_sessions["manager"]   = manager
        self._mcp_sessions["loop"]      = _loop   # the loop the sessions live on
        self._mcp_sessions["all_tools"] = manager.all_tools
        print(
            f"[mcpbench] PersistentMultiServerManager ready — "
            f"{len(manager.all_tools)} tools across {len(cmds)} servers.",
            file=sys.stderr,
        )
        return manager

    # ── 1. ToolRegistry ───────────────────────────────────────────────────────

    def build_registry(self) -> ToolRegistry:
        tool_map = self._discover_tools()
        tools: Dict[Tuple[str, str], ToolSpec] = {}
        for server, discovered_tools in tool_map.items():
            for tool_name, raw_spec in discovered_tools.items():
                raw_name = raw_spec.get("name", tool_name)
                if raw_name != tool_name:
                    raise ValueError(
                        f"Tool name mismatch for server={server!r}: "
                        f"dict key={tool_name!r}, payload name={raw_name!r}"
                    )
                tools[(server, tool_name)] = ToolSpec(
                    server       = server,
                    name         = tool_name,
                    description  = raw_spec.get("description", ""),
                    input_schema = raw_spec["input_schema"],
                    output_schema= raw_spec.get("output_schema"),
                )
        return ToolRegistry(tools=tools)
    
    # ── 2. Query planner (per-query graph builder) ────────────────────────────

    def build_query_planner(self, registry: ToolRegistry):
        """
        Return a QueryPlanner that builds a GraphTemplate for each specific query.

        The planner receives the query text and the full tool catalog, calls the
        configured LLM once per unique query, and returns a validated GraphTemplate
        covering every tool call needed to complete that task.

        If no LLM credentials are configured, or all LLM attempts fail, the
        QueryPlanner falls back to a keyword-overlap heuristic that selects tools
        by name/description similarity to the query and chains them linearly.
        """
        from ..query_planner import QueryPlanner

        tool_catalog = self._build_tool_catalog(self._discover_tools())

        return QueryPlanner(
            registry           = registry,
            tool_catalog       = tool_catalog,
            provider           = self.provider, 
            openrouter_api_key = self.openrouter_api_key,
            azure_api_key      = self.azure_api_key,
            azure_endpoint     = self.azure_endpoint,
            model              = self.template_model,
            base_url           = self.template_base_url,
            api_key            = self.template_api_key,
            max_steps          = 6,
            max_retries        = 2,
        )

    def _build_tool_catalog(
        self,
        tool_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build {server: {tool_name: {"description": ..., "input_schema": ...}}}
        for the QueryPlanner.

        Including input_schema lets the LLM see parameter names/types and make
        better inferences about the output structure when writing produces paths.
        """
        catalog: Dict[str, Dict[str, Any]] = {}
        for server, tools in tool_map.items():
            if not tools:
                continue
            catalog[server] = {}
            for tool_name, spec in tools.items():
                if not isinstance(spec, dict):
                    catalog[server][tool_name] = {"description": ""}
                    continue
                entry: Dict[str, Any] = {
                    "description": spec.get("description", ""),
                }
                if spec.get("input_schema"):
                    entry["input_schema"] = spec["input_schema"]
                catalog[server][tool_name] = entry
        return catalog

    # ── 3. Datasets ───────────────────────────────────────────────────────────

    def load_datasets(self):
        tasks: List[Dict[str, Any]] = []
        keys = ["single", "2server", "3server"] if self.task_complexity == "all" else [self.task_complexity]
        for key in keys:
            rel_path = _DEFAULT_TASK_FILES[key]
            full_path = os.path.join(self.root, rel_path)
            if not os.path.exists(full_path):
                continue
            with open(full_path) as f:
                raw = json.load(f)
            # MCP-Bench task format: list of dicts with "task_id", "task",
            # "servers", "ground_truth" (optional), "tools_required" (optional)
            for server_item in raw.get("server_tasks", []):
                for item in server_item.get("tasks", []):
                    tasks.append({
                        "id":            item.get("task_id", item.get("id", len(tasks))),
                        "text":          item.get("task_description", item.get("question", "")),
                        "fuzzy_description": item.get("fuzzy_description", []),
                        "distraction_servers": item.get("distraction_servers", []),
                        "dependency_analysis": item.get("dependency_analysis", []),
                        "servers":       server_item.get("servers", []),
                        "server_name":       server_item.get("server_name", []),
                        "combination_type":       server_item.get("combination_type", []),

                        "ground_truth":  item.get("dependency_analysis", None),
                        "complexity":    key,
                        # Template hint: pick the matching template by server names
                        "_servers":     server_item.get("servers", []),
                    })
        assert len(tasks) != 0, "MCPBench data not loaded..."
        rng = random.Random(0)

        # Group by complexity
        buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for task in tasks:
            buckets[task["complexity"]].append(task)

        # Shuffle each bucket
        for bucket in buckets.values():
            rng.shuffle(bucket)

        split_idx = 0.6

        train_tasks: List[Dict[str, Any]] = []
        test_tasks: List[Dict[str, Any]] = []

        for _, bucket in buckets.items():
            n = len(bucket)
            cut = int(split_idx * n)
            train_tasks.extend(bucket[:cut])
            test_tasks.extend(bucket[cut:])

        rng.shuffle(train_tasks)
        rng.shuffle(test_tasks)

        return train_tasks, test_tasks

    # ── 4. Tool execution via MCP protocol ────────────────────────────────────

    def call_tool(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Execute one tool call, preferring the persistent server manager.

        Fast path — PersistentMultiServerManager (matching TaskExecutor):
            manager = self._get_server_manager()
            result  = await manager.call_tool(tool_name, params)

        The manager uses the tool_name directly (no server prefix needed —
        all_tools is keyed by bare tool name and the manager resolves the
        server internally).  The result is a CallToolResult with .isError
        and .content (list of TextContent items with .text).

        Slow path — per-call stdio transport (same as original _async_call_tool):
        Used when the manager is unavailable (offline, no mcp package, CI).
        Each invocation spawns the server subprocess, runs the tool, and
        tears down — correct but ~2× slower than the persistent sessions.
        """
            
        import asyncio

        manager = self._get_server_manager()
        if manager is not None:
            # ── Fast path: dispatch onto the manager's persistent loop ─────
            loop = self._mcp_sessions.get("loop")
            if loop is None or loop.is_closed():
                print("[mcpbench] manager loop unavailable — evicting.", file=sys.stderr)
                self._mcp_sessions.pop("manager", None)
                self._mcp_sessions.pop("loop", None)
                raise RuntimeError(f"[mcpbench] manager.call_tool({server}:{tool}) failed.")

            async def _run():
                print(f"[mcpbench] calling {server}:{tool}  params={args}", file=sys.stderr)
                print(f"[mcpbench] calling {server}:{tool}  params={args}")
                result_obj = await manager.call_tool(f"{server}:{tool}", args)
                #print("HERERERE MCPBENCHADAPTER: ", result_obj, flush=True)
                is_error = getattr(result_obj, "isError", False) or False
                if hasattr(result_obj, "content") and result_obj.content:
                    text = "".join(
                        item.text for item in result_obj.content if hasattr(item, "text")
                    )
                else:

                    text = str(result_obj)
                # Some MCP servers return errors as plain content without setting
                # isError=True (e.g. HTTP 4xx/5xx wrapped in a success response).
                # Detect common error prefixes so the step is marked ok=False.
                if not is_error and text:
                    stripped = text.strip()
                    if (stripped.lower().startswith("error")
                            or stripped.startswith("Exception")
                            or stripped.startswith("Traceback")):
                        is_error = True
                #print("RESULT OBJECT:",text, flush=True)
                return is_error, text

            try:
                future = asyncio.run_coroutine_threadsafe(_run(), loop)
                is_error, text = future.result(timeout=self.timeout_s)
            except Exception as exc:
                text = f"[mcpbench] ERROR in calling tool {tool!r} on persistent session: {exc}"
                print(
                    text,
                    file=sys.stderr,
                )
                #self._mcp_sessions.pop("manager", None)
                #self._mcp_sessions.pop("loop", None)
                #raise RuntimeError(f"[mcpbench] manager.call_tool({server}:{tool}) failed.") from exc
                return False, text, None 

            if is_error:
                return False, text, None
            return True, None, {"content": text}


    # ── 5. LLM judge scoring ──────────────────────────────────────────────────

    def score_result(self, res: ExecResult, wf: CompiledWorkflow) -> MetricBreakdown:
        """
        Score an ExecResult using mcp-bench's native LLMJudge (evaluator.py).
        """
        #if not res.ok or res.final_output is None:
        #    return None
 
        # Bedrock uses the boto3 credential chain — no explicit API key needed.
        # base_url (local) only needs base_url to be set, not an API key.
        if self.judge_provider not in ("bedrock", "base_url"):
            key = self.openrouter_api_key or self.azure_api_key
            if not key:
                return None
        if self.judge_provider == "base_url" and not self.local_base_url:
            return None

        max_attempts = 3
        last_result = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = self._score_with_llm_judge(res, wf)
            except Exception as exc:
                exc_str = str(exc)
                if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                    raise
                print(f"[mcpbench] score_result via LLMJudge failed (attempt {attempt}/{max_attempts}): {exc!r}", file=sys.stderr)
                print(f"[mcpbench] score_result via LLMJudge failed (attempt {attempt}/{max_attempts}): {exc!r}", flush=True)
                if attempt < max_attempts:
                    import time
                    time.sleep(2)
                continue

            if result is None:
                print(f"[mcpbench] score_result via LLMJudge returned None (attempt {attempt}/{max_attempts})", flush=True)
            elif (result.task_fulfillment == 0.0 and result.grounding == 0.0
                  and result.tool_appropriateness == 0.0 and result.parameter_accuracy == 0.0):
                print(f"[mcpbench] score_result via LLMJudge returned all-zero scores (attempt {attempt}/{max_attempts})", flush=True)
                last_result = result
            else:
                return result

            if attempt < max_attempts:
                import time
                time.sleep(5)

        if last_result is not None:
            return last_result

        return MetricBreakdown(
                task_fulfillment = 0.0,
                grounding = 0.0,
                tool_appropriateness  = 0.0,
                parameter_accuracy        =  0.0,
                dependency_awareness     = 0.0,
                parallelism_and_efficiency     = 0.0,
                
                task_completion_score     =  0.0,
                tool_selection_score     = 0.0,
                planning_effectiveness_and_efficiency_score     = 0.0,
                
                task_fulfillment_reasoning     = "",
                grounding_reasoning     = "",
                tool_appropriateness_reasoning     = "",
                parameter_accuracy_reasoning     = "",
                dependency_awareness_reasoning     = "",
                parallelism_and_efficiency_reasoning     =  "",
                #latency_ms      = float(sum(log.latency_ms for log in res.logs)),
            )
        
    def _score_with_llm_judge(self, res: ExecResult, wf: CompiledWorkflow) -> Optional[MetricBreakdown]:
        """
        Internal: run mcp-bench's LLMJudge and convert scores to MetricBreakdown.
        Raises on any failure so score_result() can catch and return None.
        """
        import sys as _sys
        _sys.path.insert(0, self.root)
 
        from benchmark.evaluator import LLMJudge            # mcp-bench/benchmark/evaluator.py
 
        # ── 1. LLMProvider bridge ─────────────────────────────────────────────
        provider = _MCPBenchLLMProvider(
            provider           = self.judge_provider,
            model              = self.judge_model,
            openrouter_api_key = self.openrouter_api_key,
            azure_api_key      = self.azure_api_key,
            azure_endpoint     = self.azure_endpoint,
            base_url           = self.local_base_url,
            api_key            = self.local_api_key,
        )
 
        judge = LLMJudge(llm_provider=provider, enable_judge_stability=self.enable_judge_stability)
 
        # ── 2. Build execution_results list ───────────────────────────────────
        # LLMJudge expects: [{tool, success, parameters, server, output}, ...]
        execution_count = 0
        execution_results = []
        for log in res.logs:
            if log.kind != "tool":
                continue
            # tool_ref is stored as "server.tool" in StepLog.tool
            server, _, tool_name = (log.tool or "").partition(".")
            execution_results.append({
                "tool":       tool_name or log.tool,
                "server":     server or "",
                "success":    log.ok,
                "parameters": log.input_args or {},
                "output":     log.output,
                "error":      log.error_type,
            })
            execution_count += 1 if log.ok else 0
 
        # ── 3. Build available_tools flat dict ────────────────────────────────
        # Shape: {tool_name: {server, description, input_schema}}
        available_tools: dict = {}
        # Prefer the live all_tools from the manager (has descriptions)
        all_tools_cache = self._mcp_sessions.get("all_tools", {})
        if all_tools_cache:
            available_tools = dict(all_tools_cache)
        elif self._tool_map:
            for server_name, tools in self._tool_map.items():
                for tool_name, spec in tools.items():
                    spec_dict = spec if isinstance(spec, dict) else {}
                    available_tools[tool_name] = {
                        "server":       server_name,
                        "description":  spec_dict.get("description", ""),
                        "input_schema": spec_dict.get("input_schema", {}),
                    }
        '''print("[mcpbench/_score_with_llm_judge] COMPILED WORKFLOW: ")
        print(wf) 
        print("***"*100, flush=True)
        print("[mcpbench/_score_with_llm_judge] OUTPUT: ")
        print(res) 
        print("***"*100, flush=True)
        print("===="*100, flush=True)
        '''
        # ── 4. Build task text and metadata from final_output / query ─────────
        final_output = res.final_output
        task_text    = wf.query.get("text")
        dependency_analysis =  wf.query.get("dependency_analysis") 
        final_solution = str(final_output.get("content", final_output)) if isinstance(final_output, dict) else str(final_output)
        total_rounds = len(execution_results)
 
        # Accumulated information: full trajectory text for the judge
        accumulated_info = "\n".join(
            f"Round {i+1}: tool={r['tool']} server={r['server']} "
            f"ok={r['success']} output={str(r.get('output',''))[:300]}"
            for i, r in enumerate(execution_results)
        )
 
        # ── 5. Run async judge on the persistent loop ─────────────────────────
        # Stability mode runs 5 LLM calls; budget accordingly.
        judge_timeout = 600.0 if self.enable_judge_stability else 300.0
        loop = self._mcp_sessions.get("loop")
        if loop is not None and not loop.is_closed():
            import asyncio
            future = asyncio.run_coroutine_threadsafe(
                judge.evaluate_task_performance(
                    task                    = task_text,
                    final_solution          = final_solution,
                    execution_results       = execution_results,
                    total_rounds            = total_rounds,
                    available_tools         = available_tools,
                    accumulated_information = accumulated_info,
                    dependency_analysis     = dependency_analysis
                ),
                loop,
            )
            scores = future.result(timeout=judge_timeout)
        else:
            import asyncio
            scores = asyncio.run(
                judge.evaluate_task_performance(
                    task                    = task_text,
                    final_solution          = final_solution,
                    execution_results       = execution_results,
                    total_rounds            = total_rounds,
                    available_tools         = available_tools,
                    accumulated_information = accumulated_info,
                    dependency_analysis     = dependency_analysis
                )
            )
 
        if scores is None:
            print(f"[score LLMJudge] Error. The score is None.",flush=True)
            return None
        return MetricBreakdown(
            task_fulfillment = float(scores.get("task_fulfillment",0.0)),
            grounding = float(scores.get("grounding",0.0)),
            tool_appropriateness  = float(scores.get("tool_appropriateness",0.0)),
            parameter_accuracy        = float(scores.get("parameter_accuracy", 0.0)),
            dependency_awareness     = float(scores.get("dependency_awareness",0.0)),
            parallelism_and_efficiency     = float(scores.get("parallelism_and_efficiency",0.0)),
            
            task_completion_score     = float(scores.get("task_completion_score",0.0)),
            tool_selection_score     = float(scores.get("tool_selection_score",0.0)),
            planning_effectiveness_and_efficiency_score     = float(scores.get("planning_effectiveness_and_efficiency_score",0.0)),
            
            task_fulfillment_reasoning     = str(scores.get("task_fulfillment_reasoning","")),
            grounding_reasoning     = str(scores.get("grounding_reasoning","")),
            tool_appropriateness_reasoning     = str(scores.get("tool_appropriateness_reasoning","")),
            parameter_accuracy_reasoning     = str(scores.get("parameter_accuracy_reasoning","")),
            dependency_awareness_reasoning     = str(scores.get("dependency_awareness_reasoning","")),
            parallelism_and_efficiency_reasoning     = str(scores.get("parallelism_and_efficiency_reasoning","")),
            #latency_ms      = float(sum(log.latency_ms for log in res.logs)),
        )