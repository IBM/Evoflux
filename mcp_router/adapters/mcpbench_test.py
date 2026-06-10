from __future__ import annotations

import json
import sys, os, shlex
import random
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark_adapter import BenchmarkAdapter
from ..dsl import ToolSpec, ToolRegistry, GraphTemplate, Node, Action, PromptParams
from ..executor import ExecResult
from ..compiler import CompiledWorkflow
from ..metrics import MetricBreakdown, compute_metrics


"""
mcp_router/adapters/mcpbench.py
--------------------------------
BenchmarkAdapter for Accenture MCP-Bench.
https://github.com/Accenture/mcp-bench

MCP-Bench has 28 MCP servers accessed via the MCP protocol (stdio/SSE).
Tasks come in three flavours: single-server, 2-server, 3-server.
Scoring uses an LLM judge (o4-mini by default).

Setup:
    git clone https://github.com/Accenture/mcp-bench
    cd mcp-bench
    conda create -n mcpbench python=3.10 && conda activate mcpbench
    cd mcp_servers && bash ./install.sh && cd ..
    # set API keys in mcp_servers/api_key and .env

    pip install mcp   # MCP Python SDK

    from mcp_router.adapters.mcpbench import MCPBenchAdapter
    from mcp_router.benchmark_adapter import BenchmarkAdapterRunner, RunnerConfig

    adapter = MCPBenchAdapter(
        mcpbench_root="/path/to/mcp-bench",
        judge_model="o4-mini",
        openrouter_api_key="sk-...",
    )
    runner = BenchmarkAdapterRunner(adapter=adapter, cfg=RunnerConfig(B=60, K=16))
    pop = runner.run_search()
"""


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_TASK_FILES = {
    "single":  "tasks/mcpbench_tasks_single_runner_format.json",
    "2server": "tasks/mcpbench_tasks_multi_2server_runner_format.json",
    "3server": "tasks/mcpbench_tasks_multi_3server_runner_format.json",
}


class MCPBenchAdapter(BenchmarkAdapter):
    """
    Connects mcp_router to the Accenture MCP-Bench framework.

    The adapter is intentionally lazy: MCP server subprocesses are only
    started when call_tool is first invoked for that server, and are
    cached for the lifetime of the adapter instance.
    """

    def __init__(
        self,
        mcpbench_root: str = ".",
        judge_model: str = "o4-mini",
        openrouter_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        task_complexity: str = "all",   # "single" | "2server" | "3server" | "all"
        timeout_s: float = 60.0,
        max_servers_per_template: int = 3,
    ):
        self.root = os.path.abspath(mcpbench_root)
        self.judge_model = judge_model
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.azure_api_key      = azure_api_key      or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint     = azure_endpoint     or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.task_complexity    = task_complexity
        self.timeout_s          = timeout_s
        self.max_servers_per_template = max_servers_per_template

        # Lazily populated
        self._server_commands: Optional[Dict[str, Any]] = None
        self._tool_map: Dict = None   # server → [tool_names]
        self._mcp_sessions: Dict[str, Any] = {}                  # server → live MCP session

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_server_commands(self) -> Dict[str, Any]:
        if self._server_commands is None:
            cmds_path = os.path.join(self.root, "mcp_servers", "commands.json")
            cmds_tools_path = os.path.join(self.root,"mcp_servers_info.json")

            assert os.path.exists(cmds_path) and os.path.exists(cmds_tools_path)
            with open(cmds_path) as f:
                self._server_commands = json.load(f)
            with open(cmds_tools_path) as f:
                _server_tools_commands = json.load(f)["servers"]
            for server, server_details in _server_tools_commands.items():
               self._server_commands[server]["tools"] = server_details["tools"] 

        return self._server_commands

    def _discover_tools(self) -> Dict[str, List[str]]:
        """
        Read tool names per server from commands.json.
        In production this would call collect_mcp_info.py; here we read
        the static cache that MCP-Bench ships with.
        """
        if self._tool_map is not None:
            return self._tool_map

        cmds = self._load_server_commands()
        tool_map: Dict[str, List[str]] = {}
        for server_name, cfg in cmds.items():
            # MCP-Bench commands.json lists tools under cfg["tools"] when
            # the static cache is present, or under cfg["tool_names"].
            tool_map[server_name] = cfg["tools"]
        self._tool_map = tool_map
        return tool_map

    # ------------------------------------------------------------------
    # 1. ToolRegistry
    # ------------------------------------------------------------------
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

                known_keys = {"name", "description", "input_schema", "output_schema"}
                metadata = {k: v for k, v in raw_spec.items() if k not in known_keys}
                tools[(server, tool_name)] = ToolSpec(
                    server=server,
                    name=tool_name,
                    description=raw_spec.get("description", ""),
                    input_schema=raw_spec["input_schema"],
                    output_schema=raw_spec.get("output_schema"),
                    metadata=metadata,
                )

        return ToolRegistry(tools=tools)

    # ------------------------------------------------------------------
    # 2. Templates
    # ------------------------------------------------------------------
    def build_templates(self) -> Dict[str, GraphTemplate]:
        """
        Generate one template per server-complexity level.
        Single-server: t1 → t2 (two sequential calls to same server).
        2-server:      t1(srv_a) → t2(srv_b).
        3-server:      t1(srv_a) → t2(srv_b) → t3(srv_c).
        We pick representative servers from the registry.
        """
        tool_map = self._discover_tools()
        servers = [server for server, tools in tool_map.items() if tools]

        def tool_names_for(server: str) -> List[str]:
            server_tools = tool_map[server]
            return list(server_tools.keys())

        def first_tool(server: str) -> str:
            names = tool_names_for(server)
            if not names:
                raise ValueError(f"No tools found for server {server!r}")
            return names[0]

        def second_tool_or_first(server: str) -> str:
            names = tool_names_for(server)
            if not names:
                raise ValueError(f"No tools found for server {server!r}")
            return names[1] if len(names) > 1 else names[0]

        templates: Dict[str, GraphTemplate] = {}

        # Single-server templates
        for server in servers[: self.max_servers_per_template]:
            t1 = first_tool(server)
            t2 = second_tool_or_first(server)
            tid = f"single_{server}"

            templates[tid] = GraphTemplate(
                template_id=tid,
                nodes=[
                    Node("t1", "tool", (server, t1)),
                    Node("t2", "tool", (server, t2), requires=["t1"]),
                ],
                edges=[("t1", "t2")],
            )

        # Two-server templates
        for i in range(min(self.max_servers_per_template, max(0, len(servers) - 1))):
            sA, sB = servers[i], servers[i + 1]
            tid = f"multi2_{sA}_{sB}"

            templates[tid] = GraphTemplate(
                template_id=tid,
                nodes=[
                    Node("t1", "tool", (sA, first_tool(sA))),
                    Node("t2", "tool", (sB, first_tool(sB)), requires=["t1"]),
                ],
                edges=[("t1", "t2")],
            )

        # Three-server templates
        for i in range(min(self.max_servers_per_template, max(0, len(servers) - 2))):
            sA, sB, sC = servers[i], servers[i + 1], servers[i + 2]
            tid = f"multi3_{sA}_{sB}_{sC}"

            templates[tid] = GraphTemplate(
                template_id=tid,
                nodes=[
                    Node("t1", "tool", (sA, first_tool(sA))),
                    Node("t2", "tool", (sB, first_tool(sB)), requires=["t1"]),
                    Node("t3", "tool", (sC, first_tool(sC)), requires=["t2"]),
                ],
                edges=[("t1", "t2"), ("t2", "t3")],
            )

        if not templates:
            templates["fallback_single"] = GraphTemplate(
                template_id="fallback_single",
                nodes=[
                    Node("t1", "tool", ("OpenAPI Explorer", "query")),
                    Node("t2", "tool", ("Scientific Computing", "calculate"), requires=["t1"]),
                ],
                edges=[("t1", "t2")],
            )
        return templates

    # ------------------------------------------------------------------
    # 3. Datasets
    # ------------------------------------------------------------------
    def load_datasets(self):
        tasks: List[Dict[str, Any]] = []

        if self.task_complexity == "all":
            keys = ["single", "2server", "3server"]
        else:
            keys = [self.task_complexity]

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

                        "ground_truth":  item.get("ground_truth", None),
                        "complexity":    key,
                        # Template hint: pick the matching template by server names
                        "_servers":     server_item.get("servers", []),
                    })
        assert len(tasks) != 0, "MCPBench data not loaded... "

        rng = random.Random(0)
        rng.shuffle(tasks)
        n = len(tasks)
        return tasks[:int(0.6*n)], tasks[int(0.6*n):int(0.8*n)], tasks[int(0.8*n):]

    # ------------------------------------------------------------------
    # 4. Tool execution via MCP protocol
    # ------------------------------------------------------------------
    def call_tool(
        self, server: str, tool: str, args: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Call a tool on the named MCP server.
        Uses the MCP Python SDK (stdio transport) to communicate with the
        server subprocess defined in commands.json.
        Falls back gracefully if the SDK or server is unavailable.
        """
        try:
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                self._async_call_tool(server, tool, args)
            )
            return result
        except Exception as exc:
            print(f"Exception call_tool(): call_error:{exc}", file=sys.stderr)
            return False, f"call_error:{exc}", None

    async def _async_call_tool(
        self, server: str, tool: str, args: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            print(
                "Import Error _async_call_tool(): MCP SDK not installed → stub response (CI / offline mode)",
                file=sys.stderr,
            )
            return True, None, {"stub": True, "server": server, "tool": tool}

        cmds = self._load_server_commands()
        if server not in cmds:
            return False, f"server_not_in_commands:{server}", None

        cfg = cmds[server]

        # cfg is like:
        # {'cmd': 'node index.js run', 'env': [], 'cwd': '../openapi-mcp-server', 'tools': {...}}

        # 1) Parse cfg["cmd"] into command + args
        cmd_str = cfg.get("cmd", "python")
        parts = shlex.split(cmd_str)
        command, cmd_args = parts[0], parts[1:]

        # 2) Build env dict: os.environ + cfg["env"] (supports list or dict)
        env = dict(os.environ)
        raw_env = cfg.get("env", {})

        if isinstance(raw_env, dict):
            env.update({k: str(v) for k, v in raw_env.items()})
        elif isinstance(raw_env, list):
            # Accept ["K=V", ...] OR [{"name":"K","value":"V"}, ...]
            for item in raw_env:
                if isinstance(item, str) and "=" in item:
                    k, v = item.split("=", 1)
                    env[k.strip()] = v.strip()
                elif isinstance(item, dict) and "name" in item and "value" in item:
                    env[str(item["name"])] = str(item["value"])
                else:
                    return False, f"bad_cfg_env_item:{item!r}", None
        else:
            return False, f"bad_cfg_env_type:{type(raw_env).__name__}", None

        # 3) Load API keys from mcp_servers/api_key
        key_file = os.path.join(self.root, "mcp_servers", "api_key")
        if os.path.exists(key_file):
            with open(key_file) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()

            
        cwd_path = self.root+cfg.get("cwd").replace("..", "/mcp_servers")
        # 4) Pass cwd through (if MCP SDK supports it, this is the correct spot)
        params = StdioServerParameters(
            command=command,
            args=cmd_args,
            env=env,
            cwd=cwd_path,
        )
        spec = self.build_registry().tools[(server, tool)]
        required = spec.input_schema
        print((server, tool))
        print(required)
        print("+++"*100)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # IMPORTANT: do not hardcode "task" unless your MCP server expects it.
                    # Prefer passing args through as-is (or args["query"] if that’s your convention).
                    tool_args = args
                    #print("HERERE TOOLS CALLING!!!!")
                    #print("TOOL",tool)
                    #print("TOOL ARGUMENT CALLING:",tool_args)
                    res = await session.call_tool(tool, tool_args)
                    #print(res)
                    return True, None, {"content": str(res.content)}
        except Exception as exc:
            print("\t Execution Error _async_call_tool():", exc, file=sys.stderr)
            return False, f"{type(exc).__name__}:{exc}", None

    # ------------------------------------------------------------------
    # 5. LLM judge scoring
    # ------------------------------------------------------------------
    def score_result(self, res: ExecResult, wf: CompiledWorkflow) -> Optional[MetricBreakdown]:
        """
        Mirror MCP-Bench's benchmark/evaluator.py LLM-judge scoring.
        Requires OPENROUTER_API_KEY or AZURE_OPENAI_API_KEY.
        """
        if not res.ok or res.final_output is None:
            return None

        key  = self.openrouter_api_key or self.azure_api_key
        if not key:
            return None

        try:
            import requests

            # Build the judge prompt the same way MCP-Bench's evaluator does
            trajectory_text = "\n".join(
                f"Step {i+1}: tool={l.tool or 'n/a'} ok={l.ok} "
                f"output={str(l.output)[:200]}"
                for i, l in enumerate(res.logs)
            )
            prompt = (
                "You are an impartial evaluator. Score the following agent trajectory "
                "on these dimensions (each 0.0–1.0):\n"
                "- task_completion: Did the agent complete the task?\n"
                "- tool_usage: Were tools called correctly and efficiently?\n"
                "- planning: Was the execution plan sensible?\n"
                "- schema_adherence: Were tool inputs/outputs well-formed?\n\n"
                f"Trajectory:\n{trajectory_text}\n\n"
                f"Final output:\n{str(res.final_output)[:500]}\n\n"
                "Respond ONLY with valid JSON: "
                "{\"task_completion\":…,\"tool_usage\":…,\"planning\":…,\"schema_adherence\":…}"
            )

            if self.openrouter_api_key:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
                    json={
                        "model": f"openai/{self.judge_model}",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                    },
                    timeout=30.0,
                )
            else:
                resp = requests.post(
                    f"{self.azure_endpoint}/openai/deployments/{self.judge_model}"
                    "/chat/completions?api-version=2024-02-01",
                    headers={"api-key": self.azure_api_key},
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                    },
                    timeout=30.0,
                )

            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

            # Strip markdown fences if present
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            scores = json.loads(text)

            # Map to MetricBreakdown
            tc = float(scores.get("task_completion", 0.5))
            tu = float(scores.get("tool_usage", 0.5))
            sa = float(scores.get("schema_adherence", 0.5))
            return MetricBreakdown(
                success=tc,
                runtime_ok=1.0 if res.ok else 0.0,
                schema_adherence=sa,
                dependency_compliance=tu,
                violation_rate=0.0,
                latency_ms=float(sum(l.latency_ms for l in res.logs)),
            )
        except Exception:
            print("Exception score_result(): Error", file=sys.stderr)
            return None   # fall back to structural proxy

    # ------------------------------------------------------------------
    # 6. Template-aware initial actions
    # ------------------------------------------------------------------
    def build_init_actions(self, D_fb, templates):
        """
        For each task in D_fb, find a template whose servers match the
        task's required servers. Seed one base Action per matched template.
        """
        def _best_template(servers: List[str]) -> str:
            n = len(servers)
            # Prefer templates that explicitly mention the required servers
            for tid, tpl in templates.items():
                tpl_servers = {n.tool_ref[0] for n in tpl.nodes if n.tool_ref}
                if set(servers[:n]) == tpl_servers:
                    return tid
            # Fall back by complexity
            if n >= 3:
                candidates = [t for t in templates if t.startswith("multi3")]
            elif n == 2:
                candidates = [t for t in templates if t.startswith("multi2")]
            else:
                candidates = [t for t in templates if t.startswith("single")]
            return candidates[0] if candidates else list(templates.keys())[0]

        seen_templates = set()
        for q in D_fb:
            tid = _best_template(q.get("_servers", []))
            seen_templates.add(tid)

        actions = []
        for tid in seen_templates:
            actions.append(Action(g=tid, e=(), p=PromptParams(max_steps=8)))
            actions.append(Action(g=tid, e=(), p=PromptParams(max_steps=6)))
            actions.append(Action(g=tid, e=(), p=PromptParams(max_steps=8, strict_schema=False)))

        return actions or [
            Action(g=list(templates.keys())[0], e=(), p=PromptParams(max_steps=8))
        ]