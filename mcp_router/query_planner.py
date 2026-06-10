"""
mcp_router/query_planner.py
---------------------------
QueryPlanner: builds a GraphTemplate for ONE specific query.

Called by Compiler._get_base_plan(query).  The planner sends the query
text and the tool catalog to an LLM, asks for the minimal ordered set of
tool calls needed to complete that specific task, and converts the
response into a validated GraphTemplate.

This is the counterpart of the old TemplatePlanner — but instead of
producing reusable skeletons for a batch of task types, it produces a
single executable graph for a single query.

Offline / no-credentials fallback
----------------------------------
When no LLM credentials are configured, QueryPlanner falls back to a
simple heuristic: it picks the top-N tools from the registry based on
name/description overlap with the query text and chains them linearly.
This is enough for CI testing and toy benchmarks.

Bedrock support
---------------
When aws_region is set (or AWS_REGION / AWS_DEFAULT_REGION env vars exist),
the planner uses boto3's bedrock-runtime Converse API. Authentication
flows through the standard boto3 credential chain (env vars, IAM role,
SSO, etc.). Set bedrock_model_id to the full Bedrock model identifier,
e.g. "us.anthropic.claude-sonnet-4-20250514-v1:0".
"""
from __future__ import annotations
import sys
import hashlib
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import copy
from .dsl import GraphTemplate, Node, ToolRegistry


# ---------------------------------------------------------------------------
# Bedrock model ID lookup
# ---------------------------------------------------------------------------
# Maps short aliases to full Bedrock model identifiers.
# Uses cross-region inference profile IDs (us. prefix) by default.
# Override with bedrock_model_id for bare IDs or other region prefixes.

BEDROCK_MODEL_ALIASES: Dict[str, str] = {
    "claude-sonnet-4":     "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4":       "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4.5":    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4.5":   "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4.6":     "us.anthropic.claude-opus-4-6-v1:0",
    "claude-sonnet-4.6":   "us.anthropic.claude-sonnet-4-6-v1:0",
}


def _resolve_bedrock_model_id(model: Optional[str]) -> str:
    """
    Resolve a model string to a Bedrock model ID.

    Accepts full Bedrock IDs (passed through unchanged), short aliases
    from BEDROCK_MODEL_ALIASES, or falls back to Claude Sonnet 4.
    """
    if model is None:
        return BEDROCK_MODEL_ALIASES["claude-sonnet-4"]
    # Already looks like a full Bedrock ID
    if "anthropic." in model:
        return model
    return BEDROCK_MODEL_ALIASES.get(model, model)


class QueryPlanner:
    """
    Produces a query-specific GraphTemplate by calling an LLM.

    Parameters
    ----------
    registry : ToolRegistry
        Used to validate every (server, tool) pair in the LLM response.
    tool_catalog : Dict[str, Dict[str, Any]]
        {server: {tool_name: {description, input_schema, ...}}}
        Passed to the LLM so it knows what tools exist and what they do.
    provider : str
        Which LLM backend to use for every call. Must be one of
        "bedrock", "openrouter", "azure", or "base_url".
        Eliminates ambiguity when multiple credential sets are present.
        Defaults to "bedrock".
    openrouter_api_key / azure_api_key / azure_endpoint / base_url :
        LLM credentials for the corresponding provider.
    bedrock_model_id : str or None
        Full Bedrock model identifier, or a short alias
        (see BEDROCK_MODEL_ALIASES).
    model : str
        LLM model identifier (for non-Bedrock providers).
    max_steps : int
        Maximum nodes the LLM may propose.  Keeps graphs tractable.
    max_retries : int
        How many times to re-prompt on parse failure.
    """

    VALID_PROVIDERS = {"bedrock", "openrouter", "azure", "base_url"}

    def __init__(
        self,
        registry: ToolRegistry,
        tool_catalog: Dict[str, Dict[str, Any]],
        provider: str = "bedrock",
        openrouter_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None, # bedrock_model_id model
        # ──────────────────────
        max_steps: int = 6,
        max_retries: int = 2,
    ):
        # ── Validate provider ─────────────────────────────────────────
        provider = provider.lower().strip()
        if provider == "aws":
            provider = "bedrock"      # treat "aws" as an alias
        if provider not in self.VALID_PROVIDERS:
            raise ValueError(
                f"Unknown provider {provider!r}. "
                f"Choose from {sorted(self.VALID_PROVIDERS)}"
            )
        self.provider = provider

        self.registry           = registry
        self.tool_catalog       = tool_catalog
        self.openrouter_api_key = openrouter_api_key
        self.azure_api_key      = azure_api_key
        self.azure_endpoint     = azure_endpoint
        self.base_url           = base_url
        self.api_key            = api_key
        self.model              = model
        self.max_steps          = max_steps
        self.max_retries        = max_retries

        # Bedrock config
        self._bedrock_client = None  # lazy-init

        # Verify the chosen provider has the credentials it needs
        self._has_credentials = self._check_provider_credentials()

    # ------------------------------------------------------------------
    # Provider credential check
    # ------------------------------------------------------------------

    def _check_provider_credentials(self) -> bool:
        """
        Return True if the selected provider has enough configuration
        to make an LLM call. Prints a warning (not an exception) when
        credentials look incomplete so the heuristic fallback still works.
        """
        checks = {
            "bedrock":    lambda: bool(self.model),
            "openrouter": lambda: bool(self.openrouter_api_key),
            "azure":      lambda: bool(self.azure_api_key and self.azure_endpoint),
            "base_url":   lambda: bool(self.base_url),
        }
        ok = checks[self.provider]()
        if not ok:
            print(
                f"[query_planner] WARNING provider={self.provider!r} selected "
                f"but required credentials are missing. "
                f"Falling back to heuristic planner.",
                file=sys.stderr,
            )
        return ok

    # ------------------------------------------------------------------
    # Unified LLM dispatch
    # ------------------------------------------------------------------

    def _dispatch_llm_call(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Route an LLM request to the provider selected at init time.

        Every call site (_call_llm, _call_llm_raw) goes through here,
        so there is exactly one place that decides which backend to hit.
        """
        if self.provider == "bedrock":
            return self._call_bedrock_converse(
                system_text, user_text, max_tokens, temperature,
            )
        elif self.provider == "openrouter":
            return self._call_openrouter(
                system_text, user_text, max_tokens, temperature,
            )
        elif self.provider == "azure":
            return self._call_azure(
                system_text, user_text, max_tokens, temperature,
            )
        elif self.provider == "base_url":
            return self._call_base_url(
                system_text, user_text, max_tokens, temperature,
            )
        else:
            raise RuntimeError(f"Unknown provider {self.provider!r}")

    # ------------------------------------------------------------------
    # Lazy Bedrock client
    # ------------------------------------------------------------------

    def _get_bedrock_client(self):
        """
        Build and cache a boto3 bedrock-runtime client.

        Authentication piggybacks on the standard boto3 credential chain,
        so AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
        (or an IAM role, SSO session, etc.) all work transparently.
        """
        if self._bedrock_client is None:
            import boto3
            self._bedrock_client = boto3.client(service_name="bedrock-runtime")
        return self._bedrock_client

    # ------------------------------------------------------------------
    # Primary entry point (called by Compiler._get_base_plan)
    # ------------------------------------------------------------------

    def plan(self, query: Dict[str, Any], error_context: Optional[str] = None) -> GraphTemplate:
        """
        Return the base GraphTemplate for this query.

        If LLM credentials are available: call the LLM.
        Otherwise: fall back to the heuristic planner.

        Parameters
        ----------
        error_context : str or None
            If provided (on a retry), a description of what went wrong with the
            previous plan.  It is appended to the LLM prompt so the model can
            correct its mistakes before producing the next plan.
        """
        if self._has_credentials and self.model is not None:
            for attempt in range(1 + self.max_retries):
                try:
                    raw, token_cost = self._call_llm(
                        query,
                        error_context=error_context,
                        is_retry=(attempt > 0),
                    )
                    return self._parse_and_validate(raw, token_cost, query, self.registry)
                except Exception as exc:
                    exc_str = str(exc)
                    if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                        raise RuntimeError(
                            f"[query_planner] AWS credentials have expired: {exc}"
                        ) from exc
                    if attempt < self.max_retries:
                        print(f"[query_planner] attempt {attempt+1} failed ({exc}), retrying…",file=sys.stderr)
            print("[query_planner] all LLM attempts failed, using heuristic fallback",file=sys.stderr)

        return self._heuristic_plan(query)

    # ------------------------------------------------------------------
    # Output synthesizer
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: Dict[str, Any],
        tool_outputs: Dict[str, Any],
    ) -> str:
        """
        Synthesize a final answer from the original query and aggregated
        tool execution outputs.

        Called by MCPExecutor when processing the output node.  Takes the
        raw per-node tool results and the original task description, then
        asks the LLM to produce a single, human-readable answer that
        directly addresses the user's request.

        Falls back to a JSON dump of tool_outputs when no LLM credentials
        are available or the call fails.
        """
        if not self._has_credentials:
            return json.dumps(tool_outputs, indent=2, default=str)

        task_text = query.get("text", str(query))

        system_text = (
            "You are a helpful assistant that synthesizes tool execution results "
            "into a clear, concise final answer. Given the original user task and "
            "the outputs returned by the tools that were executed to complete it, "
            "produce a single answer that directly addresses the task. "
            "Return only the final answer — no preamble, no explanation of steps."
        )

        user_text = (
            f"User task:\n{task_text}\n\n"
            f"Tool execution outputs:\n"
            f"{json.dumps(tool_outputs, indent=2, default=str)}\n\n"
            "Synthesize a final answer based on the above tool outputs."
        )

        try:
            response, _ = self._dispatch_llm_call(
                system_text=system_text,
                user_text=user_text,
                max_tokens=3000,
                temperature=0.1,
            )
            return response
        except Exception as exc:
            exc_str = str(exc)
            if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                raise
            print(
                f"[query_planner] WARNING: synthesize LLM call failed ({exc}); "
                "falling back to raw tool outputs",
                file=sys.stderr,
            )
            return json.dumps(tool_outputs, indent=2, default=str)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        query: Dict[str, Any],
        error_context: Optional[str] = None,
        is_retry: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        task_text = query.get("text", str(query))
        prompt    = _build_prompt(task_text, self.tool_catalog, self.max_steps,
                                  error_context=error_context)

        system_text = (
            "You are a workflow planner for an LLM tool-use agent. "
            "Given a task and a catalog of MCP tools, you output the "
            "minimal ordered sequence of tool calls needed to complete "
            "the task. You respond with valid JSON only — no prose."
        )

        if is_retry:
            system_text += (
                " IMPORTANT: your previous response could not be parsed. "
                "Do not over think or over explain — output ONLY the raw JSON object. "
            )

        return self._dispatch_llm_call(
            system_text, prompt, max_tokens=3000, temperature=0.1,
        )

    # ------------------------------------------------------------------
    # Bedrock Converse
    # ------------------------------------------------------------------

    def _call_bedrock_converse(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call Claude via the Bedrock Converse API and return
        (response_text, cost_metrics) in the same shape the rest of
        the planner expects.

        The Converse API uses a different message/response schema than
        OpenAI-compatible endpoints, so this method translates between
        the two worlds.
        """
        client   = self._get_bedrock_client()
        model_id = _resolve_bedrock_model_id(
            self.model
        )

        messages = [
            {
                "role": "user",
                "content": [{"text": user_text}],
            }
        ]

        response = client.converse(
            modelId=model_id,
            system=[{"text": system_text}],
            messages=messages,
            inferenceConfig={
                "maxTokens":   max_tokens,
                "temperature": temperature,
            },
        )

        # Extract text from the Converse response shape
        output_message = response["output"]["message"]
        message_text = ""
        for block in output_message.get("content", []):
            if "text" in block:
                message_text += block["text"]

        # Build cost_metrics from Bedrock's usage fields
        usage = response.get("usage", {})
        cost_metrics = {
            "prompt_tokens":      usage.get("inputTokens"),
            "completion_tokens":  usage.get("outputTokens"),
            "total_tokens":       usage.get("totalTokens"),
            "cost":               None,   # Bedrock does not return dollar cost
            "upstream_inference_cost":              None,
            "upstream_inference_prompt_cost":        None,
            "upstream_inference_completions_cost":   None,
            "model_id":   model_id,
            "stop_reason": response.get("stopReason"),
        }

        return message_text, cost_metrics

    # ------------------------------------------------------------------
    # OpenRouter provider
    # ------------------------------------------------------------------

    def _call_openrouter(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        import requests

        model = self.model if "/" in self.model else f"openai/{self.model}"
        payload = {
            "model":       model,
            "messages":    [
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
            json=payload,
            timeout=200.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._extract_openai_response(data)

    # ------------------------------------------------------------------
    # Azure OpenAI provider
    # ------------------------------------------------------------------

    def _call_azure(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        import requests

        url = (
            f"{self.azure_endpoint}/openai/deployments/{self.model}"
            "/chat/completions?api-version=2024-02-01"
        )
        payload = {
            "messages":    [
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(
            url,
            headers={"api-key": self.azure_api_key},
            json=payload,
            timeout=200.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._extract_openai_response(data)

    # ------------------------------------------------------------------
    # Generic base_url provider (any OpenAI-compatible endpoint)
    # ------------------------------------------------------------------

    def _call_base_url(
        self,
        system_text: str,
        user_text: str,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Tuple[str, Dict[str, Any]]:
        import requests

        url     = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model":       self.model,
            "messages":    [
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=200.0)
        resp.raise_for_status()
        data = resp.json()
        return self._extract_openai_response(data)

    # ------------------------------------------------------------------
    # Shared OpenAI-format response parser
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_openai_response(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Pull message text and cost metrics from an OpenAI-compatible
        /chat/completions response. Used by openrouter, azure, and
        base_url providers.
        """
        message = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        cost_details = usage.get("cost_details", {})
        cost_metrics = {
            "prompt_tokens":      usage.get("prompt_tokens"),
            "completion_tokens":  usage.get("completion_tokens"),
            "total_tokens":       usage.get("total_tokens"),
            "cost":               usage.get("cost"),
            "upstream_inference_cost":              cost_details.get("upstream_inference_cost"),
            "upstream_inference_prompt_cost":        cost_details.get("upstream_inference_prompt_cost"),
            "upstream_inference_completions_cost":   cost_details.get("upstream_inference_completions_cost"),
        }
        #print("query_planner/_extract_openai_response")
        #print(message,flush=True)
        #print("++"*1000)
        return message, cost_metrics

    # ------------------------------------------------------------------
    # Response → GraphTemplate
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        original_text: str,
        token_cost: Dict[str, Any],
        query: Dict[str, Any],
        registry: ToolRegistry,
    ) -> GraphTemplate:
        """
        Parse the LLM JSON response and build a validated GraphTemplate.

        Expected LLM output shape:
        {
            "description": "brief summary of the plan",
            "steps": [
                {"node_id": "t1", "server": "time-mcp",  "tool": "get_current_time",
                 "requires": []},
                {"node_id": "t2", "server": "math-mcp",  "tool": "calculate",
                 "requires": ["t1"]}
            ]
        }
        """
        text = original_text
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip()

        # Extract the first JSON object or array, ignoring any leading prose
        # (e.g. <think>...</think> blocks or other preamble from reasoning models)
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            text = match.group(1)

        parsed = json.loads(text)
        if isinstance(parsed, list):
            # LLM returned just the steps array — wrap it
            parsed = {"description": "", "steps": parsed}

        description = parsed.get("description", "")
        steps       = parsed.get("steps", [])

        if not steps:
            raise ValueError("LLM returned no steps")

        nodes: List[Node] = []
        edges: List[Tuple[str, str]] = []
        seen_ids: set = set()
        #print("QueryPlanner/_parseandvalidate()",flush=True)
        #print("Description:",description,flush=True)
        #print("Steps:",steps,flush=True)
        #print("++"*100)
        for step in steps[: self.max_steps]:
            nid    = str(step.get("node_id", "")).strip()
            kind   = str(step.get("kind", "tool")).strip()
            server = str(step.get("server",  "")).strip()
            tool   = str(step.get("tool",    "")).strip()
            tool_parameter = step.get("args",     {})
            produces       = {str(k): str(v)
                              for k, v in step.get("produces", {}).items()}
            reqs   = [str(r).strip() for r in step.get("requires", [])]

            if not nid:
                raise ValueError(f"Step missing node_id: {step}")
            if nid in seen_ids:
                raise ValueError(f"Duplicate node_id {nid!r}")

            seen_ids.add(nid)

            # Output aggregator node — no tool reference required
            if kind == "output" or nid == "output":
                nodes.append(Node(
                    node_id  = nid,
                    kind     = "output",
                    requires = reqs,
                ))
                for req in reqs:
                    edges.append((req, nid))
                continue

            if not server or not tool:
                raise ValueError(f"Step missing node_id/server/tool: {step}")
            if not registry.exists(server, tool):
                raise ValueError(f"({server!r}, {tool!r}) not in registry")

            nodes.append(Node(
                node_id  = nid,
                kind     = "tool",
                tool_ref = (server, tool),
                params   = tool_parameter,
                produces = produces,
                requires = reqs,
            ))
            for req in reqs:
                edges.append((req, nid))

        if not nodes:
            raise ValueError("No valid nodes after validation")

        tid = _make_template_id(query, description)
        return GraphTemplate(
            template_id = tid,
            nodes       = nodes,
            edges       = sorted(set(edges)),
            description = description,
            token_cost = token_cost,
            original_text=original_text

        )

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM)
    # ------------------------------------------------------------------

    def _heuristic_plan(
        self,
        query: Dict[str, Any]
    ) -> GraphTemplate:
        """
        Simple fallback: pick tools whose name/description overlap with
        the query text and chain them linearly.

        This is intentionally naive — it exists so the system can run
        end-to-end in CI without any API keys.
        """
        task_text  = query.get("text", str(query)).lower()
        hint_servers = [s.lower() for s in query.get("_servers", [])]

        # Score each (server, tool) by token overlap with the query
        scored: List[Tuple[float, str, str]] = []
        query_tokens = set(re.split(r"\W+", task_text))

        for (server, tool), spec in self.registry.tools.items():
            desc   = self.tool_catalog.get(server, {}).get(tool, {})
            if isinstance(desc, dict):
                desc_text = desc.get("description", "")
            else:
                desc_text = str(desc)
            combined   = f"{server} {tool} {desc_text}".lower()
            tokens     = set(re.split(r"\W+", combined))
            overlap    = len(query_tokens & tokens)
            # Boost servers that are explicitly mentioned in the query metadata
            if server.lower() in hint_servers:
                overlap += 10
            scored.append((overlap, server, tool))

        scored.sort(reverse=True)
        top = scored[: min(self.max_steps, 3)]  # at most 3 steps in fallback

        if not top:
            # Absolute last resort: first tool in registry
            (server, tool) = next(iter(self.registry.tools))
            top = [(0, server, tool)]

        nodes: List[Node] = []
        edges: List[Tuple[str, str]] = []
        for i, (_, server, tool) in enumerate(top):
            nid  = f"t{i+1}"
            reqs = [f"t{i}"] if i > 0 else []
            nodes.append(Node(node_id=nid, kind="tool",
                              tool_ref=(server, tool), requires=reqs))
            if reqs:
                edges.append((reqs[0], nid))

        tid = _make_template_id(query, "heuristic")
        return GraphTemplate(
            template_id = tid,
            nodes       = nodes,
            edges       = edges,
            token_cost  = None,
            description = "heuristic fallback plan",
            heuristic   = True,
        )

    # ------------------------------------------------------------------
    # LLM-guided edit proposal  (called by EvolutionarySearch._mutate)
    # ------------------------------------------------------------------

    def propose_edit(
        self,
        query:         Dict[str, Any],
        failing_nodes: List[str],
        current_nodes: List[Dict[str, Any]],
        registry,
        error_context: Optional[str] = None,
        exploration_mode: bool = False,
        allowed_ops: Optional[List[str]] = None,
    ) -> Tuple[Optional["TypedEdit"], Dict[str,Any]]:
        """
        Ask the LLM: "given this query, these failing nodes, and the current
        graph, what single structural edit would most improve the workflow?"

        Returns a validated TypedEdit, or None if the call fails / parse errors.

        Parameters
        ----------
        query         : the task query dict (needs "text" key)
        failing_nodes : node_ids sorted by error rate descending
        current_nodes : list of {node_id, server, tool, requires} dicts
        registry      : ToolRegistry — used to validate any (server, tool) pair
        """
        if not self._has_credentials:
            return None

        from .dsl import TypedEdit

        task_text = query.get("text", str(query))
        prompt    = _build_edit_prompt(
            task_text, failing_nodes, current_nodes, self.tool_catalog,
            error_context=error_context,
            exploration_mode=exploration_mode,
            allowed_ops=allowed_ops,
        )

        for attempt in range(1 + self.max_retries):
            try:
                raw, token_cost  = self._call_llm_raw(prompt)
                edit = _parse_edit(raw, registry)
                if edit is not None:
                    return edit, token_cost
            except Exception as exc:
                exc_str = str(exc)
                if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                    raise
                if attempt < self.max_retries:
                    print(f"[query_planner] propose_edit attempt {attempt+1} failed ({exc}), retrying…")

        return None, None

    def _call_llm_raw(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Call the LLM with a custom prompt string (not a full query plan)."""

        system_text = (
            "You are a workflow optimizer for an LLM tool-use agent. "
            "Given a failing workflow, you propose ONE targeted structural "
            "edit to fix or improve it. Respond with valid JSON only."
        )

        return self._dispatch_llm_call(
            system_text, prompt, max_tokens=3000, temperature=0.2,
        )

    # ------------------------------------------------------------------
    # Meta-guidance  (AdaEvolve L3 — triggered on persistent stagnation)
    # ------------------------------------------------------------------

    def meta_guide(
        self,
        query:            Dict[str, Any],
        best_nodes:       List[Dict[str, Any]],
        failed_summaries: List[str],
    ) -> Optional[GraphTemplate]:
        """
        High-level strategy redesign triggered when per-query growth signal G_t
        has stayed below tau_m for an extended period (AdaEvolve Level 3).

        Sends the current best workflow structure, the task, and a digest of
        recent low-scoring attempts to the LLM and asks for a *qualitatively
        different* plan — one that avoids the patterns the search has already
        exhausted.

        Parameters
        ----------
        query            : task query dict (needs "text" key)
        best_nodes       : serialisable node list from the current best candidate
                           [{node_id, server, tool, requires}, ...]
        failed_summaries : short strings describing recent low-scoring attempts

        Returns the new GraphTemplate on success, or None on any failure.
        """
        if not self._has_credentials:
            return None

        task_text = query.get("text", str(query))
        prompt    = _build_meta_guidance_prompt(
            task_text, best_nodes, failed_summaries,
            self.tool_catalog, self.max_steps,
        )

        system_text = (
            "You are a senior workflow architect for an LLM tool-use agent. "
            "The evolutionary search has stagnated — all recent variants score "
            "similarly. Propose a FUNDAMENTALLY DIFFERENT plan that avoids the "
            "patterns already tried. Respond with valid JSON only — no prose."
        )

        for attempt in range(1 + self.max_retries):
            try:
                raw, token_cost = self._dispatch_llm_call(
                    system_text, prompt, max_tokens=3000, temperature=0.4,
                )
                return self._parse_and_validate(raw, token_cost, query, self.registry)
            except Exception as exc:
                exc_str = str(exc)
                if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                    raise
                if attempt < self.max_retries:
                    print(
                        f"[query_planner] meta_guide attempt {attempt+1} failed ({exc}), retrying…",
                        file=sys.stderr,
                    )
        return None
# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_template_id(query: Dict[str, Any], description: str) -> str:
    payload = {"q": query, "d": description}
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _build_prompt(
    task_text: str,
    tool_catalog: Dict[str, Dict[str, Any]],
    max_steps: int,
    error_context: Optional[str] = None,
) -> str:
    # Build catalog section
    catalog_lines: List[str] = []
    for server in sorted(tool_catalog):
        tools = tool_catalog[server]
        if not tools:
            continue
        catalog_lines.append(f"SERVER: {server}")
        for tool_name, spec in sorted(tools.items()):
            desc = ""
            input_schema: Dict[str, Any] = {}
            if isinstance(spec, dict):
                desc = spec.get("description", "")
                input_schema = spec.get("input_schema") or {}
            if desc:
                desc = textwrap.shorten(desc, width=120, placeholder="…")
                catalog_lines.append(f"  {tool_name}  —  {desc}")
            else:
                catalog_lines.append(f"  {tool_name}")
            # Show input parameter names so the LLM can infer the response shape
            props = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
            if props:
                param_names = ", ".join(props.keys())
                catalog_lines.append(f"    params: {param_names}")
        catalog_lines.append("")
    catalog_text = "\n".join(catalog_lines).strip()

    schema_example = json.dumps({
        "steps": [
            {
                "node_id": "t1",
                "server": "Paper Search",
                "tool": "search_arxiv",
                "params": {"query": "CRISPR machine learning", "max_results": 5},
                "produces": {"paper_id": "content.results[0].id"},
                "requires": [],
            },
            {
                "node_id": "t2",
                "server": "Paper Search",
                "tool": "download_arxiv",
                "params": {"paper_id": "$t1.paper_id", "save_path": "./papers"},
                "produces": {},
                "requires": ["t1"],
            },
            {
                "node_id": "output",
                "kind": "output",
                "params": {},
                "produces": {},
                "requires": ["t2"],
            },
        ]
    }, indent=2)

    error_section = ""
    if error_context:
        error_section = f"\n\n## Previous plan failed — fix these issues\n{error_context}\n\nRevise your plan to avoid repeating these errors."

    return textwrap.dedent(f"""
        ## Task
        {task_text}

        ## Available MCP tool servers
        {catalog_text}

        ## Your job
        Plan the minimal ordered sequence of tool calls needed to complete
        the task above.  Use at most {max_steps} steps.

        Rules:
        - Use ONLY server and tool names from the catalog above — exact spelling.
        - node_id values must be unique: t1, t2, t3, …
        - "requires" lists node_ids that must complete before this step runs.
          A step with no dependencies uses an empty list [].
        - "args" contains the static input parameters for the tool call.
        - "produces" declares named values extracted from this node's output for
          use by downstream nodes.  Each entry maps a short local name to a
          dot-notation path into the tool's output dict:
            path syntax:  "content"              the entire raw text response
                          "content.field"        plain field inside the JSON content
                          "content.list[0].key"  array index then field
          Tool outputs always have a "content" key whose value is the raw text response
          (often JSON).  Use "content.<field>" paths to reach nested data.
          IMPORTANT: Only use specific field paths (e.g. "content.papers[0].id") if
          you are confident the field exists in the response. When unsure of the exact
          schema, use "content" to capture the whole response and pass it downstream —
          the downstream tool can receive the full JSON and the executor will try to
          extract what it needs.
          If no downstream node needs this output, set "produces" to {{}}.
        - In "args", reference an upstream node's named output with "$node_id.name"
          where "name" matches a key declared in that node's "produces".
        - Include EVERY tool call needed to fully answer the task.
          Do not include unnecessary steps.
        - If the plan has multiple terminal steps (steps that no later step
          depends on), add a final step with node_id "output", kind "output",
          requires listing all those terminal node_ids, and empty args/produces.
          This node aggregates all results into the final output.
          Example of when this is needed: fetching BTC price (t1) and ETH price
          (t2) in parallel both produce results — add an output node that
          requires ["t1","t2"] so both values are returned together.

        ## Output format
        Respond with ONLY a JSON object — no prose, no markdown fences:

        {schema_example}{error_section}
    """).strip()



def _parse_edit(text: str, registry) -> "Optional[TypedEdit]":
    """
    Parse the LLM's JSON edit proposal into a TypedEdit.

    Expected shape (one of):
        {"op": "swap_tool",       "node_id": "t2", "new_server": "math-mcp", "new_tool": "calculate"}
        {"op": "add_tool_step",   "node_id": "t4", "server": "time-mcp", "tool": "get_time", "after_node_id": "t3"}
        {"op": "remove_tool_step","node_id": "t2"}
        {"op": "reorder_step",    "node_id": "t3", "new_requires": ["t1"]}
        {"op": "insert_validator","after_node_id": "t2", "validator_id": "v1"}
        {"op": "set_param",       "node_id": "t1", "key": "strict_schema", "value": false}
    """
    import re, json
    from .dsl import TypedEdit

    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text.strip(), flags=re.MULTILINE)
    text = text.strip()

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        text = match.group(1)

    parsed = json.loads(text)
    op     = parsed.get("op", "")

    VALID_OPS = {
        "swap_tool", "add_tool_step", "remove_tool_step",
        "reorder_step", "insert_validator", "set_param",
    }
    if op not in VALID_OPS:
        return None

    # Validate (server, tool) pairs if present
    if op == "swap_tool":
        ns, nt = parsed.get("new_server", ""), parsed.get("new_tool", "")
        if not registry.exists(ns, nt):
            return None
        return TypedEdit(op=op, args={
            "node_id": parsed["node_id"], "new_server": ns, "new_tool": nt,
        })

    if op == "add_tool_step":
        ns, nt = parsed.get("server", ""), parsed.get("tool", "")
        if not registry.exists(ns, nt):
            return None
        args = {"node_id": parsed["node_id"], "server": ns, "tool": nt}
        if "after_node_id" in parsed:
            args["after_node_id"] = parsed["after_node_id"]
        return TypedEdit(op=op, args=args)

    if op == "remove_tool_step":
        return TypedEdit(op=op, args={"node_id": parsed["node_id"]})

    if op == "reorder_step":
        return TypedEdit(op=op, args={
            "node_id":      parsed["node_id"],
            "new_requires": list(parsed.get("new_requires", [])),
        })

    if op == "insert_validator":
        return TypedEdit(op=op, args={
            "after_node_id": parsed["after_node_id"],
            "validator_id":  parsed.get("validator_id", f"v_{parsed['after_node_id']}"),
        })

    if op == "set_param":
        return TypedEdit(op=op, args={
            "node_id": parsed["node_id"],
            "key":     parsed["key"],
            "value":   parsed["value"],
        })

    return None


def _build_edit_prompt(
    task_text:        str,
    failing_nodes:    List[str],
    current_nodes:    List[Dict[str, Any]],
    tool_catalog:     Dict[str, Dict[str, Any]],
    error_context:    Optional[str] = None,
    exploration_mode: bool = False,
    allowed_ops:      Optional[List[str]] = None,
) -> str:
    import json, textwrap

    graph_text = json.dumps(current_nodes, indent=2)

    failing_text = (
        ", ".join(failing_nodes) if failing_nodes
        else "none (execution succeeded but result was incorrect)"
    )

    # Condense catalog to keep prompt short
    catalog_lines = []
    for server in sorted(tool_catalog):
        tools = tool_catalog[server]
        if not tools:
            continue
        catalog_lines.append(f"SERVER: {server}")
        for tool_name in sorted(tools):
            desc = tools[tool_name].get("description", "") if isinstance(tools[tool_name], dict) else ""
            desc = textwrap.shorten(desc, width=80, placeholder="…") if desc else ""
            catalog_lines.append(f"  {tool_name}" + (f"  — {desc}" if desc else ""))
    catalog_text = "\n".join(catalog_lines)

    example = json.dumps({
        "op": "swap_tool",
        "node_id": "t2",
        "new_server": "math-mcp",
        "new_tool": "calculate"
    })

    error_section = (
        f"\n        ## Last known error\n        {error_context}"
        if error_context else ""
    )

    if exploration_mode:
        job_instruction = (
            "You are in EXPLORATION mode. Propose a NOVEL or CREATIVE structural edit "
            "that tries a meaningfully different approach — for example, replacing a tool "
            "with a different one from the catalog, adding a previously unused tool step, "
            "or restructuring the dependency order in an unconventional way. "
            "Do NOT simply fix the failing node; aim for variety."
        )
    else:
        job_instruction = (
            "You are in EXPLOITATION mode. Propose a TARGETED, INCREMENTAL edit that "
            "directly addresses the highest-error failing node listed above. "
            "Prefer fixing or swapping the problematic node rather than restructuring "
            "the whole graph."
        )

    _all_ops = {
        "swap_tool":        "node_id, new_server, new_tool",
        "add_tool_step":    "node_id (new), server, tool, after_node_id (optional)",
        "remove_tool_step": "node_id",
        "reorder_step":     "node_id, new_requires (list of node_ids)",
        "insert_validator": "after_node_id, validator_id",
        "set_param":        "node_id, key, value",
    }
    _permitted = set(allowed_ops) if allowed_ops else set(_all_ops)
    ops_lines = "\n".join(
        f"  {op:<20} — {fields}"
        for op, fields in _all_ops.items()
        if op in _permitted
    )
    if allowed_ops:
        ops_constraint = f"You MUST use one of these ops: {', '.join(sorted(_permitted))}."
    else:
        ops_constraint = ""

    return textwrap.dedent(f"""
        ## Task
        {task_text}

        ## Current workflow graph
        {graph_text}

        ## Failing nodes (highest error rate first)
        {failing_text}{error_section}

        ## Available tools
        {catalog_text}

        ## Your job
        {job_instruction}

        Valid ops and their required fields:
{ops_lines}

        {ops_constraint}
        Use ONLY server and tool names from the catalog above.

        ## Output format
        Respond with ONLY a JSON object — no prose, no markdown fences:
        {example}
    """).strip()


def _build_meta_guidance_prompt(
    task_text:        str,
    best_nodes:       List[Dict[str, Any]],
    failed_summaries: List[str],
    tool_catalog:     Dict[str, Dict[str, Any]],
    max_steps:        int,
) -> str:
    """
    Build the prompt for AdaEvolve Level-3 meta-guidance.

    The LLM receives the current best workflow structure, a digest of recent
    low-scoring attempts, and the full tool catalog, and is asked to produce
    a plan that is qualitatively different from anything already tried.
    """
    current_plan = json.dumps(best_nodes, indent=2) if best_nodes else "none"
    failed_text  = "\n".join(f"  - {s}" for s in failed_summaries) or "  - (none recorded)"

    catalog_lines: List[str] = []
    for server in sorted(tool_catalog):
        tools = tool_catalog[server]
        if not tools:
            continue
        catalog_lines.append(f"SERVER: {server}")
        for tool_name, spec in sorted(tools.items()):
            desc = spec.get("description", "") if isinstance(spec, dict) else ""
            desc = textwrap.shorten(desc, width=100, placeholder="…") if desc else ""
            catalog_lines.append(f"  {tool_name}" + (f"  — {desc}" if desc else ""))
    catalog_text = "\n".join(catalog_lines).strip()

    schema_example = json.dumps({
        "description": "Alternative strategy using a different tool sequence",
        "steps": [
            {"node_id": "t1", "server": "example-server", "tool": "some_tool",
             "args": {}, "produces": {}, "requires": []},
        ]
    }, indent=2)

    return textwrap.dedent(f"""
        ## Task
        {task_text}

        ## Current best workflow (search has stagnated here)
        {current_plan}

        ## Recent low-scoring attempts
        {failed_text}

        ## Available MCP tool servers
        {catalog_text}

        ## Your job
        The evolutionary search has exhausted local improvements around the current
        best workflow. Propose a FUNDAMENTALLY DIFFERENT plan that:
          - Uses a different combination or ordering of tools
          - Approaches the task from a new angle not represented above
          - Does NOT simply swap one tool for a near-equivalent

        Use at most {max_steps} steps. Apply the same JSON rules as always:
        - Only use server/tool names from the catalog — exact spelling.
        - node_id values must be unique: t1, t2, t3, …
        - "requires" lists upstream node_ids (empty list = no dependency).
        - "args" contains static tool input parameters.
        - "produces" maps short names to dot-notation paths into the tool output.

        ## Output format
        Respond with ONLY a JSON object — no prose, no markdown fences:
        {schema_example}
    """).strip()