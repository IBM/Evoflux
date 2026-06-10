from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import threading

from .compiler import CompiledWorkflow
from .dsl import Node, ToolRegistry
from .perturbations import PerturbationSampler


import re as _re


def _deep_find(data: Any, key: str) -> Any:
    """
    Depth-first search for *key* anywhere inside *data* (dict/list/JSON string).
    Returns the first value found, or None.  Used as a last-resort fallback when
    the dot-notation path produced by the LLM doesn't match the actual structure.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            found = _deep_find(v, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _deep_find(item, key)
            if found is not None:
                return found
    return None


def _extract_path(data: Any, path: str) -> Any:
    """
    Traverse *data* using a dot-notation path with integer or wildcard indexing.
    JSON strings encountered during traversal are auto-parsed.

    Supported syntax
    ----------------
    ``"content"``                    → data["content"]
    ``"content.results[0].id"``      → data["content"]["results"][0]["id"]
    ``"items[2].name"``              → data["items"][2]["name"]
    ``"candlesticks[*].close"``      → [item["close"] for item in data["candlesticks"]]
    ``"content.rows[*].value"``      → list of "value" from each row in content

    Returns ``None`` if any segment cannot be resolved (unresolvable wildcards
    return an empty list rather than ``None``).
    """
    if not path:
        return data

    # Auto-parse a JSON string at this level before descending
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return None

    # Split only on the first dot so the rest is handled by recursion
    first, _, rest = path.partition(".")

    # ── wildcard: "key[*]" ──────────────────────────────────────────────
    m_wild = _re.match(r'^(\w+)\[\*\]$', first)
    if m_wild:
        key = m_wild.group(1)
        if not isinstance(data, dict):
            return None
        arr = data.get(key)
        if arr is None:
            return None
        if isinstance(arr, str):
            try:
                arr = json.loads(arr)
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(arr, list):
            return None
        if not rest:
            return arr                        # "$t1.items[*]" → whole list
        # Apply remaining path to every element and collect results
        return [_extract_path(item, rest) for item in arr]

    # ── integer index: "key[N]" or plain "key" ──────────────────────────
    m = _re.match(r'^(\w+)(?:\[(\d+)\])?$', first)
    if not m:
        return None
    key, idx_str = m.group(1), m.group(2)

    if not isinstance(data, dict):
        return None
    current = data.get(key)

    if idx_str is not None and current is not None:
        try:
            current = current[int(idx_str)]
        except (IndexError, TypeError, ValueError):
            return None

    if not rest:
        return current
    return _extract_path(current, rest)


def _coerce_value(
    val: Any,
    expected_type: Optional[str],
    items_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convert *val* to the primitive type named by *expected_type* (a JSON Schema
    ``"type"`` string).  Returns *val* unchanged when coercion is impossible or
    *expected_type* is unknown.

    *items_schema* is the JSON Schema ``"items"`` sub-schema for array properties;
    it is used to coerce each element of the array individually.
    """
    if expected_type is None or val is None:
        return val

    # Shared helper: extract a scalar number from any value shape.
    # Handles: plain numeric, numeric string, JSON object string, dict.
    def _to_number(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
            # Maybe a JSON object — try extracting first numeric field
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                # Plain text response (e.g. "68.9 inches", "eGFR: 65.3") —
                # extract the first number from the string.
                m = _re.search(r'-?\d+(?:\.\d+)?', v)
                if m:
                    try:
                        return float(m.group())
                    except (ValueError, TypeError):
                        pass
                return None
        if isinstance(v, dict):
            _NUM_KEYS = (
                "value", "result", "price", "lastPrice", "last",
                "close", "c", "open", "o", "high", "h", "low", "l",
                "amount", "count", "total", "number", "n",
            )
            for k in _NUM_KEYS:
                if k in v:
                    try:
                        return float(v[k])
                    except (TypeError, ValueError):
                        pass
            # Fallback: first value that parses as float
            for fv in v.values():
                try:
                    return float(fv)
                except (TypeError, ValueError):
                    pass
        return None

    if expected_type in ("number", "float"):
        n = _to_number(val)
        return n if n is not None else val

    if expected_type == "integer":
        n = _to_number(val)
        return int(n) if n is not None else val

    if expected_type == "boolean":
        if isinstance(val, str):
            if val.lower() in ("true", "1", "yes"):
                return True
            if val.lower() in ("false", "0", "no"):
                return False
        return val

    if expected_type == "string":
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        if not isinstance(val, str):
            return str(val)
        return val
        # NOTE: format normalization is applied after this in _coerce_args

    if expected_type == "array":
        # Parse a JSON string first
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return val   # not parseable — nothing more we can do

        if isinstance(val, dict):
            # Tool output was a dict — search its values for a list.
            # Prefer the first list whose items are numbers (most likely the
            # intended payload for tools like mean/sum/std).
            numeric_list = None
            any_list = None
            for v in val.values():
                if isinstance(v, list):
                    if any_list is None:
                        any_list = v
                    if all(isinstance(x, (int, float)) for x in v):
                        numeric_list = v
                        break
            val = numeric_list if numeric_list is not None else (any_list or val)

        if not isinstance(val, list):
            return val

        # Coerce each element to the items type declared in the schema
        item_type = (items_schema or {}).get("type") if items_schema else None
        if item_type is None:
            return val

        coerced_items = []
        for elem in val:
            if isinstance(item_type, list):
                item_type = next((t for t in item_type if t != "null"), None)
            if item_type in ("number", "float", "integer") and not isinstance(elem, (int, float)):
                # Element needs to be a scalar — delegate to _coerce_value which
                # now handles dict/string → number via _to_number internally.
                scalar_val = _coerce_value(elem, item_type, items_schema=None)
                if isinstance(scalar_val, (int, float)) and not isinstance(scalar_val, bool):
                    coerced_items.append(scalar_val)
                # If still not a number, skip rather than crash the tool
            else:
                coerced_items.append(_coerce_value(elem, item_type, items_schema=None))
        return coerced_items

    if expected_type == "object":
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return val

    return val


def _normalize_format(val: Any, prop_schema: Dict[str, Any]) -> Any:
    """
    Apply format normalization to *val* based on the schema property's
    ``pattern``, ``format``, and ``description`` fields.

    This runs after type coercion so *val* is already the right Python type
    (usually str).  It detects the intended semantic format from schema hints
    and reformats the value to match.

    Currently handled formats
    -------------------------
    HH:MM time        — pattern like \\d{2}:\\d{2}, format "time",
                        or description containing "HH:MM" / "24-hour"
    YYYY-MM-DD date   — format "date" or description containing "YYYY-MM-DD"
    ISO datetime      — format "date-time"
    """
    if not isinstance(prop_schema, dict):
        return val

    pattern = prop_schema.get("pattern", "")
    fmt     = prop_schema.get("format", "").lower()
    desc    = prop_schema.get("description", "").lower()

    # ── HH:MM time ────────────────────────────────────────────────────────────
    is_time_fmt = (
        fmt == "time"
        or "hh:mm" in desc
        or "24-hour" in desc
        or "time format" in desc
        or _re.search(r'\\d\{2\}:\\\\?d\{2\}', pattern)   # regex pattern hint
        or _re.search(r'\d{2}:\d{2}', pattern)             # literal example in pattern
    )
    if is_time_fmt and isinstance(val, str):
        s = val.strip()
        # Already valid HH:MM — just zero-pad
        m = _re.match(r'^(\d{1,2}):(\d{2})$', s)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        # Decimal / integer hour → HH:00
        try:
            h = float(s)
            return f"{int(h):02d}:{round((h % 1) * 60):02d}"
        except ValueError:
            pass
        # ISO datetime or "HH:MM:SS" → strip to HH:MM
        m = _re.match(r'.*?(\d{1,2}):(\d{2})(?::\d{2})?', s)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"

    # ── YYYY-MM-DD date ───────────────────────────────────────────────────────
    is_date_fmt = (
        fmt == "date"
        or "yyyy-mm-dd" in desc
        or "date format" in desc
    )
    if is_date_fmt and isinstance(val, str):
        s = val.strip()
        if _re.match(r'^\d{4}-\d{2}-\d{2}$', s):
            return s   # already correct
        # ISO datetime → strip time part
        m = _re.match(r'^(\d{4}-\d{2}-\d{2})[ T]', s)
        if m:
            return m.group(1)

    # ── ISO date-time ─────────────────────────────────────────────────────────
    if fmt == "date-time" and isinstance(val, str):
        s = val.strip()
        if not s.endswith("Z") and "+" not in s and _re.match(r'^\d{4}-\d{2}-\d{2}', s):
            return s + "Z"

    return val


def _coerce_args(
    args: Dict[str, Any],
    input_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Coerce and remap *args* to match the tool's *input_schema*.

    Two passes:

    Pass 1 — coerce values whose keys already match schema properties.

    Pass 2 — positional remapping: if the LLM used wrong parameter names
    (e.g. ``a``/``b`` instead of ``minuend``/``subtrahend``), detect the
    mismatch and map the unrecognised args onto the missing required
    properties in the order they appear in the schema.
    """
    properties = input_schema.get("properties", {}) if input_schema else {}
    required   = set(input_schema.get("required", [])) if input_schema else set()
    if not properties:
        return args

    def _coerce_one(val: Any, prop: Dict[str, Any]) -> Any:
        expected = prop.get("type")
        if isinstance(expected, list):
            expected = next((t for t in expected if t != "null"), None)
        items_schema = prop.get("items") if isinstance(prop.get("items"), dict) else None
        coerced_val = _coerce_value(val, expected, items_schema=items_schema)
        return _normalize_format(coerced_val, prop)

    coerced: Dict[str, Any] = {}

    # Pass 1: keys that exist in the schema
    unknown_keys: List[str] = []
    for key, val in args.items():
        if key in properties:
            coerced[key] = _coerce_one(val, properties[key])
        else:
            unknown_keys.append(key)

    # Pass 2: remap unknown keys onto missing required properties positionally
    missing_required = [k for k in properties if k in required and k not in coerced]
    if unknown_keys and missing_required:
        print(
            f"[executor] arg name mismatch: passed={unknown_keys} "
            f"missing={missing_required} — remapping positionally",
            flush=True,
        )
        for schema_key, passed_key in zip(missing_required, unknown_keys):
            coerced[schema_key] = _coerce_one(args[passed_key], properties[schema_key])

    return coerced


def _resolve_args(params: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace ``$node_id`` / ``$node_id.name`` references in *params* with
    values from *memory*.

    Each entry in *memory* is a dict that contains:
    * the raw tool output keys (e.g. ``"content"``)
    * any named extractions declared in the node's ``produces`` field

    Rules
    -----
    * ``"$t1"``       → the entire memory entry for node t1.
    * ``"$t1.name"``  → ``memory["t1"]["name"]`` (named extraction or raw key).
    * Non-string values and strings without ``$`` are passed through unchanged.
    * Unresolvable references keep their placeholder string for debuggability.
    """
    resolved: Dict[str, Any] = {}
    for key, val in params.items():
        if not (isinstance(val, str) and val.startswith("$")):
            resolved[key] = val
            continue

        ref = val[1:]                              # strip leading "$"
        node_id, _, path = ref.partition(".")
        upstream = memory.get(node_id, val)        # fall back to placeholder

        if not path:
            resolved[key] = upstream              # "$t1" → whole memory entry
        else:
            # Use _extract_path so dot-notation, [N], and [*] all work.
            # If the direct path fails, retry with a "content." prefix because
            # MCP tool outputs are always wrapped in {"content": "<raw_string>"},
            # so paths like "btc_candlesticks[*].close" often need to go through
            # "content.btc_candlesticks[*].close" at the raw-output level.
            extracted = _extract_path(upstream, path)
            if extracted is None and not path.startswith("content"):
                extracted = _extract_path(upstream, "content." + path)
            if extracted is not None:
                resolved[key] = extracted
            else:
                # Path didn't match the actual tool output structure (LLM guessed
                # wrong field names).  Try a deep-search on the last key segment
                # before falling back to raw content string.
                raw_content = (
                    upstream.get("content") if isinstance(upstream, dict) else None
                )
                last_key = path.rstrip("]").rsplit(".", 1)[-1].split("[")[0]
                deep = _deep_find(raw_content, last_key)
                if deep is not None:
                    print(
                        f"[executor] WARNING: could not resolve reference '${node_id}.{path}' "
                        f"— available keys: {list(upstream.keys()) if isinstance(upstream, dict) else type(upstream).__name__}"
                        f"; resolved via deep-search on key '{last_key}'",
                        flush=True,
                    )
                    resolved[key] = deep
                else:
                    print(
                        f"[executor] WARNING: could not resolve reference '${node_id}.{path}' "
                        f"— available keys: {list(upstream.keys()) if isinstance(upstream, dict) else type(upstream).__name__}"
                        + ("; falling back to raw content" if raw_content is not None else "; no content fallback"),
                        flush=True,
                    )
                    resolved[key] = raw_content if raw_content is not None else val

    return resolved


@dataclass
class StepLog:
    node_id: str
    kind: str
    tool: Optional[str]
    ok: bool
    error_type: Optional[str]
    latency_ms: int
    input_args: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None


@dataclass
class ExecResult:
    ok: bool
    logs: List[StepLog]
    final_output: Optional[Dict[str, Any]]
    violation: Optional[str] = None
    # Errors collected across all parallel nodes in a failed wave.
    # Non-empty when ok=False due to tool failures (not policy violations).
    # Feed this to the re-planner so it has the full failure context.
    replan_errors: List[Dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class MetricBreakdown:
    task_fulfillment : float
    grounding : float
    tool_appropriateness : float
    parameter_accuracy : float
    dependency_awareness : float
    parallelism_and_efficiency : float
    task_completion_score : float
    tool_selection_score : float
    planning_effectiveness_and_efficiency_score : float
    task_fulfillment_reasoning : str
    grounding_reasoning : str
    tool_appropriateness_reasoning : str
    parameter_accuracy_reasoning : str
    dependency_awareness_reasoning : str
    parallelism_and_efficiency_reasoning : str
    
# Type alias for the scoring callable passed into the executor.
# Signature: (ExecResult, CompiledWorkflow) -> float
# Built-in default uses compute_metrics + scalarize from metrics.py.
# Adapters override this with a real judge (LLM, rule-based, etc.).
ScoringFn = Callable[[ExecResult, "CompiledWorkflow"], float]


def _default_scoring_fn(res: ExecResult, wf: "CompiledWorkflow") -> float:
    """
    Built-in fallback: structural proxy metrics + linear scalarization.
    Imported lazily to avoid a circular import (metrics → executor → metrics).
    """
    from .metrics import compute_metrics, scalarize
    return scalarize(compute_metrics(res))


class MCPExecutor:
    """
    Instrumented execution harness.

    Two extension points for adapter integration:

    1. _call_tool(server, tool, args) → (ok, err, output)
       Override in a subclass (AdaptedExecutor) to route calls to the real
       benchmark backend instead of the built-in echo stub.

    2. score_fn: ScoringFn
       Injected at construction time.  Called by score() to convert an
       ExecResult into a scalar reward.  EvolutionarySearch and teacher.py
       both call executor.score(res) rather than compute_metrics/scalarize
       directly, so the adapter's LLM judge is used consistently everywhere:
       candidate scoring, fair comparison, and teacher label construction.

    3. synthesizer_fn: Optional[Callable[[Dict, Dict], Any]]
       Optional LLM synthesizer injected at construction time.  When set,
       the output node calls synthesizer_fn(query, aggregated_tool_outputs)
       after aggregating leaf results, producing a single synthesized final
       answer instead of a raw dict of node outputs.

    crn_seed in run() forks a fresh PerturbationSampler per call so that
    two calls with the same seed see identical perturbations — the CRN
    (Common Random Numbers) guarantee required for fair comparison.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sampler: PerturbationSampler,
        score_fn: ScoringFn,
        timeout_s: float = 30.0,
        synthesizer_fn: Optional[Callable] = None,
    ):
        self.registry     = registry
        self.base_sampler = sampler
        self.timeout_s    = timeout_s
        # score_fn=None means "use the built-in structural proxy"
        self._score_fn: ScoringFn = score_fn
        # synthesizer_fn=None means output node returns raw aggregated dict
        self._synthesizer_fn: Optional[Callable] = synthesizer_fn

    # ------------------------------------------------------------------
    # Public scoring entry point
    # ------------------------------------------------------------------

    def score(self, res: ExecResult, wf: CompiledWorkflow) -> MetricBreakdown:
        """
        Executes LLM Judge from their corresponding benchmark.
        """
        return self._score_fn(res, wf)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, wf: CompiledWorkflow, crn_seed: int) -> ExecResult:
        # Fork a fresh sampler keyed to crn_seed → identical perturbations
        # for any two calls that share the same seed (CRN property).
        local_sampler = PerturbationSampler(self.base_sampler.base, seed=crn_seed)
        pert = local_sampler.sample()

        logs: List[StepLog] = []
        memory: Dict[str, Any] = {}   # node_id → output mem entry
        logs_lock = threading.Lock()
        steps = 0

        # Walk nodes in waves: each wave contains all nodes whose requires are
        # already satisfied by completed nodes.  Nodes in the same wave have no
        # mutual dependencies and are executed in parallel.
        completed: set = set()
        remaining: List[Node] = list(wf.nodes)

        while remaining:
            # Collect every node whose dependencies are fully satisfied.
            wave: List[Node] = [
                n for n in remaining
                if all(r in completed for r in (n.requires or []))
            ]

            if not wave:
                # Topo sort guarantees this never happens; guard anyway.
                return ExecResult(
                    ok=False, logs=logs, final_output=None,
                    violation="unresolvable_dependency",
                )

            remaining = [n for n in remaining if n not in wave]

            if steps + len(wave) > wf.policy.max_steps:
                return ExecResult(
                    ok=False, logs=logs, final_output=None,
                    violation="max_steps_exceeded",
                )

            # Snapshot memory so every node in this wave reads a consistent view
            # of outputs from prior waves (nodes in the same wave are independent).
            mem_snapshot: Dict[str, Any] = dict(memory)

            # ── Server / tool pre-check (main thread, before any parallel work) ──
            # Validate every tool node in the wave against the registry now,
            # so failures are discovered sequentially before threads are spawned.
            wave_errors_precheck: List[Dict[str, Any]] = []
            for n in wave:
                if n.kind == "tool" and n.tool_ref is not None:
                    server, tool = n.tool_ref
                    if not self.registry.exists(server, tool):
                        t_err = StepLog(
                            node_id=n.node_id,
                            kind=n.kind,
                            tool=f"{server}.{tool}",
                            ok=False,
                            error_type="unknown_tool",
                            latency_ms=0,
                        )
                        logs.append(t_err)
                        wave_errors_precheck.append({
                            "node_id": n.node_id,
                            "tool": t_err.tool,
                            "error_type": "unknown_tool",
                            "server": server,
                        })
            if wave_errors_precheck:
                return ExecResult(
                    ok=False,
                    logs=logs,
                    final_output=None,
                    replan_errors=wave_errors_precheck,
                )

            # ── Pre-sample perturbation decisions (main thread, RNG not thread-safe) ─
            # All calls to local_sampler must happen here before threads start;
            # random.Random is not thread-safe across concurrent callers.
            node_perturbations: Dict[str, Tuple[bool, bool, int]] = {
                n.node_id: (
                    local_sampler.should_timeout(pert.timeout_inject_p),
                    local_sampler.should_error(pert.tool_error_inject_p),
                    local_sampler.jitter_ms(pert.latency_jitter_ms),
                )
                for n in wave
            }

            # ── Execute the wave ──────────────────────────────────────────────
            # Results: list of (StepLog, mem_update | None, error_detail | None)
            wave_results: List[Tuple[StepLog, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []

            if len(wave) == 1:
                n = wave[0]
                wave_results.append(
                    self._execute_node(n, mem_snapshot, *node_perturbations[n.node_id], query=wf.query)
                )
            else:
                futures = {}
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    for n in wave:
                        futures[pool.submit(
                            self._execute_node, n, mem_snapshot,
                            *node_perturbations[n.node_id],
                            query=wf.query,
                        )] = n
                    for fut in as_completed(futures):
                        wave_results.append(fut.result())

            # ── Collect results; accumulate all errors before deciding ────────
            wave_errors: List[Dict[str, Any]] = []
            for step_log, mem_update, err_detail in wave_results:
                with logs_lock:
                    logs.append(step_log)
                if step_log.ok:
                    completed.add(step_log.node_id)
                    if mem_update:
                        memory.update(mem_update)
                else:
                    wave_errors.append(err_detail or {
                        "node_id": step_log.node_id,
                        "error_type": step_log.error_type,
                    })

            steps += len(wave)

            # If any node in this wave failed, surface all errors for re-planning
            # rather than stopping on the first failure.
            if wave_errors:
                return ExecResult(
                    ok=False,
                    logs=logs,
                    final_output=None,
                    replan_errors=wave_errors,
                )

        final = memory.get(wf.nodes[-1].node_id) if wf.nodes else None
        return ExecResult(ok=True, logs=logs, final_output=final)

    # ------------------------------------------------------------------
    # Single-node execution (called from run(), safe to use in threads)
    # ------------------------------------------------------------------

    def _execute_node(
        self,
        n: Node,
        memory: Dict[str, Any],
        do_timeout: bool,
        do_error: bool,
        jitter_ms: int,
        query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[StepLog, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Execute one node and return (StepLog, mem_update, error_detail).

        mem_update  — dict to merge into the shared memory on success, or None.
        error_detail — structured error info to feed the re-planner on failure.

        Thread-safe: reads only from the immutable *memory* snapshot passed in;
        writes are returned as a plain dict and merged by the caller.
        do_timeout / do_error / jitter_ms are pre-sampled by run() in the main
        thread so no RNG state is shared across threads.
        query — the original compiled workflow query, forwarded to the output
        node so the synthesizer_fn can combine it with tool outputs.
        """
        t0 = time.time()
        ok, err, out, in_args = True, None, None, None

        if n.kind == "tool":
            if n.tool_ref is None:
                ok, err = False, "missing_tool_ref"
            else:
                server, tool = n.tool_ref
                in_args = _resolve_args(n.params, memory)
                # Registry lookup is read-only — safe in threads.
                spec = self.registry.tools.get((server, tool))
                if spec is not None:
                    in_args = _coerce_args(in_args, spec.input_schema or {})
                if do_timeout:
                    ok, err = False, "timeout_injected"
                elif do_error:
                    ok, err = False, "tool_error_injected"
                else:
                    ok, err, out = self._call_tool(server, tool, in_args)

        elif n.kind == "output":
            # Aggregate all upstream leaf outputs into a single dict keyed by node_id
            aggregated = {req_id: memory[req_id] for req_id in (n.requires or []) if req_id in memory}
            # If a synthesizer is configured, call it to produce a final LLM-generated answer
            if self._synthesizer_fn is not None:
                try:
                    synthesized = self._synthesizer_fn(query or {}, aggregated)
                    out = {"synthesized_output": synthesized, "tool_outputs": aggregated}
                except Exception as exc:
                    print(f"[executor] WARNING: synthesizer_fn failed: {exc}; falling back to raw aggregation", flush=True)
                    out = aggregated
            else:
                out = aggregated
            ok = True

        elif n.kind == "validator":
            req = n.requires[0] if n.requires else None
            if req and req not in memory:
                ok, err = False, "validator_missing_upstream"

        latency = int((time.time() - t0) * 1000) + jitter_ms
        step_log = StepLog(
            node_id=n.node_id,
            kind=n.kind,
            tool=f"{n.tool_ref[0]}.{n.tool_ref[1]}" if n.tool_ref else None,
            ok=ok,
            error_type=err,
            latency_ms=latency,
            input_args=in_args,
            output=out,
        )

        if not ok:
            error_detail = {
                "node_id": n.node_id,
                "tool": step_log.tool,
                "error_type": err,
                "input_args": in_args,
            }
            return step_log, None, error_detail

        # Build memory entry for this node
        mem_entry: Dict[str, Any] = (
            dict(out) if isinstance(out, dict) else {"content": out}
        ) if out is not None else {}
        for name, path in (n.produces or {}).items():
            extracted = _extract_path(out, path)
            # Retry with "content." prefix (tool outputs are often wrapped in
            # {"content": "<json_string>"}, so paths like "results[0].id" need
            # to go through "content.results[0].id").
            if extracted is None and not path.startswith("content"):
                extracted = _extract_path(out, "content." + path)
            # Last resort: deep-search the raw content JSON for the final key
            # segment (handles cases where the LLM guessed the wrong path but
            # the field exists somewhere in the nested structure).
            if extracted is None:
                last_key = path.rstrip("]").rsplit(".", 1)[-1].split("[")[0]
                raw = (out or {}).get("content") if isinstance(out, dict) else out
                extracted = _deep_find(raw, last_key)
                if extracted is not None:
                    print(
                        f"[executor] WARNING: produces '{name}' path '{path}' failed "
                        f"for node {n.node_id}; resolved via deep-search on key '{last_key}'",
                        flush=True,
                    )
            if extracted is not None:
                mem_entry[name] = extracted
            else:
                raw_preview = ""
                if isinstance(out, dict) and "content" in out:
                    c = out["content"]
                    raw_preview = f"; content={str(c)[:200]!r}"
                print(
                    f"[executor] WARNING: produces '{name}' path '{path}' could not "
                    f"be extracted from node {n.node_id} output — downstream refs will "
                    f"fall back to raw content{raw_preview}",
                    flush=True,
                )

        return step_log, {n.node_id: mem_entry}, None

    # ------------------------------------------------------------------
    # Tool dispatch — override in AdaptedExecutor
    # ------------------------------------------------------------------

    def _call_tool(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Built-in echo stub.  Subclass and override to route to a real backend.
        Return (ok, error_string_or_None, output_dict_or_None).
        """
        return True, None, {"server": server, "tool": tool, "echo": args}


# ---------------------------------------------------------------------------
# AdaptedExecutor
# ---------------------------------------------------------------------------

class AdaptedExecutor(MCPExecutor):
    """
    Concrete executor that wires a BenchmarkAdapter's call_tool and
    score_result into the standard MCPExecutor interface.

    Constructed exclusively by BenchmarkAdapterRunner._make_executor_factory()
    so a single class definition is shared across run_search and validate.

    score_fn here is the pre-composed function from BenchmarkAdapter._make_score_fn():
    it already handles the try-adapter-judge-first, fall-back-to-structural-proxy
    logic, so AdaptedExecutor does not re-wrap it.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sampler: PerturbationSampler,
        timeout_s: float,
        tool_fn: Callable,           # adapter.call_tool
        score_fn: Optional[ScoringFn] = None,
        synthesizer_fn: Optional[Callable] = None,
    ):
        super().__init__(
            registry=registry,
            sampler=sampler,
            timeout_s=timeout_s,
            score_fn=score_fn,       # None → _default_scoring_fn in MCPExecutor.__init__
            synthesizer_fn=synthesizer_fn,
        )
        self._tool_fn = tool_fn

    def _call_tool(self, server, tool, args):
        return self._tool_fn(server, tool, args)