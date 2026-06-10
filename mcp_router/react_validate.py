"""
mcp_router/react_validate.py
----------------------------
ReAct (Reasoning + Acting) validation runner.

Implements the ReAct prompting method from:
    "ReAct: Synergizing Reasoning and Acting in Language Models"
    Yao et al., ICLR 2023  (https://arxiv.org/abs/2210.03629)

Instead of pre-planning a full tool-call graph (as evolution.py does),
ReAct interleaves free-form reasoning traces (Thought) with grounded
tool calls (Action) and their results (Observation) in a sequential loop.
At each step the LLM sees the full trajectory so far and can adapt its
plan in light of what it has already observed.

Loop structure (mirrors the paper's Figure 2)
---------------------------------------------
    Thought_t  : the model's reasoning about the current state
    Action_t   : one tool call chosen by the model, or Finish[answer]
    Observation_t : the tool's return value

This continues until:
  - the model emits Finish[answer], or
  - max_steps is reached (trajectory is truncated and scored as-is)

After the trajectory is complete, a real CompiledWorkflow is built from the
observed (tool, args, output) steps so the same LLMJudge used by the
search/validate stages can score the run with an identical interface.

Integration with the existing pipeline
---------------------------------------
ReactValidationRunner mirrors BenchmarkAdapterRunner.validate() so it
can be called from run_pipeline.py or scripts/react_validate.py in place
of the standard single-plan validate stage.

Output format
-------------
Results are written to  results/react_val_history_<model>.jsonl
Each line is a JSON object with the same top-level keys as
val_workflow_history_<model>.jsonl so existing analysis tools work.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        desc = kwargs.get("desc", "")
        items = list(it)
        total = len(items)
        for i, x in enumerate(items):
            if i % max(1, total // 10) == 0:
                print(f"  [{desc}] {i}/{total}")
            yield x

from .dsl import Node, GraphTemplate, PromptParams
from .compiler import CompiledWorkflow
from .executor import ExecResult, StepLog, MetricBreakdown


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReActStep:
    """One Thought / Action / Observation triple."""
    thought:      str
    action:       Optional[str]            # "<server>/<tool>" or None when finishing
    action_input: Optional[Dict[str, Any]]
    observation:  Optional[str]            # raw string from the tool (or None)
    ok:           bool = True
    latency_ms:   int = 0
    error:        Optional[str] = None
    token_cost:   Optional[Dict[str, Any]] = None   # LLM cost for this step's call


@dataclass
class ReActTrajectory:
    """Full trajectory for one query."""
    query_id:     str
    query:        Dict[str, Any]
    steps:        List[ReActStep] = field(default_factory=list)
    finish:       Optional[str] = None   # the model's final answer
    truncated:    bool = False           # True if max_steps was hit
    token_cost:   Optional[Dict[str, Any]] = None
    score:        float = 0.0
    score_detail: Optional[MetricBreakdown] = None


# ---------------------------------------------------------------------------
# ReAct system / user prompts (paper §3 format)
# ---------------------------------------------------------------------------

_REACT_SYSTEM = """\
You are a helpful assistant that solves tasks step by step using available tools.

At every step output EXACTLY one of these two formats:

Format A — take a tool action:
Thought: <your reasoning about what to do next>
Action: <server_name>/<tool_name>
Action Input: <valid JSON object with the tool arguments>

Format B — finish when you have the final answer:
Thought: <your reasoning about why you have the answer>
Finish: <your final answer as plain text>

Rules:
- Output ONLY the fields shown above — no extra prose, no markdown fences.
- Action must be exactly "<server>/<tool>" using the names from the catalog.
- Action Input must be a single-line valid JSON object.
- Do not repeat an identical (Action, Action Input) pair already tried.
- If a tool returns an error, try a different approach or a different tool.
"""


def _build_react_user_prompt(
    task_text:            str,
    tool_catalog:         Dict[str, Dict[str, Any]],
    trajectory:           List[ReActStep],
    max_steps:            int,
    consecutive_failures: int = 0,
) -> str:
    """
    Build the user-turn prompt the LLM sees at every ReAct step.

    Contains:
      1. The original task description
      2. The flattened tool catalog (server / tool / description / schema)
      3. The full Thought / Action / Observation history so far
      4. A CORRECTION REQUIRED block when the last step(s) failed, with an
         explicit directive to change approach — not just a passive observation.
    """
    # ── Tool catalog ──────────────────────────────────────────────────────
    catalog_lines: List[str] = []
    for server, tools in tool_catalog.items():
        for tool_name, meta in tools.items():
            desc   = meta.get("description", "")
            schema = meta.get("input_schema") or meta.get("inputSchema") or {}
            props  = schema.get("properties", {})
            req    = schema.get("required", [])
            param_strs = []
            for pname, pmeta in props.items():
                ptype = pmeta.get("type", "any")
                pdesc = pmeta.get("description", "")
                flag  = " (required)" if pname in req else ""
                param_strs.append(f"      {pname}: {ptype}{flag} — {pdesc}")
            params_text = "\n".join(param_strs) if param_strs else "      (no parameters)"
            catalog_lines.append(
                f"  {server}/{tool_name}\n"
                f"    Description: {desc}\n"
                f"    Parameters:\n{params_text}"
            )
    catalog_text = "\n".join(catalog_lines)

    # ── Trajectory so far ─────────────────────────────────────────────────
    history_lines: List[str] = []
    for step in trajectory:
        history_lines.append(f"Thought: {step.thought}")
        if step.action is not None:
            ai = json.dumps(step.action_input or {})
            history_lines.append(f"Action: {step.action}")
            history_lines.append(f"Action Input: {ai}")
            obs = step.observation if step.observation is not None else "(no output)"
            history_lines.append(f"Observation: {obs}")
    history_text = "\n".join(history_lines)

    # ── Correction block — shown only when the last step(s) failed ────────
    correction_block = ""
    if consecutive_failures > 0 and trajectory:
        last = trajectory[-1]
        correction_lines = [
            "=" * 60,
            f"CORRECTION REQUIRED — execution failure {consecutive_failures} in a row.",
            f"  Failed action : {last.action}",
            f"  With input    : {json.dumps(last.action_input or {})}",
            f"  Error         : {last.error}",
            "",
        ]
        if consecutive_failures == 1:
            correction_lines += [
                "The tool call above failed. Analyse the error carefully:",
                "  - Is the tool name spelled exactly as it appears in the catalog?",
                "  - Are the required parameters present and correctly typed?",
                "  - Would a different tool achieve the same goal?",
                "Try a different action or corrected parameters on your next step.",
            ]
        else:
            correction_lines += [
                f"You have failed {consecutive_failures} times in a row.",
                "You MUST change your strategy completely:",
                "  - Do NOT repeat the same action or the same parameters.",
                "  - Pick a DIFFERENT tool from the catalog.",
                "  - If no suitable tool exists, use Finish to report what you know so far.",
                "Valid tool names (copy exactly): "
                + ", ".join(
                    f"{srv}/{t}"
                    for srv, tools in tool_catalog.items()
                    for t in tools
                ),
            ]
        correction_lines.append("=" * 60)
        correction_block = "\n".join(correction_lines) + "\n\n"

    steps_left = max_steps - len(trajectory)
    return (
        f"Task: {task_text}\n\n"
        f"Available tools:\n{catalog_text}\n\n"
        + (f"Trajectory so far:\n{history_text}\n\n" if history_text else "")
        + correction_block
        + f"Steps remaining: {steps_left}\n\n"
        "Continue the trajectory. Output the next Thought + Action or Thought + Finish."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_ACTION_RE    = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)
_ACTION_IN_RE = re.compile(r"Action\s+Input\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE   = re.compile(r"Thought\s*:\s*(.+?)(?=\nAction|\nFinish|$)", re.IGNORECASE | re.DOTALL)
_FINISH_RE    = re.compile(r"Finish\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_react_response(text: str) -> Tuple[str, Optional[str], Optional[Dict], Optional[str]]:
    """
    Parse one LLM turn into (thought, action, action_input, finish).

    Returns
    -------
    thought      : str  — reasoning (may be empty)
    action       : str | None  — "<server>/<tool>"
    action_input : dict | None — parsed JSON from Action Input
    finish       : str | None  — final answer if Finish was emitted
    """
    # Strip chain-of-thought blocks emitted by reasoning models (DeepSeek-R1,
    # QwQ, etc.) before looking for the structured ReAct fields.  The model
    # often puts ALL its output inside <think>…</think> and then emits the
    # structured response after the closing tag; stripping the block first
    # makes the regexes below work correctly.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    thought = ""
    tm = _THOUGHT_RE.search(text)
    if tm:
        thought = tm.group(1).strip()

    fm = _FINISH_RE.search(text)
    if fm:
        return thought, None, None, fm.group(1).strip()

    am  = _ACTION_RE.search(text)
    aim = _ACTION_IN_RE.search(text)

    action       = am.group(1).strip() if am else None
    action_input = None
    if aim:
        raw = aim.group(1).strip()
        raw = re.split(r'\nObservation\s*:', raw, maxsplit=1)[0].strip()
        try:
            action_input = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            m_braces = re.search(r'\{.*\}', raw, re.DOTALL)
            if m_braces:
                try:
                    action_input = json.loads(m_braces.group())
                except (json.JSONDecodeError, ValueError):
                    action_input = {}
            else:
                action_input = {}

    return thought, action, action_input, None


# ---------------------------------------------------------------------------
# Build a real CompiledWorkflow from the ReAct trajectory
# ---------------------------------------------------------------------------

def _build_workflow_from_trajectory(traj: ReActTrajectory) -> CompiledWorkflow:
    """
    Construct a CompiledWorkflow whose nodes reflect each tool call in the
    ReAct trajectory.

    The workflow is a linear chain (step_0 → step_1 → … → output) since
    ReAct is inherently sequential.  Nodes that had action=None (the Finish
    step) are skipped — only actual tool calls become graph nodes.

    This real CompiledWorkflow is passed to adapter.score_result() so the
    LLMJudge receives the same wf.query / wf.graph interface it expects,
    making ReAct scores directly comparable to search/validate scores.
    """
    tool_nodes: List[Node] = []
    edges: List[Tuple[str, str]] = []
    prev_id: Optional[str] = None

    for i, step in enumerate(traj.steps):
        if step.action is None or step.action == "__format_error__":
            # Finish step or parse-failure placeholder — no tool call
            continue

        node_id = f"react_step_{i}"
        parts   = step.action.strip().split("/", 1)
        server  = parts[0].strip() if len(parts) == 2 else ""
        tool    = parts[1].strip() if len(parts) == 2 else parts[0].strip()

        requires = [prev_id] if prev_id else []
        node = Node(
            node_id  = node_id,
            kind     = "tool",
            tool_ref = (server, tool),
            params   = step.action_input or {},
            requires = requires,
            produces = {},
        )
        tool_nodes.append(node)
        if prev_id:
            edges.append((prev_id, node_id))
        prev_id = node_id

    # ── Output node ───────────────────────────────────────────────────────
    leaf_ids = [prev_id] if prev_id else []
    out_node = Node(node_id="output", kind="output", requires=leaf_ids)
    all_nodes = tool_nodes + [out_node]
    all_edges = edges + [(lid, "output") for lid in leaf_ids]

    # ── GraphTemplate (template excludes the output aggregator node) ──────
    traj_hash = hashlib.sha256(
        json.dumps({"qid": traj.query_id, "n_steps": len(traj.steps)},
                   sort_keys=True).encode()
    ).hexdigest()[:32]

    graph = GraphTemplate(
        template_id   = traj_hash,
        nodes         = tool_nodes,   # template = tool nodes only
        edges         = edges,
        description   = f"ReAct trajectory for query {traj.query_id}",
        heuristic     = True,
        token_cost    = traj.token_cost,
    )

    return CompiledWorkflow(
        action_id = traj_hash,
        nodes     = all_nodes,
        edges     = all_edges,
        policy    = PromptParams(max_steps=max(len(traj.steps), 1)),
        query     = traj.query,
        graph     = graph,
    )


# ---------------------------------------------------------------------------
# Build ExecResult from the trajectory
# ---------------------------------------------------------------------------

def _trajectory_to_exec_result(traj: ReActTrajectory) -> ExecResult:
    """
    Convert a ReActTrajectory into an ExecResult compatible with the
    LLMJudge.

    StepLog.tool is stored as "server.tool" (dot-separated) to match the
    format written by MCPExecutor and expected by _score_with_llm_judge:
        server, _, tool_name = (log.tool or "").partition(".")
    """
    logs: List[StepLog] = []
    for i, step in enumerate(traj.steps):
        if step.action is None or step.action == "__format_error__":
            # Finish step or parse-failure placeholder — no tool log entry
            continue

        # "server/tool" → "server.tool" for judge compatibility
        tool_dot = step.action.replace("/", ".", 1)

        logs.append(StepLog(
            node_id    = f"react_step_{i}",
            kind       = "tool",
            tool       = tool_dot,
            ok         = step.ok,
            error_type = step.error if not step.ok else None,
            latency_ms = step.latency_ms,
            input_args = step.action_input,
            output     = {
                "content": step.observation,
                "thought": step.thought,      # extra ReAct context for the judge
            } if step.observation else None,
        ))

    # Final output: prefer the explicit Finish answer, else last observation
    final_output: Optional[Dict[str, Any]] = None
    if traj.finish is not None:
        final_output = {"content": traj.finish}
    elif traj.steps:
        last = traj.steps[-1]
        if last.observation:
            final_output = {"content": last.observation}

    all_ok = all(s.ok for s in traj.steps if s.action is not None)
    return ExecResult(
        ok           = all_ok and traj.finish is not None,
        logs         = logs,
        final_output = final_output,
        violation    = "max_steps_exceeded" if traj.truncated else None,
    )


# ---------------------------------------------------------------------------
# ReactValidationRunner — mirrors BenchmarkAdapterRunner.validate()
# ---------------------------------------------------------------------------

class ReactValidationRunner:
    """
    Runs the ReAct loop on every query in D_val and scores the result using
    the same LLMJudge that evaluate_task_performance uses in the search /
    validate stages.

    Parameters
    ----------
    adapter   : BenchmarkAdapter
        Provides call_tool(), score_result(), and load_datasets().
    planner   : QueryPlanner
        Provides _dispatch_llm_call() for LLM access and the tool catalog.
    max_steps : int
        Maximum Thought/Action/Observation rounds before truncating.
    on_record : callable, optional
        on_record(qid, rec_dict) — called after each query so results are
        persisted incrementally (same streaming pattern as run_pipeline.py).
    """

    def __init__(
        self,
        adapter,
        planner,
        max_steps: int = 15,
        tool_retries: int = 10,
        parse_retries: int = 3,
        max_tokens: int = 2048,
        on_record: Optional[Callable] = None,
    ):
        self.adapter       = adapter
        self.planner       = planner
        self.max_steps     = max_steps
        self.tool_retries  = tool_retries
        self.parse_retries = parse_retries
        self.max_tokens    = max_tokens
        self._on_record    = on_record
        self.tool_catalog: Dict[str, Dict[str, Any]] = getattr(planner, "tool_catalog", {})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, D_val: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Run the ReAct loop on every query in D_val.

        Returns a list of per-query result dicts (sorted by score descending)
        using the same keys as BenchmarkAdapterRunner.validate().
        """
        if D_val is None:
            _, D_val = self.adapter.load_datasets()

        results: List[Dict[str, Any]] = []
        for i, query in enumerate(tqdm(D_val, desc="react-validate")):
            qid  = str(query.get("id", i))
            traj = self._run_query(query, qid)
            rec  = self._trajectory_to_record(traj)

            if self._on_record is not None:
                try:
                    self._on_record(qid, rec)
                except Exception as exc:
                    print(f"[react] on_record failed for {qid}: {exc}", flush=True)

            results.append({
                "query_id":  qid,
                "action_id": traj.score_detail and f"react_{qid}" or f"react_{qid}",
                "ok":        rec.get("ok", False),
                "score":     traj.score,
                "score_detail": rec.get("score_detail"),
                "violation": rec.get("violation"),
                "n_steps":   len([s for s in traj.steps if s.action not in (None, "__format_error__")]),
                "finish":    traj.finish,
            })
            print(
                f"[react] {qid}: steps={len(traj.steps)}  "
                f"ok={rec.get('ok')}  score={traj.score:.3f}  "
                f"finish={'yes' if traj.finish else 'no'}",
                flush=True,
            )

        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Per-query ReAct loop
    # ------------------------------------------------------------------

    def _run_query(self, query: Dict[str, Any], qid: str) -> ReActTrajectory:
        task_text          = query.get("text", str(query))
        traj               = ReActTrajectory(query_id=qid, query=query)
        total_cost: Dict[str, Any] = {}
        consecutive_failures      = 0   # consecutive tool-execution failures
        consecutive_parse_failures = 0  # consecutive response-parse failures

        for step_idx in range(self.max_steps):
            # ── Build prompt ──────────────────────────────────────────
            # Pass consecutive_failures so the prompt injects a targeted
            # correction directive when the last step(s) failed.
            user_text = _build_react_user_prompt(
                task_text            = task_text,
                tool_catalog         = self.tool_catalog,
                trajectory           = traj.steps,
                max_steps            = self.max_steps,
                consecutive_failures = consecutive_failures,
            )

            # ── LLM call ──────────────────────────────────────────────
            try:
                response_text, step_cost = self.planner._dispatch_llm_call(
                    system_text = _REACT_SYSTEM,
                    user_text   = user_text,
                    max_tokens  = self.max_tokens,
                    temperature = 0.0,
                )
                total_cost = _merge_cost(total_cost, step_cost)
            except Exception as exc:
                exc_str = str(exc)
                if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                    raise
                print(f"[react] {qid} step {step_idx}: LLM call failed ({exc}), stopping",
                      flush=True)
                traj.truncated = True
                break

            _pt = step_cost.get("prompt_tokens", 0) or 0
            _ct = step_cost.get("completion_tokens", 0) or 0
            print(
                f"[react] {qid} step {step_idx}: tokens prompt={_pt} completion={_ct}",
                flush=True,
            )

            # ── Parse ─────────────────────────────────────────────────
            thought, action, action_input, finish = _parse_react_response(response_text)

            if finish is not None:
                traj.steps.append(ReActStep(
                    thought=thought, action=None, action_input=None,
                    observation=None, ok=True, token_cost=step_cost,
                ))
                traj.finish = finish
                break

            if action is None:
                consecutive_parse_failures += 1
                print(
                    f"[react] {qid} step {step_idx}: could not parse action "
                    f"({consecutive_parse_failures}/{self.parse_retries}) — ",
                    flush=True,
                )
                if consecutive_parse_failures >= self.parse_retries:
                    traj.truncated = True
                    break
                # Feed a format-error observation back so the model can self-correct.
                traj.steps.append(ReActStep(
                    thought      = thought or "",
                    action       = "__format_error__",
                    action_input = {},
                    observation  = (
                        "FORMAT ERROR: your response did not contain a valid Action + "
                        "Action Input or Finish block. "
                        "Output EXACTLY one of:\n"
                        "  Thought: <reasoning>\n  Action: <server>/<tool>\n  Action Input: {\"param\": value}\n"
                        "OR\n"
                        "  Thought: <reasoning>\n  Finish: <your final answer>"
                    ),
                    ok           = False,
                    error        = "parse_failure",
                    token_cost   = step_cost,
                ))
                consecutive_failures += 1
                continue

            # ── Execute tool ──────────────────────────────────────────
            # Failure is surfaced as an Observation AND triggers a targeted
            # correction block in the next prompt so the LLM knows it must
            # change approach.  We abort only after `tool_retries`
            # consecutive failures to prevent an infinite error loop.
            t0           = time.monotonic()
            obs, ok, err = self._call_tool(action, action_input or {})
            latency      = int((time.monotonic() - t0) * 1000)

            if ok:
                consecutive_failures       = 0
                consecutive_parse_failures = 0
            else:
                consecutive_failures += 1
                print(
                    f"[react] {qid} step {step_idx}: execution failure "
                    f"({consecutive_failures}/{self.tool_retries}) — {err}",
                    flush=True,
                )

            traj.steps.append(ReActStep(
                thought      = thought,
                action       = action,
                action_input = action_input or {},
                observation  = obs,
                ok           = ok,
                latency_ms   = latency,
                error        = err,
                token_cost   = step_cost,
            ))

            if consecutive_failures >= self.tool_retries:
                print(
                    f"[react] {qid}: {self.tool_retries} consecutive execution "
                    f"failures — stopping loop.",
                    flush=True,
                )
                traj.truncated = True
                break

        else:
            traj.truncated = True

        traj.token_cost = total_cost if total_cost else None

        # ── Build workflow + exec_result, then score ──────────────────
        wf       = _build_workflow_from_trajectory(traj)
        exec_res = _trajectory_to_exec_result(traj)

        try:
            mb = self.adapter.score_result(exec_res, wf)
            if mb is not None:
                score_vals = [
                    mb.task_fulfillment,
                    mb.grounding,
                    mb.tool_appropriateness,
                    mb.parameter_accuracy,
                    mb.dependency_awareness,
                    mb.parallelism_and_efficiency,
                ]
                traj.score        = float(np.mean([v for v in score_vals if v is not None] or [0.0]))
                traj.score_detail = mb
        except Exception as exc:
            print(f"[react] {qid}: scoring failed ({exc})", flush=True)

        return traj

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _call_tool(
        self,
        action: str,
        action_input: Dict[str, Any],
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Execute one tool call via the adapter.
        action is expected as "<server>/<tool>"; falls back when prefix absent.
        Returns (observation_str, ok, error_or_None).
        """
        parts = action.strip().split("/", 1)
        if len(parts) == 2:
            server, tool = parts[0].strip(), parts[1].strip()
        else:
            tool   = parts[0].strip()
            server = self._infer_server(tool)

        try:
            ok, error, output = self.adapter.call_tool(server, tool, action_input)
        except Exception as exc:
            return f"Tool call raised an exception: {exc}", False, str(exc)

        if ok:
            obs = json.dumps(output, default=str) if output is not None else "(empty output)"
            return obs, True, None
        else:
            return f"Error: {error}", False, error

    def _infer_server(self, tool_name: str) -> str:
        """Find the first server that has a tool with this name."""
        for server, tools in self.tool_catalog.items():
            if tool_name in tools:
                return server
        return ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _trajectory_to_record(self, traj: ReActTrajectory) -> Dict[str, Any]:
        """
        Convert a ReActTrajectory to a JSON-serialisable dict that mirrors
        the format written by run_pipeline.py for downstream compatibility.
        """
        exec_res = _trajectory_to_exec_result(traj)
        wf       = _build_workflow_from_trajectory(traj)

        score_detail_dict: Optional[Dict] = None
        if traj.score_detail is not None:
            try:
                score_detail_dict = dict(traj.score_detail.__dict__)
            except AttributeError:
                pass

        return {
            "candidate_id":  f"react_{traj.query_id}",
            "query_id":      traj.query_id,
            "action_id":     wf.action_id,
            "score":         traj.score,
            "score_detail":  score_detail_dict,
            "feasible":      True,
            "feasibility_reasons": [],
            "ok":            exec_res.ok,
            "violation":     exec_res.violation,
            "n_steps":       len([s for s in traj.steps if s.action is not None]),
            "truncated":     traj.truncated,
            "finish":        traj.finish,
            "token_cost":    traj.token_cost,
            "action": {
                "q": traj.query,
                "e": [],
                "p": {"max_steps": self.max_steps},
            },
            "workflow": {
                "description": wf.graph.description,
                "heuristic":   wf.graph.heuristic,
                "nodes": [
                    {
                        "node_id":  n.node_id,
                        "kind":     n.kind,
                        "server":   n.tool_ref[0] if n.tool_ref else None,
                        "tool":     n.tool_ref[1] if n.tool_ref else None,
                        "params":   dict(n.params) if n.params else {},
                        "requires": list(n.requires) if n.requires else [],
                    }
                    for n in wf.nodes
                ],
                "edges": [list(e) for e in wf.edges],
            },
            "trajectory": [
                {
                    "thought":      s.thought,
                    "action":       s.action,
                    "action_input": s.action_input,
                    "observation":  s.observation,
                    "ok":           s.ok,
                    "latency_ms":   s.latency_ms,
                    "error":        s.error,
                    "token_cost":   s.token_cost,
                }
                for s in traj.steps
            ],
            "execution_result": {
                "ok":           exec_res.ok,
                "violation":    exec_res.violation,
                "final_output": exec_res.final_output,
                "logs": [
                    {
                        "node_id":    lg.node_id,
                        "kind":       lg.kind,
                        "tool":       lg.tool,
                        "ok":         lg.ok,
                        "error_type": lg.error_type,
                        "latency_ms": lg.latency_ms,
                        "output":     lg.output,
                    }
                    for lg in exec_res.logs
                ],
            },
        }


# ---------------------------------------------------------------------------
# Token cost accumulator
# ---------------------------------------------------------------------------

def _merge_cost(a: dict, b: dict) -> dict:
    """Accumulate token usage across all LLM calls in the trajectory."""
    if not a:
        return dict(b) if b else {}
    if not b:
        return dict(a)
    KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
    merged = {}
    for k in KEYS:
        av = a.get(k) or 0
        bv = b.get(k) or 0
        merged[k] = (av + bv) if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else (bv or av)
    return merged


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run ReAct validation from the command line.

    Example
    -------
        PYTHONPATH=. python3 -m mcp_router.react_validate \\
            --benchmark mcpbench \\
            --provider bedrock \\
            --model us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
            --max-steps 10 \\
            --output-dir results/react
    """
    import argparse
    import os

    p = argparse.ArgumentParser(description="ReAct validation runner")
    p.add_argument("--benchmark",      default="mcpbench",
                   choices=["toy", "assetopsbench", "mcpbench"])
    p.add_argument("--provider",       default="bedrock",
                   choices=["bedrock", "openrouter", "azure", "base_url"])
    p.add_argument("--model",          default=None)
    p.add_argument("--base-url",       default=None)
    p.add_argument("--api-key",        default=None)
    p.add_argument("--judge-provider", default=None)
    p.add_argument("--judge-model",    default=None)
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-api-key",  default=None)
    p.add_argument("--max-steps",      type=int, default=15)
    p.add_argument("--tool-retries",   type=int, default=10,
                   help="Consecutive execution failures before aborting a query "
                        "(each failure is fed back as an Observation so the LLM can self-correct).")
    p.add_argument("--output-dir",     default="results/react")
    p.add_argument("--timeout",        type=float, default=15.0)
    p.add_argument("--complexity",     default="all",
                   choices=["all", "single", "2server", "3server"])
    args = p.parse_args()

    if args.benchmark == "toy":
        from mcp_router.adapters.toy import ToyAdapter
        adapter = ToyAdapter()
    elif args.benchmark == "assetopsbench":
        from mcp_router.adapters.assetopsbench import AssetOpsBenchAdapter
        adapter = AssetOpsBenchAdapter()
    elif args.benchmark == "mcpbench":
        from mcp_router.adapters.mcpbench import MCPBenchAdapter
        adapter = MCPBenchAdapter(
            provider           = args.provider,
            template_model     = args.model,
            template_base_url  = args.base_url,
            template_api_key   = args.api_key or os.getenv("OPENROUTER_API_KEY"),
            judge_provider     = args.judge_provider or args.provider,
            judge_model        = args.judge_model    or args.model,
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY"),
            local_base_url     = args.judge_base_url or args.base_url,
            local_api_key      = args.judge_api_key  or args.api_key,
            task_complexity    = args.complexity,
            timeout_s          = args.timeout,
        )
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    registry  = adapter.build_registry()
    planner   = adapter.build_query_planner(registry)
    _, D_val  = adapter.load_datasets()
    model_tag = (args.model or "unknown").split("/")[-1]

    os.makedirs(args.output_dir, exist_ok=True)
    hist_path = os.path.join(args.output_dir, f"react_val_history_{model_tag}.jsonl")

    _hist_file = open(hist_path, "a")

    def _on_record(_qid: str, rec: Dict[str, Any]) -> None:
        rec["_meta"] = {"is_final": True, "phase": "react_validate"}
        _hist_file.write(json.dumps(rec) + "\n")
        _hist_file.flush()

    print("=" * 64)
    print(f"  ReAct Validation  |  benchmark: {args.benchmark}")
    print(f"  model: {args.model}   provider: {args.provider}")
    print(f"  max_steps: {args.max_steps}   tool_retries: {args.tool_retries}   queries: {len(D_val)}")
    print("=" * 64)

    runner = ReactValidationRunner(
        adapter      = adapter,
        planner      = planner,
        max_steps    = args.max_steps,
        tool_retries = args.tool_retries,
        on_record    = _on_record,
    )

    try:
        results = runner.run(D_val=D_val)
    finally:
        _hist_file.close()

    print(f"\n{'Rank':<5} {'query_id':<20} {'Score':>7} {'Steps':>6} {'OK':>5}")
    print("-" * 45)
    for i, r in enumerate(results[:20], 1):
        print(f"{i:<5} {str(r['query_id']):<20} "
              f"{r['score']:>7.3f} {r['n_steps']:>6} {str(r['ok']):>5}")

    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    ok_rate    = sum(1 for r in results if r["ok"])  / len(results) if results else 0.0
    print(f"\nMean score: {mean_score:.4f}   OK rate: {ok_rate:.1%}")

    report_path = os.path.join(args.output_dir, "react_validation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "benchmark":  args.benchmark,
            "model":      args.model,
            "provider":   args.provider,
            "max_steps":  args.max_steps,
            "mean_score": mean_score,
            "ok_rate":    ok_rate,
            "results":    results,
        }, f, indent=2)

    print(f"\nHistory  → {hist_path}")
    print(f"Report   → {report_path}")


if __name__ == "__main__":
    main()
