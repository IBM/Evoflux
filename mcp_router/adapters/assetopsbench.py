
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark_adapter import BenchmarkAdapter
from ..dsl import ToolSpec, ToolRegistry, GraphTemplate, Node, Action, PromptParams
from ..executor import ExecResult
from ..compiler import CompiledWorkflow
from ..metrics import MetricBreakdown


"""
mcp_router/adapters/assetopsbench.py
-------------------------------------
BenchmarkAdapter for IBM AssetOpsBench (Industry 4.0).
https://github.com/IBM/AssetOpsBench

Setup:
    # 1. Clone and start the Docker stack
    git clone https://github.com/IBM/AssetOpsBench
    cd AssetOpsBench
    docker-compose -f benchmark/docker-compose.yml up

    # 2. (Optional) Install HF datasets for scenario loading
    pip install datasets

    # 3. Run
    from mcp_router.adapters.assetopsbench import AssetOpsBenchAdapter
    from mcp_router.benchmark_adapter import BenchmarkAdapterRunner
    runner = BenchmarkAdapterRunner(adapter=AssetOpsBenchAdapter())
    pop = runner.run_search()

Port map (default Docker Compose):
    IoT agent   → http://localhost:8001
    FMSA agent  → http://localhost:8002
    TSFM agent  → http://localhost:8003
    WO agent    → http://localhost:8004
    LLM judge   → http://localhost:8005
"""


# Domain → Docker endpoint
_ENDPOINTS: Dict[str, str] = {
    "iot":  "http://localhost:8001",
    "fmsa": "http://localhost:8002",
    "tsfm": "http://localhost:8003",
    "wo":   "http://localhost:8004",
}
_JUDGE_URL = "http://localhost:8005/judge"

# Which template to use per AssetOpsBench scenario type
_TEMPLATE_FOR_TYPE: Dict[str, str] = {
    "IoT":       "iot_single",
    "FMSA":      "fmsa_single",
    "TSFM":      "tsfm_single",
    "Workorder": "e2e_iot_fmsa_wo",
}


class AssetOpsBenchAdapter(BenchmarkAdapter):
    """
    Connects mcp_router to the IBM AssetOpsBench Docker stack.
    All four abstract methods are wired to the real agents and HF dataset.
    """

    def __init__(
        self,
        endpoints: Optional[Dict[str, str]] = None,
        judge_url: str = _JUDGE_URL,
        timeout_s: float = 30.0,
    ):
        self.endpoints  = endpoints or _ENDPOINTS
        self.judge_url  = judge_url
        self.timeout_s  = timeout_s

    # ------------------------------------------------------------------
    # 1. ToolRegistry
    # ------------------------------------------------------------------
    def build_registry(self) -> ToolRegistry:
        def obj():
            return {"type": "object", "properties": {}, "additionalProperties": True}

        tools = {}

        for t in ["get_sites", "get_assets", "get_sensors", "get_history"]:
            tools[("iot", t)] = ToolSpec("iot", t, obj(), obj())

        for t in ["get_sensors", "get_failure_modes", "get_failure_sensor_mapping"]:
            tools[("fmsa", t)] = ToolSpec("fmsa", t, obj(), obj())

        for t in ["forecasting", "timeseries_anomaly_detection",
                  "forecasting_tune", "forecasting_evaluation"]:
            tools[("tsfm", t)] = ToolSpec("tsfm", t, obj(), obj())

        for t in ["generate_work_order", "get_work_orders",
                  "predict_next_work_order_probability"]:
            tools[("wo", t)] = ToolSpec("wo", t, obj(), obj())

        return ToolRegistry(tools=tools)

    # ------------------------------------------------------------------
    # 2. Templates
    # ------------------------------------------------------------------
    def build_templates(self) -> Dict[str, GraphTemplate]:
        return {
            # ── IoT single-domain ──────────────────────────────────────
            "iot_single": GraphTemplate(
                template_id="iot_single",
                nodes=[
                    Node("t1", "tool", ("iot", "get_sites")),
                    Node("t2", "tool", ("iot", "get_assets"),  requires=["t1"]),
                    Node("t3", "tool", ("iot", "get_history"), requires=["t2"]),
                ],
                edges=[("t1", "t2"), ("t2", "t3")],
            ),
            # ── FMSA single-domain ─────────────────────────────────────
            "fmsa_single": GraphTemplate(
                template_id="fmsa_single",
                nodes=[
                    Node("t1", "tool", ("fmsa", "get_sensors")),
                    Node("t2", "tool", ("fmsa", "get_failure_modes"),         requires=["t1"]),
                    Node("t3", "tool", ("fmsa", "get_failure_sensor_mapping"), requires=["t2"]),
                ],
                edges=[("t1", "t2"), ("t2", "t3")],
            ),
            # ── TSFM single-domain ─────────────────────────────────────
            "tsfm_single": GraphTemplate(
                template_id="tsfm_single",
                nodes=[
                    Node("t1", "tool", ("tsfm", "forecasting")),
                    Node("t2", "tool", ("tsfm", "timeseries_anomaly_detection"), requires=["t1"]),
                ],
                edges=[("t1", "t2")],
            ),
            # ── Cross-domain: IoT → FMSA → WO ─────────────────────────
            "e2e_iot_fmsa_wo": GraphTemplate(
                template_id="e2e_iot_fmsa_wo",
                nodes=[
                    Node("t1", "tool", ("iot",  "get_history")),
                    Node("t2", "tool", ("fmsa", "get_failure_modes"),  requires=["t1"]),
                    Node("t3", "tool", ("wo",   "generate_work_order"), requires=["t2"]),
                ],
                edges=[("t1", "t2"), ("t2", "t3")],
            ),
        }

    # ------------------------------------------------------------------
    # 3. Datasets
    # ------------------------------------------------------------------
    def load_datasets(self):
        rows = []
        try:
            from datasets import load_dataset
            ds = load_dataset("ibm-research/AssetOpsBench", split="train")
            rows = [dict(r) for r in ds]
        except Exception:
            pass

        if not rows:
            import json as _json, os as _os
            local = _os.path.join(_os.path.dirname(__file__), "data", "assetopsbench_scenarios.json")
            try:
                with open(local) as f:
                    rows = _json.load(f)
            except Exception:
                pass

        if not rows:
            # Offline synthetic fallback — mirrors real AssetOpsBench schema
            _templates = [
                ("IoT",       "List all IoT sites"),
                ("FMSA",      "List failure modes of Chiller 6"),
                ("TSFM",      "Forecast Chiller 9 Condenser Water Flow"),
                ("Workorder", "Get work orders for CWC04013 in 2017"),
            ]
            rows = [
                {"id": i, "text": txt, "type": typ,
                 "category": "Knowledge Query", "deterministic": True,
                 "characteristic_form": txt, "note": None}
                for i, (typ, txt) in enumerate(_templates * 40)
            ]

        queries = [
            {
                "id":               r["id"],
                "text":             r["text"],
                "type":             r["type"],
                "category":         r["category"],
                "deterministic":    r["deterministic"],
                "characteristic_form": r["characteristic_form"],
                # template hint consumed by build_init_actions
                "_template":        _TEMPLATE_FOR_TYPE.get(r["type"], "iot_single"),
            }
            for r in rows
        ]

        rng = random.Random(0)
        rng.shuffle(queries)
        n = len(queries)
        return queries[:int(0.6*n)], queries[int(0.6*n):int(0.8*n)], queries[int(0.8*n):]

    # ------------------------------------------------------------------
    # 4. Tool execution
    # ------------------------------------------------------------------
    def call_tool(self, server, tool, args) -> Tuple[bool, Optional[str], Optional[Dict]]:
        try:
            import requests
        except ImportError:
            # requests not available: return stub so offline dev still works
            return True, None, {"stub": True, "server": server, "tool": tool}

        base = self.endpoints.get(server)
        if base is None:
            return False, f"unknown_server:{server}", None

        query = args.get("query", {})
        payload = {
            "task":   query.get("text", ""),
            "tool":   tool,
            "params": {k: v for k, v in args.items() if k != "query"},
        }
        r = requests.post(f"{base}/{tool}", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return True, None, r.json()
        '''except Exception as exc:
            kind = type(exc).__name__
            return False, f"{kind}:{exc}", None'''

    # ------------------------------------------------------------------
    # 5. LLM judge scoring
    # ------------------------------------------------------------------
    def score_result(self, res: ExecResult, wf: CompiledWorkflow) -> Optional[MetricBreakdown]:
        if not res.ok or res.final_output is None:
            return None
        try:
            import requests
            payload = {
                "trajectory": [
                    {"node": l.node_id, "tool": l.tool,
                     "ok": l.ok, "output": l.output}
                    for l in res.logs
                ],
                "final_output": res.final_output,
            }
            r = requests.post(self.judge_url, json=payload, timeout=30.0)
            r.raise_for_status()
            s = r.json()
            return MetricBreakdown(
                success=s.get("correctness", 0.0),
                runtime_ok=1.0,
                schema_adherence=s.get("format_adherence", 0.0),
                dependency_compliance=s.get("step_compliance", 0.0),
                violation_rate=s.get("hallucination_rate", 0.0),
                latency_ms=float(sum(l.latency_ms for l in res.logs)),
            )
        except Exception:
            return None   # fall back to structural proxy

    # ------------------------------------------------------------------
    # 6. Domain-aware initial actions
    # ------------------------------------------------------------------
    def build_init_actions(self, D_fb, templates):
        seen = set(q.get("_template", "iot_single") for q in D_fb)
        actions = []
        for tpl_id in seen:
            if tpl_id not in templates:
                continue
            actions.append(Action(g=tpl_id, e=(), p=PromptParams(max_steps=8)))
            actions.append(Action(g=tpl_id, e=(), p=PromptParams(max_steps=6, strict_schema=False)))
            actions.append(Action(g=tpl_id, e=(), p=PromptParams(max_steps=8, verifier_strict=False)))
        return actions or [Action(g="iot_single", e=(), p=PromptParams(max_steps=8))]