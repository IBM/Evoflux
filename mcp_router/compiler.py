from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import copy


from .dsl import Action, GraphTemplate, Node, ToolRegistry, TypedEdit, PromptParams


@dataclass(frozen=True)
class CompiledWorkflow:
    action_id: str
    nodes: List[Node]           # topological order
    edges: List[Tuple[str, str]]
    policy: PromptParams
    query: Dict[str, Any]       # the query this workflow was compiled for
    graph: GraphTemplate


class Compiler:
    """
    Query-aware compiler.

    For each Action the compiler:
      1. Calls the LLM planner to build a base GraphTemplate grounded in
         the action's specific query (q).  The planner looks at the query
         text and the tool catalog, then returns the full set of tool calls
         needed to complete that task in dependency order.
      2. Applies the TypedEdits from the action on top of that base graph.
      3. Topologically sorts nodes, canonicalises edges, and returns a
         CompiledWorkflow the Executor can walk directly.

    The base plan is cached by query hash so the same query never calls
    the LLM twice across multiple evolutionary generations.
    """

    def __init__(self, registry: ToolRegistry):
        """
        Parameters
        ----------
        registry : ToolRegistry
            Used to validate every (server, tool) pair — both in plans
            returned by the LLM and in swap_tool / add_tool_step edits.
        """
        self.registry = registry
        

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compile(self, a: Action) -> CompiledWorkflow:
        """
        Compile an Action into an executable CompiledWorkflow.

        Step 1 — get base plan for this query (cached after first call)
        Step 2 — apply structural edits
        Step 3 — topological sort, canonicalise, return
        """
        base = a.graph 

        nodes = list(copy.deepcopy(base.nodes))
        edges = list(copy.deepcopy(base.edges))

        for edit in self._dedupe_edits(list(a.e)):
            nodes, edges = self._apply_edit(nodes, edges, edit)

        nodes, edges = self._topo_sort(nodes, edges)
        nodes, edges = self._inject_output_node(nodes, edges)
        return CompiledWorkflow(
            action_id = a.stable_id(),
            nodes     = nodes,
            edges     = edges,
            policy    = a.p,
            query     = a.q,
            graph     = base
        )

    
    # ------------------------------------------------------------------
    # Edit application
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe_edits(edits: List[TypedEdit]) -> List[TypedEdit]:
        """
        Prevents redundant edits accumulating across mutation generations.
        Processes in reverse; keeps the last (most recent) edit per slot.
        """
        seen: set = set()
        out:  List[TypedEdit] = []
        for e in reversed(edits):
            args = e.args
            if e.op == "swap_tool":
                dk = (e.op, args["node_id"], None)
            elif e.op == "insert_validator":
                dk = (e.op, args["after_node_id"], None)
            elif e.op == "set_param":
                dk = (e.op, args["node_id"], args["key"])
            elif e.op == "add_tool_step":
                dk = (e.op, args["node_id"], None)
            elif e.op == "remove_tool_step":
                dk = (e.op, args["node_id"], None)
            elif e.op == "reorder_step":
                dk = (e.op, args["node_id"], None)
            else:
                dk = (e.op, id(e), None)
            if dk not in seen:
                seen.add(dk)
                out.append(e)
        return list(reversed(out))

    def _apply_edit(
        self,
        nodes: List[Node],
        edges: List[Tuple[str, str]],
        edit:  TypedEdit,
    ) -> Tuple[List[Node], List[Tuple[str, str]]]:
        op   = edit.op
        args = edit.args

        # ── swap_tool ─────────────────────────────────────────────────
        if op == "swap_tool":
            ns, nt = args["new_server"], args["new_tool"]
            if not self.registry.exists(ns, nt):
                raise ValueError(f"swap_tool: ({ns!r}, {nt!r}) not in registry")
            out = []
            for n in nodes:
                if n.node_id == args["node_id"]:
                    if n.kind != "tool":
                        raise ValueError("swap_tool only allowed on tool nodes")
                    out.append(Node(node_id=n.node_id, kind=n.kind,
                                    tool_ref=(ns, nt),
                                    params=dict(n.params),
                                    requires=list(n.requires),
                                    produces=dict(n.produces)))
                else:
                    out.append(n)
            return out, edges

        # ── insert_validator ──────────────────────────────────────────
        if op == "insert_validator":
            after = args["after_node_id"]
            vid   = args["validator_id"]
            if any(n.node_id == vid for n in nodes):
                raise ValueError(f"insert_validator: id {vid!r} already exists")
            v         = Node(node_id=vid, kind="validator",
                             requires=[after], params={"type": "schema"})
            outgoing  = [dst for (src, dst) in edges if src == after]
            new_edges = [(s, d) for (s, d) in edges
                         if not (s == after and d in outgoing)]
            new_edges.append((after, vid))
            for dst in outgoing:
                new_edges.append((vid, dst))
            return nodes + [v], new_edges

        # ── set_param ─────────────────────────────────────────────────
        if op == "set_param":
            out = []
            for n in nodes:
                if n.node_id == args["node_id"]:
                    out.append(Node(node_id=n.node_id, kind=n.kind,
                                    tool_ref=n.tool_ref,
                                    params={**dict(n.params),
                                            args["key"]: args["value"]},
                                    requires=list(n.requires),
                                    produces=dict(n.produces)))
                else:
                    out.append(n)
            return out, edges

        # ── add_tool_step ─────────────────────────────────────────────
        if op == "add_tool_step":
            nid = args["node_id"]
            ns, nt = args["server"], args["tool"]
            if any(n.node_id == nid for n in nodes):
                raise ValueError(f"add_tool_step: node_id {nid!r} already exists")
            if not self.registry.exists(ns, nt):
                raise ValueError(f"add_tool_step: ({ns!r}, {nt!r}) not in registry")
            after     = args.get("after_node_id")
            requires  = [after] if after else []
            new_node  = Node(node_id=nid, kind="tool",
                             tool_ref=(ns, nt), requires=requires)
            new_edges = list(edges)
            if after:
                new_edges.append((after, nid))
            return nodes + [new_node], new_edges

        # ── remove_tool_step ──────────────────────────────────────────
        if op == "remove_tool_step":
            nid     = args["node_id"]
            removed = next((n for n in nodes if n.node_id == nid), None)
            if removed is None:
                return nodes, edges   # idempotent
            parents   = list(removed.requires)
            new_nodes = [n for n in nodes if n.node_id != nid]
            rewired   = []
            for n in new_nodes:
                if nid in n.requires:
                    new_req = [r for r in n.requires if r != nid] + parents
                    rewired.append(Node(node_id=n.node_id, kind=n.kind,
                                        tool_ref=n.tool_ref, params=dict(n.params),
                                        requires=new_req, produces=dict(n.produces)))
                else:
                    rewired.append(n)
            new_edges = [(s, d) for (s, d) in edges
                         if s != nid and d != nid]
            for n in rewired:
                for p in n.requires:
                    if (p, n.node_id) not in new_edges:
                        new_edges.append((p, n.node_id))
            return rewired, new_edges

        # ── reorder_step ──────────────────────────────────────────────
        if op == "reorder_step":
            nid      = args["node_id"]
            new_reqs = list(args["new_requires"])
            existing = {n.node_id for n in nodes}
            for r in new_reqs:
                if r not in existing:
                    raise ValueError(f"reorder_step: unknown requires node {r!r}")
            out = []
            for n in nodes:
                if n.node_id == nid:
                    out.append(Node(node_id=n.node_id, kind=n.kind,
                                    tool_ref=n.tool_ref, params=dict(n.params),
                                    requires=new_reqs, produces=dict(n.produces)))
                else:
                    out.append(n)
            new_edges = []
            for n in out:
                for r in n.requires:
                    new_edges.append((r, n.node_id))
            return out, sorted(set(new_edges))

        raise ValueError(f"Unknown edit op: {op!r}")

    # ------------------------------------------------------------------
    # Output node injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_output_node(
        nodes: List[Node],
        edges: List[Tuple[str, str]],
    ) -> Tuple[List[Node], List[Tuple[str, str]]]:
        """
        Ensure the graph ends with a single "output" aggregator node.

        Finds all leaf nodes — nodes that no other node depends on — and if
        there is more than one, injects a synthetic Node(kind="output") that
        requires all of them.  The executor merges each leaf's memory entry
        into a single final_output dict keyed by node_id.

        If there is already exactly one leaf, or if the last node is already
        kind="output", the graph is returned unchanged.
        """
        if nodes and nodes[-1].kind == "output":
            return nodes, edges

        if not nodes:
            return nodes, edges

        depended_on = {r for n in nodes for r in (n.requires or [])}
        leaves = [n.node_id for n in nodes if n.node_id not in depended_on]

        out_node = Node(
            node_id="output",
            kind="output",
            requires=leaves,
        )
        new_edges = list(edges) + [(leaf, "output") for leaf in leaves]
        return nodes + [out_node], new_edges

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    @staticmethod
    def _topo_sort(
        nodes: List[Node],
        edges: List[Tuple[str, str]],
    ) -> Tuple[List[Node], List[Tuple[str, str]]]:
        """
        Kahn's algorithm.  Returns nodes in a valid execution order (all
        dependencies before dependants) and edges canonicalised.

        Raises ValueError if the graph has a cycle.
        """
        node_map  = {n.node_id: n for n in nodes}
        in_degree: Dict[str, int] = {n.node_id: 0 for n in nodes}
        children:  Dict[str, List[str]] = {n.node_id: [] for n in nodes}

        for src, dst in edges:
            if src in in_degree and dst in in_degree:
                in_degree[dst] += 1
                children[src].append(dst)

        queue  = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        result: List[Node] = []

        while queue:
            nid = queue.pop(0)
            result.append(node_map[nid])
            for child in sorted(children[nid]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(nodes):
            raise ValueError("Workflow graph contains a cycle — cannot compile.")

        return result, sorted(set(edges))