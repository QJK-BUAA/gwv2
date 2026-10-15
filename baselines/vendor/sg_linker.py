"""Steps 2-4 of SchemaGraphSQL: schema linking via graph pathfinding.

This module reproduces the paper's full configuration space (Section 2.4 + the
ablation in Section 6, Tables 4/5/7), which is defined by *two orthogonal axes*:

  1. Source/Destination cardinality -- how many anchor tables the LLM may return:
       * "1-1": exactly one source and one destination (Prompt 2),
       * "1-n": one source, many destinations,
       * "n-1": many sources, one destination,
       * "n-n": many sources and many destinations (Prompt 1).

  2. Path aggregation -- how the candidate shortest paths are combined into the
     final filtered sub-schema:
       * "standard"     : build the union of all shortest paths and let the LLM
                          choose between the individual paths and that union
                          (Prompt 3). This backs the 1-1 / 1-n / n-1 / n-n rows.
       * "no-union"     : the LLM picks the single most appropriate shortest path;
                          no union option is offered.
       * "force-union"  : always take the union of all shortest paths directly
                          (the paper's primary, recall-oriented configuration).
       * "force-longest": deterministically take the longest among the candidate
                          shortest paths, removing the LLM path decision.

The tables on the resulting path(s) form the filtered sub-schema handed to the
SQL generator.

Pipeline stages:
  * Step 2 (identify_src_dst): one LLM call returns source / destination tables.
  * Step 3 (shortest path): Dijkstra/BFS shortest join path(s) on the table graph.
  * Step 4 (aggregation): combine the shortest paths per the chosen strategy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from .llm import LLM
from .prompts import PROMPT_JOIN_PATH, PROMPT_SRC_DST_11, PROMPT_SRC_DST_NN
from .schema import Schema

CARDINALITIES = ("1-1", "1-n", "n-1", "n-n")
AGGREGATIONS = ("standard", "no-union", "force-union", "force-longest")

# Friendly single-name configs -> (cardinality, aggregation). These reproduce the
# seven rows reported in the paper's ablation tables (Tables 4/5/7).
CONFIG_ALIASES: dict[str, tuple[str, str]] = {
    "1-1": ("1-1", "standard"),
    "1-n": ("1-n", "standard"),
    "n-1": ("n-1", "standard"),
    "n-n": ("n-n", "standard"),
    "no-union": ("n-n", "no-union"),
    "force-union": ("n-n", "force-union"),
    "force-longest": ("n-n", "force-longest"),
}


def resolve_config(config: str) -> tuple[str, str]:
    """Map a friendly config name (e.g. "force-union") to (cardinality, aggregation)."""
    if config in CONFIG_ALIASES:
        return CONFIG_ALIASES[config]
    raise ValueError(
        f"Unknown config '{config}'. Known: {sorted(CONFIG_ALIASES)} "
        f"or pass cardinality/aggregation explicitly."
    )


@dataclass
class LinkResult:
    sources: list[str]
    destinations: list[str]
    linked_tables: list[str]
    paths: list[list[str]]
    join_conditions: list[str]
    config: str
    cardinality: str = ""
    aggregation: str = ""
    note: str = ""
    extras: dict = field(default_factory=dict)


_SRC_DST = re.compile(r"src\s*=\s*([^,]*(?:,[^,]*)*?)\s*,\s*dst\s*=\s*(.+)", re.IGNORECASE)


def _match_tables(raw: str, schema: Schema) -> list[str]:
    valid = {t.lower(): t for t in schema.table_names()}
    out: list[str] = []
    for tok in re.split(r"[,\s]+", raw.strip()):
        tok = tok.strip().strip('"`.')
        if not tok:
            continue
        key = tok.split(".")[-1].lower()
        if key in valid and valid[key] not in out:
            out.append(valid[key])
    return out


def identify_src_dst(question: str, schema: Schema, llm: LLM,
                     cardinality: str = "n-n") -> tuple[list[str], list[str]]:
    """One LLM call to extract anchor tables, trimmed per the cardinality axis."""
    system = PROMPT_SRC_DST_11 if cardinality == "1-1" else PROMPT_SRC_DST_NN
    user = f"Schema:\n{schema.compact_text()}\n\nQuestion: {question}"
    resp = llm.complete(system, user)
    m = _SRC_DST.search(resp.replace("\n", " "))
    if not m:
        # Fallback: any table mentioned in the response.
        tabs = _match_tables(resp, schema)
        half = max(1, len(tabs) // 2)
        srcs, dsts = (tabs[:half] or tabs), (tabs[half:] or tabs)
    else:
        srcs = _match_tables(m.group(1), schema)
        dsts = _match_tables(m.group(2), schema)

    # Enforce cardinality: a "1" on either side keeps only the top anchor.
    single_src = cardinality in ("1-1", "1-n")
    single_dst = cardinality in ("1-1", "n-1")
    if single_src and srcs:
        srcs = srcs[:1]
    if single_dst and dsts:
        dsts = dsts[:1]
    # Backfill empty side from the other so pathfinding still has anchors.
    if not srcs:
        srcs = dsts[:1] if single_src else list(dsts)
    if not dsts:
        dsts = srcs[:1] if single_dst else list(srcs)
    return srcs, dsts


def _all_shortest_paths(graph: nx.Graph, s: str, d: str) -> list[list[str]]:
    if s == d:
        return [[s]]
    if s not in graph or d not in graph:
        return []
    try:
        return list(nx.all_shortest_paths(graph, s, d))
    except nx.NetworkXNoPath:
        return []


def _gather_candidate_paths(
    graph: nx.Graph, srcs: list[str], dsts: list[str]
) -> tuple[list[list[str]], set[str]]:
    """All (de-duplicated) shortest paths over every source-destination pair.

    Returns the candidate path list plus the set of anchor tables for pairs that
    were disconnected in the graph (so they can still be kept downstream)."""
    candidates: list[list[str]] = []
    seen: set[tuple] = set()
    anchor_only: set[str] = set()
    for s in srcs:
        for d in dsts:
            sp = _all_shortest_paths(graph, s, d)
            if not sp:
                anchor_only.update([s, d])
                continue
            for p in sp:
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    candidates.append(p)
    return candidates, anchor_only


def _join_conditions_for_path(graph: nx.Graph, path: list[str]) -> list[str]:
    conds: list[str] = []
    for a, b in zip(path, path[1:]):
        data = graph.get_edge_data(a, b) or {}
        for (t1, c1, t2, c2) in data.get("joins", []):
            conds.append(f"{t1}.{c1} = {t2}.{c2}")
    return conds


def _select_path_llm(question: str, paths: list[list[str]], llm: LLM,
                     union_nodes: list[str] | None = None) -> tuple[int, list[str]]:
    """Ask the LLM to pick one join path. When ``union_nodes`` is given it is
    offered as an extra candidate (the "standard" aggregation), letting the model
    choose between the individual shortest paths and their union."""
    listing_lines = [f"path_id: {i} -> {' - '.join(p)}" for i, p in enumerate(paths)]
    union_id = -1
    if union_nodes is not None:
        union_id = len(paths)
        listing_lines.append(f"path_id: {union_id} -> {' - '.join(union_nodes)} (union)")
    user = f"Question: {question}\n\nCandidate join paths:\n" + "\n".join(listing_lines)
    resp = llm.complete(PROMPT_JOIN_PATH, user)
    m = re.search(r"path_id\s*:?\s*(\d+)", resp)
    if m:
        idx = int(m.group(1))
        if union_nodes is not None and idx == union_id:
            return union_id, list(union_nodes)
        if 0 <= idx < len(paths):
            return idx, paths[idx]
    return 0, paths[0]


def link(question: str, schema: Schema, graph: nx.Graph, llm: LLM,
         config: str | None = None,
         cardinality: str | None = None,
         aggregation: str | None = None) -> LinkResult:
    """Run schema linking.

    Either pass a friendly ``config`` name (e.g. "force-union", "1-1", "n-n",
    "no-union", "force-longest") or set ``cardinality`` and ``aggregation``
    explicitly. Explicit axes take precedence over ``config``.
    """
    if cardinality is None or aggregation is None:
        card, agg = resolve_config(config or "force-union")
        cardinality = cardinality or card
        aggregation = aggregation or agg
    if cardinality not in CARDINALITIES:
        raise ValueError(f"cardinality must be one of {CARDINALITIES}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
    label = config if config else f"{cardinality}_{aggregation}"

    srcs, dsts = identify_src_dst(question, schema, llm, cardinality=cardinality)
    if not srcs and not dsts:
        return LinkResult([], [], schema.table_names(), [], [], label,
                          cardinality=cardinality, aggregation=aggregation,
                          note="LLM returned no tables; falling back to full schema")

    candidates, anchor_only = _gather_candidate_paths(graph, srcs, dsts)

    chosen_paths: list[list[str]] = []
    linked_nodes: set[str] = set()
    note = ""

    if not candidates:
        linked_nodes = set(srcs) | set(dsts)
        note = "no path between any source and destination; using anchors only"
    elif aggregation == "force-union":
        chosen_paths = candidates
        for p in candidates:
            linked_nodes.update(p)
    elif aggregation == "force-longest":
        best = max(candidates, key=len)
        chosen_paths = [best]
        linked_nodes.update(best)
        if len(candidates) > 1:
            note = f"force-longest: longest of {len(candidates)} shortest paths"
    elif aggregation == "no-union":
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            _, chosen = _select_path_llm(question, candidates, llm)
            note = f"no-union: LLM chose 1 of {len(candidates)} paths"
        chosen_paths = [chosen]
        linked_nodes.update(chosen)
    else:  # "standard": LLM chooses between individual paths and their union
        union_nodes = sorted({n for p in candidates for n in p})
        if len(candidates) == 1:
            chosen_paths = [candidates[0]]
            linked_nodes.update(candidates[0])
        else:
            idx, chosen = _select_path_llm(question, candidates, llm,
                                           union_nodes=union_nodes)
            if idx == len(candidates):  # union picked
                chosen_paths = candidates
                linked_nodes.update(union_nodes)
                note = f"standard: LLM picked the union of {len(candidates)} paths"
            else:
                chosen_paths = [chosen]
                linked_nodes.update(chosen)
                note = f"standard: LLM picked 1 of {len(candidates)} paths"

    # Always keep disconnected anchors so required tables are never dropped.
    linked_nodes.update(anchor_only)
    linked = sorted(linked_nodes) or sorted(set(srcs) | set(dsts))

    conds: list[str] = []
    seen: set[str] = set()
    for p in chosen_paths:
        for c in _join_conditions_for_path(graph, p):
            if c not in seen:
                seen.add(c)
                conds.append(c)

    return LinkResult(
        sources=srcs,
        destinations=dsts,
        linked_tables=linked,
        paths=chosen_paths,
        join_conditions=conds,
        config=label,
        cardinality=cardinality,
        aggregation=aggregation,
        note=note,
    )
