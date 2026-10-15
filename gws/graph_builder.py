"""Candidate relation graph + bounded best-first path search (Ground stage).

This module is fully deterministic and LLM-free: given a ``Schema`` and a set of
model-proposed relations, it reproduces the paper's candidate construction and
path search exactly (Sec. Ground / Appendix "Candidate graph and paths").

Candidate edges come from three sources:
  * structural enumeration (key-like source name -> key-like target column),
  * visible (declared) foreign keys,
  * model-proposed relations.
Structural and visible-FK edges receive a path-ranking prior of 0.65;
model-proposed edges receive 0.75. Duplicate unordered pairs keep the larger
score and the union of source labels.

Path search runs bounded best-first (min-heap) over simple paths with edge cost
``1 + (1 - prior)`` and the paper's caps.
"""
from __future__ import annotations

import heapq
import re
from .sql_analysis import legal_joins
from difflib import SequenceMatcher

from . import config
from .schema import Schema
from .types import Path, RelationCandidate

_KEY_SUFFIXES = ("_id", "_key", "_code")
_KEY_NAMES = ("id", "code")


def _stem(name: str) -> str:
    """Strip a trailing key suffix / 'id' / 'code' to expose the referent stem."""
    return re.sub(r"(_id|id|_key|_code|code)$", "", name.lower())


def _singularize(table: str) -> str:
    low = re.sub(r"[^a-z0-9]+", "_", table.lower()).strip("_")
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith("s") and len(low) > 3:
        return low[:-1]
    return low


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_key_like_source(column_name: str) -> bool:
    low = column_name.lower()
    return low in _KEY_NAMES or low.endswith(_KEY_SUFFIXES) or low.endswith("id")


def _type_compatible(t1: str, t2: str) -> bool:
    """Coarse SQLite type-affinity compatibility used for structural targets."""
    return norm_type(t1) == norm_type(t2)


def norm_type(t):
    t = (t or "").upper()
    if any(k in t for k in ("INT", "NUMBER", "REAL", "FLOA", "DOUB", "DEC", "NUMERIC")):
        return "NUMERIC"
    if any(k in t for k in ("CHAR", "TEXT", "CLOB", "STRING")):
        return "TEXT"
    if "DATE" in t or "TIME" in t:
        return "DATETIME"
    return t or "ANY"


def name_score(src_column: str, dst_table: str, dst_column: str) -> float:
    """Structural candidate eligibility score (Appendix "Candidate graph").

    max of: exact source-stem vs target-table match (singularized), their
    string-similarity ratio, and 0.8 * source/target column-name similarity.
    """
    stem = _stem(src_column)
    tgt_tbl = _singularize(dst_table)
    exact = 1.0 if stem and stem == tgt_tbl else 0.0
    tbl_sim = _ratio(stem, tgt_tbl)
    col_sim = 0.8 * _ratio(src_column.lower(), dst_column.lower())
    return max(exact, tbl_sim, col_sim)


def _enumerate_structural(schema: Schema) -> list[RelationCandidate]:
    """Key-like source column -> key-like target column in a different table.

    Target eligibility: primary key, named ``id``, or recorded uniqueness >=
    0.95 (uniqueness is only known if column stats were collected; the paper's
    loader does not precompute stats, so eligibility leans on PK/name).
    """
    cands: list[RelationCandidate] = []
    tables = schema.table_names()
    for src_t in tables:
        for src_c in schema.tables.get(src_t, []):
            if not _is_key_like_source(src_c.name):
                continue
            for dst_t in tables:
                if dst_t == src_t:
                    continue
                for dst_c in schema.tables.get(dst_t, []):
                    eligible = (
                        dst_c.pk
                        or dst_c.name.lower() == "id"
                        or (dst_c.uniqueness or 0.0) >= config.STRUCTURAL_UNIQUENESS
                    )
                    if not eligible:
                        continue
                    if not _type_compatible(src_c.type, dst_c.type):
                        continue
                    score = name_score(src_c.name, dst_t, dst_c.name)
                    if score < config.NAME_SCORE_MIN:
                        continue
                    cands.append(RelationCandidate(
                        src_table=src_t, src_column=src_c.name,
                        dst_table=dst_t, dst_column=dst_c.name,
                        score=config.SCORE_STRUCTURAL, sources=("structural",),
                    ))
    return cands


def _visible_fks(schema: Schema) -> list[RelationCandidate]:
    out: list[RelationCandidate] = []
    for fk in schema.declared_fks:
        out.append(RelationCandidate(
            src_table=fk.src_table, src_column=fk.src_column,
            dst_table=fk.ref_table, dst_column=fk.ref_column,
            score=config.SCORE_VISIBLE_FK, sources=("visible_fk",),
        ))
    return out


def _coerce_model_relations(schema: Schema,
                            proposed: list[dict]) -> list[RelationCandidate]:
    """Validate model-proposed relations against the actual catalog."""
    out: list[RelationCandidate] = []
    for rel in legal_joins(schema, proposed):
        st, sc = rel["left"].split(".", 1)
        dt, dc = rel["right"].split(".", 1)
        out.append(RelationCandidate(
            src_table=st, src_column=sc, dst_table=dt, dst_column=dc,
            score=config.SCORE_MODEL_PROPOSED, sources=("model",),
        ))
    return out


def build_candidate_edges(schema: Schema,
                          model_relations: list[dict] | None = None,
                          include_structural: bool = True,
                          include_model: bool = True) -> list[RelationCandidate]:
    """Merge the three candidate sources and deduplicate unordered pairs.

    Opposite directions of the same column pair are deduped keeping the larger
    score; the union of source labels is retained. Different column-level
    predicates between the same *table* pair are preserved as separate edges.
    """
    raw: list[RelationCandidate] = []
    raw.extend(_visible_fks(schema))
    if include_structural:
        raw.extend(_enumerate_structural(schema))
    if include_model:
        raw.extend(_coerce_model_relations(schema, model_relations or []))

    merged: dict[str, RelationCandidate] = {}
    for cand in raw:
        norm = cand.normalized()
        key = norm.edge_id
        if key not in merged:
            merged[key] = RelationCandidate(
                src_table=norm.src_table, src_column=norm.src_column,
                dst_table=norm.dst_table, dst_column=norm.dst_column,
                score=norm.score, sources=tuple(sorted(set(norm.sources))),
            )
        else:
            prev = merged[key]
            prev.score = max(prev.score, norm.score)
            prev.sources = tuple(sorted(set(prev.sources) | set(norm.sources)))
    return list(merged.values())


def _adjacency(edges: list[RelationCandidate]) -> dict[str, list[RelationCandidate]]:
    """Undirected table-level adjacency; each physical edge appears both ways."""
    adj: dict[str, list[RelationCandidate]] = {}
    for e in edges:
        if e.src_table == e.dst_table:
            continue
        adj.setdefault(e.src_table, []).append(e)
        adj.setdefault(e.dst_table, []).append(e)
    for values in adj.values():
        values.sort(key=lambda e: (-e.score, e.edge_id))
    return adj


def _edge_cost(edge: RelationCandidate) -> float:
    # cost(u,v) = 1 + (1 - prior(u,v)); Equation (edge cost) in Sec. Ground.
    return 1.0 + (1.0 - edge.score)


def _search_pair(adj: dict[str, list[RelationCandidate]],
                 start: str, goal: str) -> list[Path]:
    """Bounded best-first search for simple paths between two tables."""
    if start == goal or start not in adj or goal not in adj:
        return []
    # Heap entries: (cost, seq, current_table, tables_tuple, edges_tuple).
    heap: list[tuple] = [(0.0, 0, start, (start,), ())]
    seq = 1
    removals = 0
    found: list[Path] = []
    while heap and removals < config.MAX_HEAP_REMOVALS:
        cost, _, node, tables, path_edges = heapq.heappop(heap)
        removals += 1
        if node == goal and len(path_edges) >= 1:
            found.append(Path(tables=list(tables), edges=list(path_edges), cost=cost))
            if len(found) >= config.MAX_PATHS_PER_PAIR:
                break
            continue
        if len(path_edges) >= config.MAX_PATH_EDGES:
            continue
        for edge in adj.get(node, []):
            nxt = edge.dst_table if edge.src_table == node else edge.src_table
            if nxt in tables:            # table-simple: no repeated tables
                continue
            heapq.heappush(heap, (
                cost + _edge_cost(edge), seq, nxt,
                tables + (nxt,), path_edges + (edge,),
            ))
            seq += 1
    return found


def search_paths(edges: list[RelationCandidate],
                 terminals: list[str]) -> list[Path]:
    """Enumerate bounded paths between all terminal table pairs.

    Unordered endpoint pairs are visited in lexicographic order; the overall cap
    (12 paths) accumulates in that order (Sec. Ground / Appendix).
    """
    adj = _adjacency(edges)
    uniq = sorted({t for t in terminals})
    pairs = [(a, b) for i, a in enumerate(uniq) for b in uniq[i + 1:]]
    all_paths: list[Path] = []
    for a, b in pairs:
        for p in _search_pair(adj, a, b):
            all_paths.append(p)
            if len(all_paths) >= config.MAX_PATHS_OVERALL:
                return all_paths
    return all_paths
