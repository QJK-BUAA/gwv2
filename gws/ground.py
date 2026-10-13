"""Ground stage: requirements, independent draft, candidate relations, paths.

Two independent model calls (requirements and draft) receive the common context
but not each other's output. The draft is parsed with SQLGlot to resolve
physical source columns; parse failures are tolerated (they accompany the draft
rather than triggering another call). Candidate edges and bounded paths are then
built deterministically by ``graph_builder``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import graph_builder, prompts
from .parsing import extract_json, extract_sql
from .schema import Schema
from .types import Path, RelationCandidate, Requirement


@dataclass
class GroundResult:
    requirements: list[Requirement]
    model_relations: list[dict]
    draft_sql: str
    draft_tables: list[str]
    draft_columns: list[str]
    candidates: list[RelationCandidate]
    paths: list[Path]
    raw_requirements: dict[str, Any] = field(default_factory=dict)


def _common_context(schema: Schema, question: str, evidence: str,
                    tables: list[str] | None = None) -> str:
    ddl = schema.to_ddl_text(tables=tables, annotate=True)
    parts = [f"-- Database: {schema.name}", ddl, "", f"Question: {question}"]
    if evidence:
        parts.append(f"External evidence: {evidence}")
    return "\n".join(parts)


def _parse_requirements(obj: dict | None, schema: Schema) -> list[Requirement]:
    out: list[Requirement] = []
    if not obj:
        return out
    for i, rec in enumerate(obj.get("requirements", []) or []):
        if not isinstance(rec, dict):
            continue
        cols = [c for c in (rec.get("columns") or []) if _col_ok(schema, c)]
        alts = [c for c in (rec.get("alternatives") or []) if _col_ok(schema, c)]
        out.append(Requirement(
            id=str(rec.get("id") or f"r{i+1}"),
            kind=str(rec.get("kind") or "output"),
            need=str(rec.get("need") or ""),
            columns=cols, alternatives=alts,
            reason=str(rec.get("reason") or ""),
        ))
    return out


def _col_ok(schema: Schema, dotted: str) -> bool:
    if not isinstance(dotted, str) or "." not in dotted:
        return False
    t, c = dotted.split(".", 1)
    return schema.column(t, c) is not None


def parse_draft_objects(sql: str, schema: Schema) -> tuple[list[str], list[str]]:
    """Resolve physical tables/columns referenced by the draft via SQLGlot.

    Best-effort: aliases and subquery scopes are resolved to source tables where
    possible; on any parse error we fall back to a catalog-filtered token scan so
    the draft still contributes terminals to the path search.
    """
    tables: list[str] = []
    columns: list[str] = []
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="sqlite")
        alias_to_table: dict[str, str] = {}
        for tnode in tree.find_all(exp.Table):
            name = tnode.name
            if name and _table_ok(schema, name) and name not in tables:
                tables.append(name)
            alias = tnode.alias_or_name
            if alias and name:
                alias_to_table[alias] = name
        for cnode in tree.find_all(exp.Column):
            tbl = alias_to_table.get(cnode.table, cnode.table) or ""
            col = cnode.name
            if tbl and schema.column(tbl, col) is not None:
                dotted = f"{tbl}.{col}"
                if dotted not in columns:
                    columns.append(dotted)
            elif not tbl and col:
                # unqualified column: attach to any table that owns it
                for t in tables:
                    if schema.column(t, col) is not None:
                        dotted = f"{t}.{col}"
                        if dotted not in columns:
                            columns.append(dotted)
                        break
    except Exception:  # noqa: BLE001 -- parse diagnostics accompany the draft
        tables, columns = _token_scan(sql, schema)
    return tables, columns


def _table_ok(schema: Schema, name: str) -> bool:
    return name in schema.tables


def _token_scan(sql: str, schema: Schema) -> tuple[list[str], list[str]]:
    low = sql.lower()
    tables = [t for t in schema.table_names() if t.lower() in low]
    columns: list[str] = []
    for t in tables:
        for c in schema.tables.get(t, []):
            if c.name.lower() in low:
                columns.append(f"{t}.{c.name}")
    return tables, columns


def ground(llm, schema: Schema, question: str, evidence: str,
           include_structural: bool = True, include_model: bool = True,
           include_draft: bool = True) -> GroundResult:
    """Run the two Ground model calls and build the candidate graph + paths.

    ``include_*`` flags support the relation-construction ablations (Table:
    relation-construction ablation): drop the independent draft, structural
    candidates, or model-proposed candidates.
    """
    ctx = _common_context(schema, question, evidence)

    # --- call 1: requirements ---------------------------------------------- #
    req_text = llm.complete(prompts.REQUIREMENTS_SYS, ctx)
    req_obj = extract_json(req_text, ("requirements",))
    requirements = _parse_requirements(req_obj, schema)
    model_relations = (req_obj or {}).get("relations", []) or []

    # --- call 2: independent draft (no requirement response in context) ---- #
    draft_sql, draft_tables, draft_columns = "", [], []
    if include_draft:
        draft_text = llm.complete(prompts.DRAFT_SYS, ctx)
        draft_sql = extract_sql(draft_text)
        draft_tables, draft_columns = parse_draft_objects(draft_sql, schema)

    # --- candidate graph + bounded paths (deterministic) ------------------- #
    candidates = graph_builder.build_candidate_edges(
        schema, model_relations=model_relations,
        include_structural=include_structural,
        include_model=include_model,
    )
    terminals = _terminal_tables(requirements, draft_tables, schema)
    paths = graph_builder.search_paths(candidates, terminals)

    return GroundResult(
        requirements=requirements, model_relations=model_relations,
        draft_sql=draft_sql, draft_tables=draft_tables, draft_columns=draft_columns,
        candidates=candidates, paths=paths, raw_requirements=req_obj or {},
    )


def _terminal_tables(requirements: list[Requirement], draft_tables: list[str],
                     schema: Schema) -> list[str]:
    """Requirement + draft tables form the terminals for path search."""
    terms: list[str] = list(draft_tables)
    for r in requirements:
        for col in r.columns + r.alternatives:
            if "." in col:
                t = col.split(".", 1)[0]
                if t in schema.tables and t not in terms:
                    terms.append(t)
    return terms
