"""Commit stage: reconcile, audit coverage, render DDL, generate final SQL.

Two model calls (reconciliation then coverage audit) return the same JSON shape
(columns, joins, coverage, unresolved, reason). Deterministic post-processing
then enforces the coverage contract (Sec. Commit / Appendix "coverage contract
and final generation"):

  * filter identifiers and accepted join endpoints;
  * covered/replaced add mapped columns; replaced/not_needed require a reason;
  * missing/invalid coverage restores the requirement's original columns and
    records an unresolved item;
  * unresolved candidates are retained only with retain=true, together with
    accepted join endpoints; an empty final selection is rejected.

The DDL renderer keeps only selected columns and non-empty tables; a foreign key
is rendered only when the relation is accepted and both endpoints are selected;
inferred joins appear as comments. The final generation call emits one SQLite
query (K=1), cleaned and validated as a single SELECT/CTE.
"""
from __future__ import annotations

from typing import Any

from . import prompts
from .ground import GroundResult
from .parsing import extract_json, extract_sql
from .schema import Schema
from .types import (CommitContract, CoverageItem, Decision, RelationCandidate,
                    Requirement)


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
def _evidence_block(schema: Schema, question: str, evidence: str,
                    ground: GroundResult, decisions: list[Decision],
                    posteriors: dict[str, dict[str, float]],
                    maps: dict[str, str], observations: list[dict],
                    include_beliefs: bool = True) -> str:
    lines = [schema.to_ddl_text(annotate=True), "", f"Question: {question}"]
    if evidence:
        lines.append(f"External evidence: {evidence}")
    lines.append("\nRequirements:")
    for r in ground.requirements:
        lines.append(f"  [{r.id}] {r.kind}: {r.need} -> {r.columns} "
                     f"(alts {r.alternatives})")
    lines.append(f"\nIndependent draft SQL:\n{ground.draft_sql or '(none)'}")
    if ground.paths:
        lines.append("\nCandidate join paths:")
        for p in ground.paths:
            lines.append(f"  {p.path_id}")
    if observations:
        lines.append("\nProbe observations:")
        for o in observations:
            lines.append(f"  {o['probe_id']} ({o['kind']}): {o['status']}")
    if include_beliefs and decisions:
        lines.append("\nPosterior MAP recommendations:")
        for d in decisions:
            rec = maps.get(d.id, "")
            label = next((i.label for i in d.interpretations if i.key == rec), rec)
            post = posteriors.get(d.id, {})
            conf = round(post.get(rec, 0.0), 3) if post else 0.0
            lines.append(f"  {d.id}: {label} (conf {conf})")
    return "\n".join(lines)


def _raw_contract(obj: dict | None) -> dict[str, Any]:
    obj = obj or {}
    return {
        "columns": obj.get("columns") or [],
        "joins": obj.get("joins") or [],
        "coverage": obj.get("coverage") or [],
        "unresolved": obj.get("unresolved") or [],
        "reason": obj.get("reason") or "",
    }


def commit(llm, schema: Schema, question: str, evidence: str,
           ground: GroundResult, decisions: list[Decision],
           posteriors: dict[str, dict[str, float]], maps: dict[str, str],
           observations: list[dict], include_beliefs: bool = True,
           coverage_contract: bool = True) -> CommitContract:
    """Run reconcile + audit and enforce the coverage contract.

    ``include_beliefs`` toggles the downstream-belief control; ``coverage_contract``
    toggles the ordinary two-round filter baseline (Commit ablation).
    """
    ev = _evidence_block(schema, question, evidence, ground, decisions,
                         posteriors, maps, observations, include_beliefs)

    # --- call 4: reconciliation ------------------------------------------- #
    rec_text = llm.complete(prompts.RECONCILE_SYS, ev)
    rec = _raw_contract(extract_json(rec_text, ("columns",)))

    # --- call 5: coverage audit ------------------------------------------- #
    audit_input = ev + "\n\nProposed contract:\n" + _contract_summary(rec)
    aud_text = llm.complete(prompts.AUDIT_SYS, audit_input)
    aud = _raw_contract(extract_json(aud_text, ("columns",)))
    # The audit refines the reconciliation; prefer audited fields when present.
    merged = aud if aud["columns"] else rec

    if coverage_contract:
        return _apply_coverage_contract(merged, schema, ground.requirements)
    return _ordinary_filter(merged, schema)


def _contract_summary(c: dict) -> str:
    return (f"columns={c['columns']}\njoins={c['joins']}\n"
            f"coverage={c['coverage']}\nunresolved={c['unresolved']}")


# --------------------------------------------------------------------------- #
# Deterministic coverage contract
# --------------------------------------------------------------------------- #
def _col_ok(schema: Schema, dotted: Any) -> bool:
    if not isinstance(dotted, str) or "." not in dotted:
        return False
    t, c = dotted.split(".", 1)
    return schema.column(t, c) is not None


def _legal_columns(schema: Schema, cols: list) -> list[str]:
    out: list[str] = []
    for c in cols or []:
        if _col_ok(schema, c) and c not in out:
            out.append(c)
    return out


def _accepted_joins(schema: Schema, joins: list) -> list[RelationCandidate]:
    out: list[RelationCandidate] = []
    for j in joins or []:
        if not isinstance(j, dict):
            continue
        try:
            st, sc = str(j["src_table"]), str(j["src_column"])
            dt, dc = str(j["dst_table"]), str(j["dst_column"])
        except (KeyError, TypeError):
            continue
        if schema.column(st, sc) is None or schema.column(dt, dc) is None:
            continue
        out.append(RelationCandidate(st, sc, dt, dc, score=1.0, sources=("commit",)))
    return out


def _apply_coverage_contract(raw: dict, schema: Schema,
                             requirements: list[Requirement]) -> CommitContract:
    columns = _legal_columns(schema, raw["columns"])
    joins = _accepted_joins(schema, raw["joins"])
    coverage: list[CoverageItem] = []
    unresolved: list[str] = []

    req_by_id = {r.id: r for r in requirements}
    seen_cov = set()
    for item in raw["coverage"]:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("requirement_id") or "")
        state = str(item.get("state") or "").lower()
        reason = str(item.get("reason") or "")
        mapped = _legal_columns(schema, item.get("columns") or [])
        seen_cov.add(rid)
        req = req_by_id.get(rid)
        if state == "not_needed" and reason:
            coverage.append(CoverageItem(rid, "not_needed", [], reason))
        elif state == "replaced" and reason and (mapped or (req and not req.columns)):
            coverage.append(CoverageItem(rid, "replaced", mapped, reason))
            for c in mapped:
                if c not in columns:
                    columns.append(c)
        elif state == "covered" and mapped:
            coverage.append(CoverageItem(rid, "covered", mapped, reason))
            for c in mapped:
                if c not in columns:
                    columns.append(c)
        else:
            # missing/invalid coverage -> restore original requirement columns
            if req:
                restored = _legal_columns(schema, req.columns)
                for c in restored:
                    if c not in columns:
                        columns.append(c)
                unresolved.append(f"{rid}:restored")

    # requirements with no coverage record at all: restore their columns.
    for r in requirements:
        if r.id not in seen_cov:
            for c in _legal_columns(schema, r.columns):
                if c not in columns:
                    columns.append(c)
            coverage.append(CoverageItem(r.id, "covered", _legal_columns(schema, r.columns), "default"))

    # retained unresolved candidates
    for u in raw["unresolved"]:
        if isinstance(u, dict) and bool(u.get("retain")):
            label = str(u.get("label") or "")
            if _col_ok(schema, label):
                if label not in columns:
                    columns.append(label)
                unresolved.append(label)

    # accepted join endpoints must be present
    for j in joins:
        for dotted in (f"{j.src_table}.{j.src_column}", f"{j.dst_table}.{j.dst_column}"):
            if dotted not in columns:
                columns.append(dotted)

    if not columns:
        raise ValueError("empty final selection rejected by coverage contract")

    return CommitContract(columns=columns, joins=joins, coverage=coverage,
                          unresolved=unresolved, reason=raw["reason"])


def _ordinary_filter(raw: dict, schema: Schema) -> CommitContract:
    """Ordinary two-round filter baseline (no coverage-contract mechanisms)."""
    columns = _legal_columns(schema, raw["columns"])
    joins = _accepted_joins(schema, raw["joins"])
    for j in joins:
        for dotted in (f"{j.src_table}.{j.src_column}", f"{j.dst_table}.{j.dst_column}"):
            if dotted not in columns:
                columns.append(dotted)
    if not columns:
        raise ValueError("empty final selection rejected")
    return CommitContract(columns=columns, joins=joins, coverage=[],
                          unresolved=[], reason=raw["reason"])


# --------------------------------------------------------------------------- #
# DDL rendering + final generation
# --------------------------------------------------------------------------- #
def render_ddl(schema: Schema, contract: CommitContract) -> str:
    """Render DDL for selected columns only; FK + inferred-join comments."""
    selected: dict[str, list[str]] = {}
    for dotted in contract.columns:
        t, c = dotted.split(".", 1)
        selected.setdefault(t, [])
        if c not in selected[t]:
            selected[t].append(c)

    accepted_pairs = {j.edge_id for j in contract.joins}
    chunks = []
    for t, cols in selected.items():
        defs = []
        for c in cols:
            col = schema.column(t, c)
            typ = col.type if col else "TEXT"
            defs.append(f'  "{c}" {typ}')
        # declared FK rendered only when accepted and both endpoints selected
        for fk in schema.declared_fks:
            if fk.src_table != t:
                continue
            edge = RelationCandidate(fk.src_table, fk.src_column,
                                     fk.ref_table, fk.ref_column, 1.0).edge_id
            src_dotted = f"{fk.src_table}.{fk.src_column}"
            dst_dotted = f"{fk.ref_table}.{fk.ref_column}"
            if (edge in accepted_pairs and src_dotted in contract.columns
                    and dst_dotted in contract.columns):
                defs.append(f'  FOREIGN KEY ("{fk.src_column}") '
                            f'REFERENCES "{fk.ref_table}" ("{fk.ref_column}")')
        body = ",\n".join(defs)
        chunk = f'CREATE TABLE "{t}" (\n{body}\n);'
        chunks.append(chunk)

    # inferred (non-declared) accepted joins as comments
    declared = {RelationCandidate(fk.src_table, fk.src_column, fk.ref_table,
                                  fk.ref_column, 1.0).edge_id
                for fk in schema.declared_fks}
    for j in contract.joins:
        if j.edge_id not in declared:
            chunks.append(f'-- inferred join: {j.src_table}.{j.src_column} = '
                          f'{j.dst_table}.{j.dst_column}')
    return "\n\n".join(chunks)


def _evidence_bundle(ground: GroundResult, observations: list[dict],
                     contract: CommitContract) -> str:
    lines = ["Column meanings and requirements:"]
    for r in ground.requirements:
        lines.append(f"  [{r.id}] {r.kind}: {r.need}")
    if observations:
        lines.append("Observations:")
        for o in observations:
            lines.append(f"  {o['probe_id']} ({o['kind']}): {o['status']}")
    if contract.coverage:
        lines.append("Coverage:")
        for cov in contract.coverage:
            lines.append(f"  {cov.requirement_id}: {cov.state} {cov.columns}")
    return "\n".join(lines)


def generate_sql(llm, schema: Schema, question: str, evidence: str,
                 ground: GroundResult, observations: list[dict],
                 contract: CommitContract) -> str:
    """Final generation (call 6): one cleaned SQLite query (K=1)."""
    ddl = render_ddl(schema, contract)
    bundle = _evidence_bundle(ground, observations, contract)
    # Message order: Schema, Question, Evidence, Evidence bundle, Contract, SQL.
    user = (f"Schema:\n{ddl}\n\nQuestion: {question}\n")
    if evidence:
        user += f"\nEvidence: {evidence}\n"
    user += (f"\nEvidence bundle:\n{bundle}\n"
             f"\nGrounding contract reason: {contract.reason}\n\nSQL:")
    text = llm.complete(prompts.GENERATE_SYS, user)
    return clean_final_sql(extract_sql(text))


def clean_final_sql(sql: str) -> str:
    """Strip fences/semicolons; require a single SELECT/CTE, else empty string."""
    sql = extract_sql(sql)
    low = sql.lower().lstrip()
    if not (low.startswith("select") or low.startswith("with")):
        return ""
    # exclude obvious metadata references the runner forbids
    forbidden = ("pragma", "sqlite_master", "sqlite_schema")
    if any(tok in low for tok in forbidden):
        return ""
    return sql
