"""Shared helpers for the locally adapted baselines.

Every baseline runs on the SAME backend as GraphWalkerSQL (shared ``gws.llm.LLM``,
datasets, SQLite execution, and official evaluators), matching the paper's
"locally adapted baselines on the same backend" protocol (Appendix "Resource-
Aligned Protocol"). Each baseline implements its own distinguishing mechanism
and emits exactly one final SQL, scored by the official evaluator.
"""
from __future__ import annotations

import re
from collections import Counter

from gws.execute import result_signature, run_query
from gws.schema import Schema

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def full_ddl(schema: Schema, tables: list[str] | None = None) -> str:
    """Annotated full DDL (with BIRD column meanings when available)."""
    return schema.to_ddl_text(tables=tables, annotate=True)


def compact_schema(schema: Schema, tables: list[str] | None = None) -> str:
    return schema.compact_text(tables=tables)


def extract_sql(text: str) -> str:
    if not text:
        return ""
    m = _FENCE.search(text)
    sql = (m.group(1) if m else text).strip()
    return sql.rstrip(";").strip()


def clean_sql(sql: str) -> str:
    """Keep a single SELECT/CTE; drop metadata references the runner forbids."""
    sql = extract_sql(sql)
    low = sql.lower().lstrip()
    if not (low.startswith("select") or low.startswith("with")):
        return ""
    if any(tok in low for tok in ("pragma", "sqlite_master", "sqlite_schema")):
        return ""
    return sql


def common_context(schema: Schema, question: str, evidence: str,
                   tables: list[str] | None = None) -> str:
    parts = [full_ddl(schema, tables), "", f"Question: {question}"]
    if evidence:
        parts.append(f"External evidence: {evidence}")
    return "\n".join(parts)


def self_consistency(sqlite_path: str, candidates: list[str]) -> str:
    """Pick the SQL whose execution result is most common (C3-style voting).

    Candidates that fail to execute are ignored for voting; if none execute, the
    first non-empty candidate is returned verbatim.
    """
    cands = [c for c in candidates if c]
    if not cands:
        return "SELECT 1"
    sig_to_sql: dict[str, str] = {}
    votes: Counter = Counter()
    for sql in cands:
        sig = result_signature(sqlite_path, sql)
        if sig is None:
            continue
        votes[sig] += 1
        sig_to_sql.setdefault(sig, sql)
    if not votes:
        return cands[0]
    best_sig, _ = votes.most_common(1)[0]
    return sig_to_sql[best_sig]


def runs_ok(sqlite_path: str, sql: str) -> tuple[bool, str]:
    res = run_query(sqlite_path, sql, max_rows=5)
    if res.get("ok"):
        return True, ""
    return False, str(res.get("error", ""))[:200]
