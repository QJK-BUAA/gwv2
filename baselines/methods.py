"""Locally adapted baselines (paper Table: End-to-End EX).

Each ``run_*`` function returns a dict with at least ``pred_sql`` plus light
telemetry (``model_calls``, ``reads``). All share the GraphWalkerSQL backend and
emit one final SQL. These are faithful reductions of each method's distinguishing
idea onto the shared protocol -- not vendored copies of eight heterogeneous
repos -- so the comparison is controlled (same model, same subset, same official
evaluator), exactly as the paper's resource-aligned protocol requires.

Methods:
  full_schema   -- complete-schema control: one generation call over all tables.
  din_sql       -- schema linking -> classification -> decomposed generation.
  c3_sql        -- clear prompting + calibration bias + self-consistency vote.
  ta_sql        -- task alignment: dummy logical plan then symbol->SQL.
  rsl_sql       -- bidirectional schema linking then generate on reduced schema.
  schemagraphsql-- FK-graph shortest-path sub-schema then generate.
  pv_sql        -- probe exact values then generate then verify/repair.
  apex_sql      -- profile/plan then generate then execute-verify repair.
"""
from __future__ import annotations

import json

from gws.parsing import extract_json
from gws.schema import Schema
from .common import (clean_sql, common_context, compact_schema, full_ddl,
                     runs_ok, self_consistency)


# --------------------------------------------------------------------------- #
# Full-schema control
# --------------------------------------------------------------------------- #
def run_full_schema(llm, schema: Schema, question: str, evidence: str,
                    sqlite_path: str) -> dict:
    sys = ("You are a SQLite expert. Write ONE SQLite query for the question "
           "using the full schema. Return only SQL in a ```sql block. Use "
           "floating-point arithmetic for ratios.")
    text = llm.complete(sys, common_context(schema, question, evidence))
    return {"pred_sql": clean_sql(text) or "SELECT 1",
            "model_calls": 1, "reads": 0}


# --------------------------------------------------------------------------- #
# DIN-SQL: schema linking -> difficulty classification -> decomposed generation
# --------------------------------------------------------------------------- #
def run_din_sql(llm, schema: Schema, question: str, evidence: str,
                sqlite_path: str) -> dict:
    ctx = common_context(schema, question, evidence)
    link_sys = ('Identify the schema links (table.column) and any join '
                'conditions needed. Return JSON: {"columns":["t.c"],'
                '"joins":["t1.c1 = t2.c2"],"difficulty":"easy|medium|hard"}.')
    link = extract_json(llm.complete(link_sys, ctx), ("columns",)) or {}
    cols = link.get("columns") or []
    joins = link.get("joins") or []
    diff = link.get("difficulty") or "medium"

    gen_sys = ("You are a SQLite expert using the DIN-SQL decomposed strategy. "
               "Given schema links and difficulty, think step by step for hard "
               "questions (sub-questions -> intermediate -> final), then output "
               "ONE final SQLite query in a ```sql block.")
    gen_user = (f"{ctx}\n\nSchema links: {cols}\nJoins: {joins}\n"
                f"Difficulty: {diff}\n\nFinal SQL:")
    text = llm.complete(gen_sys, gen_user)
    return {"pred_sql": clean_sql(text) or "SELECT 1",
            "model_calls": 2, "reads": 0}


# --------------------------------------------------------------------------- #
# C3-SQL: clear prompting + calibration + self-consistency
# --------------------------------------------------------------------------- #
def run_c3_sql(llm, schema: Schema, question: str, evidence: str,
               sqlite_path: str, n_votes: int = 3) -> dict:
    # Clear Prompting: schema recall first, then generation with a calibration
    # instruction, voting over sampled candidates (self-consistency).
    ctx = common_context(schema, question, evidence)
    recall_sys = ('Select only the tables and columns relevant to the question. '
                  'Return JSON {"tables":[...],"columns":["t.c"]}.')
    recall = extract_json(llm.complete(recall_sys, ctx), ("tables",)) or {}
    tables = [t for t in (recall.get("tables") or []) if t in schema.tables]
    ddl = full_ddl(schema, tables or None)

    gen_sys = ("You are a SQLite expert (C3 calibrated). Avoid selecting "
               "redundant columns; prefer the simplest correct query; use "
               "floating-point division for ratios. Output ONE query in a "
               "```sql block.")
    gen_user = f"{ddl}\n\nQuestion: {question}\n"
    if evidence:
        gen_user += f"External evidence: {evidence}\n"
    candidates = []
    for i in range(n_votes):
        temp = 0.0 if i == 0 else 0.7
        candidates.append(clean_sql(llm.complete(gen_sys, gen_user, temperature=temp)))
    pred = self_consistency(sqlite_path, candidates)
    return {"pred_sql": pred or "SELECT 1",
            "model_calls": 1 + n_votes, "reads": n_votes}


# --------------------------------------------------------------------------- #
# TA-SQL: task alignment (logical plan then symbolic -> SQL)
# --------------------------------------------------------------------------- #
def run_ta_sql(llm, schema: Schema, question: str, evidence: str,
               sqlite_path: str) -> dict:
    ctx = common_context(schema, question, evidence)
    plan_sys = ('Task-align the question: produce a dummy logical plan that '
                'lists the entities, the columns each maps to, aggregations, '
                'and conditions, avoiding hallucinated schema. Return JSON '
                '{"entities":[{"name":...,"column":"t.c"}],"aggregations":[...],'
                '"conditions":[...]}.')
    plan = extract_json(llm.complete(plan_sys, ctx), ("entities",)) or {}
    gen_sys = ("You are a SQLite expert. Convert the aligned logical plan into "
               "ONE SQLite query, using only schema identifiers. Output in a "
               "```sql block.")
    gen_user = f"{ctx}\n\nAligned plan:\n{json.dumps(plan, ensure_ascii=False)}\n\nSQL:"
    text = llm.complete(gen_sys, gen_user)
    return {"pred_sql": clean_sql(text) or "SELECT 1",
            "model_calls": 2, "reads": 0}


# --------------------------------------------------------------------------- #
# RSL-SQL: bidirectional schema linking -> generation on reduced schema
# --------------------------------------------------------------------------- #
def run_rsl_sql(llm, schema: Schema, question: str, evidence: str,
                sqlite_path: str) -> dict:
    ctx = common_context(schema, question, evidence)
    # Forward: a preliminary SQL to harvest a column set (full-schema prompt).
    prelim = clean_sql(llm.complete(
        "Write a preliminary SQLite query (it may be imperfect) in a ```sql "
        "block for the question using the full schema.", ctx))
    # Backward: bidirectional linking that unions preliminary-SQL columns with
    # question-driven columns, then selects a reduced schema.
    link_sys = ('Bidirectional schema linking: union the columns used by the '
                'preliminary SQL with columns the question needs. Return JSON '
                '{"tables":[...],"columns":["t.c"]}.')
    link = extract_json(llm.complete(link_sys, f"{ctx}\n\nPreliminary SQL:\n{prelim}"),
                        ("tables",)) or {}
    tables = [t for t in (link.get("tables") or []) if t in schema.tables]
    ddl = full_ddl(schema, tables or None)
    gen = clean_sql(llm.complete(
        "You are a SQLite expert. Using the reduced schema, write ONE final "
        "SQLite query in a ```sql block.",
        f"{ddl}\n\nQuestion: {question}\n"
        + (f"External evidence: {evidence}\n" if evidence else "")))
    # Binary selection + light self-correction within budget.
    ok, err = runs_ok(sqlite_path, gen) if gen else (False, "empty")
    if not ok:
        gen2 = clean_sql(llm.complete(
            "Fix the SQLite query so it executes. Output ONE query in a ```sql block.",
            f"{ddl}\n\nQuestion: {question}\nBroken SQL:\n{gen}\nError: {err}\n"))
        gen = gen2 or gen
        calls = 4
    else:
        calls = 3
    return {"pred_sql": gen or "SELECT 1", "model_calls": calls, "reads": 1}


# --------------------------------------------------------------------------- #
# SchemaGraphSQL: FK-graph shortest-path sub-schema -> generation
# --------------------------------------------------------------------------- #
def run_schemagraphsql(llm, schema: Schema, question: str, evidence: str,
                       sqlite_path: str) -> dict:
    from collections import defaultdict, deque

    adj: dict[str, set[str]] = {t: set() for t in schema.table_names()}
    for fk in schema.declared_fks:
        if fk.src_table != fk.ref_table and fk.src_table in adj and fk.ref_table in adj:
            adj[fk.src_table].add(fk.ref_table)
            adj[fk.ref_table].add(fk.src_table)

    ctx = compact_schema(schema)
    ep_sys = ('Pick the coarse source and destination tables for the question. '
              'Return JSON {"source":"table","destination":"table"}.')
    ep = extract_json(llm.complete(ep_sys, f"{ctx}\n\nQuestion: {question}"),
                     ("source",)) or {}
    src, dst = ep.get("source"), ep.get("destination")

    def shortest(a, b):
        if a not in adj or b not in adj:
            return []
        if a == b:
            return [a]
        dist, preds = {a: 0}, defaultdict(list)
        q = deque([a])
        while q:
            n = q.popleft()
            for m in sorted(adj[n]):
                if m not in dist:
                    dist[m] = dist[n] + 1
                    preds[m].append(n)
                    q.append(m)
                elif dist[m] == dist[n] + 1:
                    preds[m].append(n)
        if b not in dist:
            return []
        # reconstruct one shortest path
        path, cur = [b], b
        while cur != a:
            cur = preds[cur][0]
            path.append(cur)
        return list(reversed(path))

    sub = set()
    if src in schema.tables:
        sub.add(src)
    if dst in schema.tables:
        sub.add(dst)
    for t in shortest(src, dst):
        sub.add(t)
    tables = [t for t in schema.table_names() if t in sub] or None
    ddl = full_ddl(schema, tables)
    gen = clean_sql(llm.complete(
        "You are a SQLite expert. Write ONE SQLite query over the given "
        "sub-schema in a ```sql block.",
        f"{ddl}\n\nQuestion: {question}\n"
        + (f"External evidence: {evidence}\n" if evidence else "")))
    return {"pred_sql": gen or "SELECT 1", "model_calls": 2, "reads": 0}


# --------------------------------------------------------------------------- #
# PV-SQL: probe exact values -> generate -> verify/repair
# --------------------------------------------------------------------------- #
def run_pv_sql(llm, schema: Schema, question: str, evidence: str,
               sqlite_path: str, max_probes: int = 3, max_repairs: int = 1) -> dict:
    from gws.execute import run_query
    ctx = common_context(schema, question, evidence)
    probe_sys = ('Request up to one read-only probe SELECT to pin down an exact '
                 'literal value or column format. Return JSON {"action":'
                 '"probe|done","probe_sql":"SELECT ... LIMIT 5"}.')
    observations = []
    reads = 0
    for _ in range(max_probes):
        obj = extract_json(llm.complete(
            probe_sys, f"{ctx}\n\nObservations so far: {observations}"),
            ("action",)) or {}
        if obj.get("action") != "probe":
            break
        psql = clean_sql(obj.get("probe_sql") or "")
        if not psql:
            break
        res = run_query(sqlite_path, psql, max_rows=5)
        reads += 1
        observations.append({"sql": psql,
                             "rows": res.get("rows") if res.get("ok") else res.get("error")})
    gen = clean_sql(llm.complete(
        "You are a SQLite expert. Using the probe observations, write ONE "
        "SQLite query in a ```sql block.",
        f"{ctx}\n\nProbe observations:\n{observations}\n\nSQL:"))
    calls = 1 + len(observations) + 1
    for _ in range(max_repairs):
        ok, err = runs_ok(sqlite_path, gen) if gen else (False, "empty")
        reads += 1
        if ok:
            break
        gen = clean_sql(llm.complete(
            "Repair the SQLite query so it executes. Output ONE query in a ```sql block.",
            f"{ctx}\n\nBroken SQL:\n{gen}\nError: {err}\n")) or gen
        calls += 1
    return {"pred_sql": gen or "SELECT 1", "model_calls": calls, "reads": reads}


# --------------------------------------------------------------------------- #
# APEX-SQL: profile/plan -> generate -> execute-verify repair
# --------------------------------------------------------------------------- #
def run_apex_sql(llm, schema: Schema, question: str, evidence: str,
                 sqlite_path: str, max_repairs: int = 2) -> dict:
    ctx = common_context(schema, question, evidence)
    plan_sys = ('Profile the question and plan the query: list required tables, '
                'columns, joins, and any value constraints to verify. Return '
                'JSON {"tables":[...],"columns":["t.c"],"plan":"..."}.')
    plan = extract_json(llm.complete(plan_sys, ctx), ("tables",)) or {}
    gen = clean_sql(llm.complete(
        "You are a SQLite expert. Follow the plan to write ONE SQLite query in "
        "a ```sql block.",
        f"{ctx}\n\nPlan:\n{json.dumps(plan, ensure_ascii=False)}\n\nSQL:"))
    calls, reads = 2, 0
    for _ in range(max_repairs):
        ok, err = runs_ok(sqlite_path, gen) if gen else (False, "empty")
        reads += 1
        if ok:
            break
        gen = clean_sql(llm.complete(
            "The query failed verification. Fix it to execute correctly and "
            "answer the question. Output ONE query in a ```sql block.",
            f"{ctx}\n\nPlan:\n{plan}\nBroken SQL:\n{gen}\nError: {err}\n")) or gen
        calls += 1
    return {"pred_sql": gen or "SELECT 1", "model_calls": calls, "reads": reads}


BASELINES = {
    "full-schema": run_full_schema,
    "din-sql": run_din_sql,
    "c3-sql": run_c3_sql,
    "ta-sql": run_ta_sql,
    "rsl-sql": run_rsl_sql,
    "schemagraphsql": run_schemagraphsql,
    "pv-sql": run_pv_sql,
    "apex-sql": run_apex_sql,
}
