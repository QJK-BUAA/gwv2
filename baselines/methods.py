"""Eight archived linker adaptations and the separate resource-aligned controls."""
from dataclasses import asdict
from functools import partial
from gws.budget import Budget, BudgetExceeded
from gws.commit import clean_final_sql
from .common import SafeDB, FixedBackend, generate, selected_ddl
from .source_adapters import ADAPTERS, TA, apex_runtime, apex_selection


def _ta_aligned(schema, backend, db, question, evidence):
    linked = ADAPTERS["ta-sql"](schema, backend, db, question, evidence)
    ddl = selected_ddl(schema, linked["columns"])
    sr = backend.llm.complete("", TA.generate_sr.format(
        sr_example=TA.sr_examples, question=question, schema=linked["columns"],
        column_description=ddl, evidence=evidence), temperature=0, seed=42, stage="ta_sr")
    fks = {f"{f.src_table}.{f.src_column}": f"{f.ref_table}.{f.ref_column}"
           for f in schema.declared_fks}
    raw = backend.llm.complete("", TA.sr2sql.format(
        question=question, schema=linked["columns"], evidence=evidence,
        column_description=ddl, SR=sr, foreign_key_dic=fks),
        temperature=0, seed=42, stage="ta_candidate")
    candidate, checks = clean_final_sql(raw), []
    while db.budget.calls < 5 and db.budget.reads < 5:
        _, rows, error = db.execute(candidate)
        checks.append({"sql": candidate, "error": error, "rows": rows})
        if not error:
            break
        candidate = clean_final_sql(backend.llm.complete(
            "Correct the candidate SQLite SQL using the task-aligned SR and execution feedback. "
            "Return one query only.",
            f"Schema:\n{ddl}\nQuestion: {question}\nEvidence: {evidence}\n"
            f"SR: {sr}\nCandidate: {candidate}\nError: {error}",
            temperature=0, seed=42, stage="ta_correction"))
    linked["details"].update(sr=sr, candidate=candidate, verification=checks)
    return linked


def _apex_aligned(schema, backend, db, question, evidence):
    module, linker, tables = apex_runtime(schema, backend, db)
    # One planning sample plus merge (2), semantic analysis (1), and released
    # global refinement/verification/correction (up to 2); one final slot remains.
    module.PLAN_SAMPLES = 1
    linker._retry_with_backoff = lambda func, *args, **kwargs: func()
    plan = linker._generate_logical_plan(question, "runtime")
    summary = linker.summarize_tables(tables, question, "runtime", evidence, logical_plan=plan)
    result = linker.refine_schema_globally(
        tables, {}, question, db.path, "runtime", summarization_result=summary, evidence=evidence)
    linked = apex_selection(schema, result)
    linked["details"].update(logical_plan=plan, semantic_analysis=summary)
    return linked


def run_baseline(name, llm, schema, question, evidence, sqlite_path, aligned=False):
    llm.budget = Budget() if aligned else None
    db = SafeDB(sqlite_path, schema, llm.budget)
    backend = FixedBackend(llm, name + ":linking")
    linked = {"columns": [], "tables": []}
    pred, error = "", ""
    try:
        if aligned and name == "ta-sql":
            linked = _ta_aligned(schema, backend, db, question, evidence)
        elif aligned and name == "apex-sql":
            linked = _apex_aligned(schema, backend, db, question, evidence)
        else:
            linked = ADAPTERS[name](schema, backend, db, question, evidence)
        if not linked["columns"]:
            raise ValueError("Baseline selected no legal columns")
        pred = generate(schema, linked, question, evidence, llm)
        if not pred:
            error = "Invalid final SQL; no fallback"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    exposed = [{"left": f"{f.src_table}.{f.src_column}",
                "right": f"{f.ref_table}.{f.ref_column}"}
               for f in schema.declared_fks
               if f"{f.src_table}.{f.src_column}" in linked["columns"]
               and f"{f.ref_table}.{f.ref_column}" in linked["columns"]]
    return {"pred_sql": pred, "status": "ok" if pred else "failure", "error": error,
            "selected_columns": linked["columns"], "selected_tables": linked["tables"],
            "selected_joins": exposed, "details": linked.get("details", {}),
            "explicit_joins": linked.get("explicit_joins", []),
            "model_calls": llm.num_calls, "reads": len(db.queries), "read_trace": db.queries,
            "tokens": llm.stats(), "model_trace": list(llm.calls),
            "budget": llm.budget.snapshot() if aligned else None}


BASELINES = {name: partial(run_baseline, name) for name in ADAPTERS}
ALIGNED_BASELINES = {name + "-aligned": partial(run_baseline, name, aligned=True)
                     for name in ("ta-sql", "apex-sql")}
