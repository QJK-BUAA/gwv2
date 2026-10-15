"""Two-round reconciliation, deterministic coverage contract, one submission."""
from dataclasses import asdict
import json
from . import paper_prompts as prompts
from .ground import _common_context, catalog
from .parsing import extract_json, extract_sql
from .sql_analysis import legal_columns, legal_joins, validate_sql
from .types import CommitContract, CoverageItem, RelationCandidate


def stable_json(value):
    def stable(x):
        if isinstance(x, dict):
            return {k: stable(v) for k, v in x.items()
                    if k not in {"seconds", "response_id", "cached", "cache_source"}}
        if isinstance(x, (list, tuple)):
            return [stable(v) for v in x]
        return x
    return json.dumps(stable(value), ensure_ascii=False, sort_keys=True, default=str)


def remove_belief_signals(value):
    if isinstance(value, dict):
        return {k: remove_belief_signals(v) for k, v in value.items()
                if k not in {"prior", "priors", "posteriors", "posterior",
                             "belief_before", "belief_after", "map_recommendations",
                             "ranked_scores"}}
    if isinstance(value, tuple):
        return tuple(remove_belief_signals(v) for v in value)
    if isinstance(value, list):
        return [remove_belief_signals(v) for v in value]
    return value


def evidence_bundle(schema, ground, decisions, posteriors, maps, observations,
                    include_beliefs=True):
    bundle = {"catalog": catalog(schema),
              "requirements": [asdict(r) for r in ground.requirements],
              "independent_draft": ground.draft_analysis,
              "candidate_paths": [asdict(p) for p in ground.paths],
              "candidate_relations": [asdict(c) for c in ground.candidates],
              "observations": [{"kind": "draft_execution", "sql": ground.draft_sql,
                                **ground.draft_execution}] + observations,
              "decisions": [asdict(d) for d in decisions]}
    if include_beliefs:
        bundle.update(posteriors=posteriors,
                      map_recommendations=maps)
    else:
        bundle = remove_belief_signals(bundle)
        for obs in bundle["observations"]:
            obs.pop("score", None)
    return bundle


def _accepted_joins(schema, values):
    out = []
    for j in legal_joins(schema, values):
        st, sc = j["left"].split(".", 1)
        dt, dc = j["right"].split(".", 1)
        out.append(RelationCandidate(st, sc, dt, dc, 1., ("commit",)))
    return out


def _legal_columns(schema, cols):
    return legal_columns(schema, cols or [])


def _apply_coverage_contract(raw, schema, requirements):
    columns = set(legal_columns(schema, raw.get("columns", [])))
    joins = _accepted_joins(schema, raw.get("joins", []))
    coverage, unresolved = [], []
    records = {}
    duplicates = set()
    for item in raw.get("coverage", []):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id", item.get("requirement_id", "")))
        if rid in records:
            duplicates.add(rid)
        records[rid] = item
    for req in requirements:
        item = records.get(req.id, {})
        status = item.get("status", item.get("state", ""))
        reason = str(item.get("reason", "")).strip()
        mapped = legal_columns(schema, item.get("columns", []))
        valid = req.id not in duplicates and (
            status == "covered" and bool(mapped)
            or status == "replaced" and bool(reason) and (bool(mapped) or not req.columns)
            or status == "not_needed" and bool(reason))
        if valid:
            if status != "not_needed":
                columns.update(mapped)
            coverage.append(CoverageItem(req.id, status, mapped if status != "not_needed" else [], reason))
        else:
            restored = legal_columns(schema, req.columns)
            columns.update(restored)
            unresolved.append({"requirement_id": req.id, "columns": restored, "retain": True,
                               "reason": "Missing or invalid coverage; restored original columns"})
            coverage.append(CoverageItem(req.id, "unresolved", restored,
                                          "Missing or invalid coverage"))
    for u in raw.get("unresolved", []):
        if isinstance(u, dict):
            record = {**u, "columns": legal_columns(schema, u.get("columns", [u.get("label")]))}
            unresolved.append(record)
            if u.get("retain") is True:
                columns.update(record["columns"])
    for j in joins:
        columns.update([f"{j.src_table}.{j.src_column}", f"{j.dst_table}.{j.dst_column}"])
    if not columns:
        raise ValueError("Empty final selection")
    return CommitContract(sorted(columns), joins, coverage, unresolved, str(raw.get("reason", "")))


def _ordinary_filter(raw, schema):
    columns = set(legal_columns(schema, raw.get("columns", [])))
    joins = _accepted_joins(schema, raw.get("joins", []))
    for j in joins:
        columns.update([f"{j.src_table}.{j.src_column}", f"{j.dst_table}.{j.dst_column}"])
    if not columns:
        raise ValueError("Empty final selection")
    return CommitContract(sorted(columns), joins, reason=str(raw.get("reason", "")))


def commit(llm, schema, question, evidence, ground, decisions, posteriors, maps,
           observations, include_beliefs=True, coverage_contract=True):
    bundle = evidence_bundle(schema, ground, decisions, posteriors, maps,
                             observations, include_beliefs)
    context = _common_context(schema, question, evidence) + "\nGrounding evidence:\n" + stable_json(bundle)
    rec = extract_json(llm.complete(prompts.RECONCILE_SYS if coverage_contract else prompts.ORDINARY_SYS,
                                   context, stage="reconcile"), ("columns",))
    if rec is None:
        raise ValueError("Invalid reconciliation response")
    audit = extract_json(llm.complete(prompts.AUDIT_SYS if coverage_contract else prompts.ORDINARY_AUDIT_SYS,
                                     context + "\nCandidate contract to audit:\n" + stable_json(rec),
                                     stage="coverage_audit" if coverage_contract else "selection_audit"),
                         ("columns",))
    if audit is None:
        raise ValueError("Invalid audit response")
    contract = (_apply_coverage_contract(audit, schema, ground.requirements) if coverage_contract
                else _ordinary_filter(audit, schema))
    contract.evidence_bundle = bundle
    contract.raw_reconciliation, contract.raw_audit = rec, audit
    return contract


def contract_payload(contract):
    return {"columns": contract.columns, "task_supported_joins": [asdict(j) for j in contract.joins],
            "coverage": [asdict(c) for c in contract.coverage],
            "unresolved": contract.unresolved, "reason": contract.reason}


def render_ddl(schema, contract):
    def quote(x):
        return '"' + x.replace('"', '""') + '"'
    chosen, accepted = set(contract.columns), {j.edge_id for j in contract.joins}
    chunks = []
    for t, cs in schema.tables.items():
        retained = [c for c in cs if f"{t}.{c.name}" in chosen]
        if not retained:
            continue
        defs = [f"  {quote(c.name)} {c.type}" for c in retained]
        keys = [c.name for c in cs if c.pk]
        if keys and set(keys) <= {c.name for c in retained}:
            defs.append("  PRIMARY KEY (" + ", ".join(quote(c) for c in keys) + ")")
        for fk in schema.declared_fks:
            a, b = f"{fk.src_table}.{fk.src_column}", f"{fk.ref_table}.{fk.ref_column}"
            if fk.src_table == t and a in chosen and b in chosen and "|".join(sorted([a, b])) in accepted:
                defs.append(f"  FOREIGN KEY ({quote(fk.src_column)}) REFERENCES "
                            f"{quote(fk.ref_table)} ({quote(fk.ref_column)})")
        chunks.append(f"CREATE TABLE {quote(t)} (\n" + ",\n".join(defs) + "\n);")
    declared = {"|".join(sorted([f"{f.src_table}.{f.src_column}", f"{f.ref_table}.{f.ref_column}"]))
                for f in schema.declared_fks}
    for j in contract.joins:
        if j.edge_id not in declared:
            chunks.append(f"-- Task-supported relation proposal: {j.src_table}.{j.src_column} = "
                          f"{j.dst_table}.{j.dst_column}")
    return "\n\n".join(chunks)


def generate_sql(llm, schema, question, evidence, ground, observations, contract,
                 catalog_mode="full"):
    if catalog_mode not in {"full", "trimmed"}:
        raise ValueError("Unknown catalog mode")
    bundle = dict(contract.evidence_bundle)
    bundle["catalog"] = catalog(schema, None if catalog_mode == "full" else contract.columns)
    user = (f"Schema:\n{render_ddl(schema, contract)}\n\nQuestion: {question}\nEvidence: {evidence}"
            f"\nEvidence bundle:\n{stable_json(bundle)}"
            f"\nGrounding contract:\n{stable_json(contract_payload(contract))}\n\nSQL:")
    return clean_final_sql(llm.complete(prompts.GENERATE_SYS, user, stage="generation:0"))


def clean_final_sql(sql):
    sql = extract_sql(sql)
    try:
        validate_sql(sql)
        return sql
    except Exception:
        return ""
