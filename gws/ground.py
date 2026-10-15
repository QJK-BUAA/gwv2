"""Independent requirement and draft calls followed by bounded graph search."""
from dataclasses import dataclass, field
import json
from . import graph_builder, paper_prompts as prompts
from .parsing import extract_json, extract_sql
from .sql_analysis import legal_columns, legal_joins, resolve_sql_schema
from .types import Requirement


@dataclass
class GroundResult:
    requirements: list
    model_relations: list
    draft_sql: str
    draft_tables: list
    draft_columns: list
    candidates: list
    paths: list
    raw_requirements: dict = field(default_factory=dict)
    draft_analysis: dict = field(default_factory=dict)
    draft_execution: dict = field(default_factory=dict)
    decision_diagnostics: list = field(default_factory=list)


def catalog(schema, columns=None):
    selected = set(columns) if columns is not None else None
    return [{"column": f"{t}.{c.name}", "type": c.type, "pk": c.pk,
             **schema.descriptions.get(t.lower(), {}).get(c.name.lower(), {})}
            for t, cs in schema.tables.items() for c in cs
            if selected is None or f"{t}.{c.name}" in selected]


def _common_context(schema, question, evidence, tables=None):
    return (f"Visible schema (relationships are uncertain):\n"
            f"{schema.to_ddl_text(tables=tables, trusted_er=True, annotate=False)}\n"
            f"Column catalog:\n{json.dumps(catalog(schema), ensure_ascii=False, sort_keys=True)}"
            f"\n\nQuestion: {question}\nEvidence: {evidence}")


def _parse_requirements(obj, schema):
    out, ids = [], set()
    for i, item in enumerate((obj or {}).get("requirements", [])):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or f"r{i+1}")
        if rid in ids:
            raise ValueError("Duplicate requirement id")
        ids.add(rid)
        out.append(Requirement(rid, str(item.get("kind", "output")), str(item.get("need", "")),
                               legal_columns(schema, item.get("columns", [])),
                               legal_columns(schema, item.get("alternatives", [])),
                               str(item.get("reason", ""))))
    if not out:
        raise ValueError("No semantic requirements returned")
    return out


def parse_draft_objects(sql, schema):
    parsed = resolve_sql_schema(sql, schema)
    return parsed["tables"], parsed["columns"]


def ground(llm, schema, question, evidence, include_structural=True,
           include_model=True, include_draft=True):
    ctx = _common_context(schema, question, evidence)
    req = extract_json(llm.complete(prompts.REQUIREMENTS_SYS, ctx, stage="requirements"),
                       ("requirements",))
    requirements = _parse_requirements(req, schema)
    relations = legal_joins(schema, (req or {}).get("joins", (req or {}).get("relations", [])))
    draft = {"sql": "", "tables": [], "columns": [], "joins": [], "diagnostics": []}
    if include_draft:
        draft = resolve_sql_schema(extract_sql(llm.complete(prompts.DRAFT_SYS, ctx,
                                                          stage="independent_draft")), schema)
    candidates = graph_builder.build_candidate_edges(schema,
        model_relations=relations + draft["joins"], include_structural=include_structural,
        include_model=include_model)
    terminals = set(draft["tables"])
    terminals.update(c.split(".", 1)[0] for r in requirements for c in r.columns)
    paths = graph_builder.search_paths(candidates, sorted(terminals))
    return GroundResult(requirements, relations, draft["sql"], draft["tables"],
                        draft["columns"], candidates, paths, req or {}, draft)
