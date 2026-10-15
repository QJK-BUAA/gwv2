"""Frozen source-backed linking procedures; all model/DB access is injected."""
import ast
from dataclasses import asdict
from pathlib import Path
import json
import re
import types
import sys
import concurrent.futures
import json_repair
from gws.schema import Schema
from gws.parsing import extract_json
from .common import schema_columns, normalize_selection, selection

VENDOR = Path(__file__).parent / "vendor"


def load_source(filename, injected=None, skip=()):
    path = VENDOR / filename
    tree = ast.parse(path.read_text(), filename=filename)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module in skip)
                 and not isinstance(node, (ast.Expr, ast.If))]
    module = types.ModuleType("frozen_" + path.stem)
    module.__file__ = str(path)
    module.__dict__.update(injected or {})
    sys.modules[module.__name__] = module
    exec(compile(tree, filename, "exec"), module.__dict__)
    return module


TA = load_source("ta_prompts.py")
RSL = load_source("rsl_prompts.py")
SG_PROMPTS = load_source("sg_prompts.py")
SG = load_source("sg_linker.py", {"LLM": object, "Schema": Schema,
    **{k: v for k, v in vars(SG_PROMPTS).items() if k.startswith("PROMPT_")}},
    skip=("llm", "schema", "prompts"))
PV = load_source("pv_sql.py", {"DatabaseEnv": object, "SQLiteEnv": object}, skip=("db",))
DIN_C3 = load_source("din_c3.py", {"schema_columns": schema_columns,
    "SafeDB": object}, skip=("gw_recovery.bridge", "gw_recovery.runtime"))


def full_schema(schema, llm, db, question, evidence):
    return selection(schema, schema_columns(schema))


def schemagraphsql(schema, llm, db, question, evidence):
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(schema.table_names())
    for fk in schema.declared_fks:
        if not graph.has_edge(fk.src_table, fk.ref_table):
            graph.add_edge(fk.src_table, fk.ref_table, joins=[])
        graph[fk.src_table][fk.ref_table]["joins"].append(
            (fk.src_table, fk.src_column, fk.ref_table, fk.ref_column))
    result = SG.link(question + ("\nEvidence: " + evidence if evidence else ""),
                     schema, graph, llm, config="force-union")
    return selection(schema, schema_columns(schema, result.linked_tables),
                     explicit_joins=result.join_conditions, details=asdict(result))


def ta_sql(schema, llm, db, question, evidence):
    pk = {t: [c.name for c in cs if c.pk] for t, cs in schema.tables.items()}
    fk = {f"{f.src_table}.{f.src_column}": f"{f.ref_table}.{f.ref_column}"
          for f in schema.declared_fks}
    raw = llm.complete("", TA.dummy_sql_prompt.format(
        database_schema=schema.compact_text(), primary_key_dic=pk, foreign_key_dic=fk,
        question_prompt=question, evidence=evidence))
    tables = [t for t in schema.table_names() if t in raw]
    names = {c.name for cs in schema.tables.values() for c in cs if c.name in raw}
    columns = {f"{t}.{c.name}" for t in tables for c in schema.tables[t] if c.name in names}
    return selection(schema, columns, details={"dummy_sql_response": raw})


def rsl_sql(schema, llm, db, question, evidence):
    info = "### Sqlite SQL tables, with their properties:\n" + schema.to_ddl_text(
        trusted_er=True, annotate=False)
    raw = llm.complete(RSL.TABLE_AUG_INSTRUCTION,
        info + f"\n### definition: {evidence}\n### Question: {question}")
    forward = extract_json(raw, ()) or {}
    prompt = (info + f"\n### tables: {forward.get('tables', [])}"
              f"\n### columns: {forward.get('columns', [])}"
              f"\n### definition: {evidence}\n### Question: {question}")
    response = llm.complete(RSL.SQL_GENERATION_INSTRUCTION, prompt)
    sql = str((extract_json(response, ()) or {}).get("sql", response))
    fcols = {str(c).replace("`", "").lower() for c in forward.get("columns", [])}
    cols = {f"{t}.{c.name}" for t, cs in schema.tables.items() for c in cs
            if f"{t}.{c.name}".lower() in fcols or c.name.lower() in sql.lower()
            or c.name.lower() in evidence.lower()}
    return selection(schema, cols, details={"forward": forward, "preliminary_sql": sql})


def pv_sql(schema, llm, db, question, evidence):
    result = PV._ground(question, db, llm, evidence=evidence, max_probes=5)
    return selection(schema, normalize_selection(schema, result.relevant_columns),
                     details=asdict(result))


def din_sql(*args):
    return DIN_C3.din_sql(*args)


def c3_sql(*args):
    return DIN_C3.c3_sql(*args)


def extract_all_blocks(content, code_format):
    # Released APEX utils.extract_all_blocks, restricted to its standalone helper.
    if content is None:
        return []
    blocks, start = [], 0
    while True:
        match = re.search(f"```\\s*{re.escape(code_format)}", content[start:])
        if not match:
            break
        begin = start + match.end()
        end = content.find("```", begin)
        if end == -1:
            break
        blocks.append(content[begin:end].strip())
        start = end + 3
    return blocks


def apex_runtime(schema, llm, db):
    class GPTChat:
        def __init__(self, model, temperature=0, **kwargs):
            self.messages = []

        def init_messages(self):
            self.messages = []

        def get_model_response_txt(self, prompt, tag=""):
            self.messages.append({"role": "user", "content": prompt})
            result = llm.llm.complete_messages_with_metadata(
                self.messages, temperature=0, seed=42, stage="apex:" + tag)
            self.messages.append({"role": "assistant", "content": result.text})
            return result.text

    class SqlEnv:
        def start_db_sqlite(self, path):
            if str(path) != str(db.path):
                raise ValueError("Unexpected database")

        def close_db(self):
            pass

        def execute_sql_api(self, sql, path, max_len=1000, **kwargs):
            import sqlparse
            outputs = []
            for statement in sqlparse.split(sql):
                names, rows, error = db.execute(statement)
                outputs.append("##ERROR##" + error if error else
                               json.dumps({"columns": names, "rows": rows}, default=str))
            return "\n".join(outputs)[:max_len]

    module = load_source("apex_schema_linking.py",
        {"GPTChat": GPTChat, "SqlEnv": SqlEnv, "extract_all_blocks": extract_all_blocks,
         "get_table_schema_info": None, "generate_table_info": None},
        skip=("chat", "sql", "utils", "preprocess"))
    module.concurrent = types.SimpleNamespace(futures=types.SimpleNamespace(
        ThreadPoolExecutor=concurrent.futures.ThreadPoolExecutor, as_completed=lambda fs: list(fs)))
    tables = {}
    for table, cs in schema.tables.items():
        columns = []
        for c in cs:
            desc = ["Primary key" if c.pk else ""]
            desc.extend(f"Visible relationship: {f.src_table}.{f.src_column} = "
                        f"{f.ref_table}.{f.ref_column}" for f in schema.declared_fks
                        if (f.src_table, f.src_column) == (table, c.name)
                        or (f.ref_table, f.ref_column) == (table, c.name))
            columns.append({"name": c.name, "type": c.type,
                            "description": "; ".join(x for x in desc if x)})
        tables[f"{schema.name}.{table}"] = {"table_name": table, "columns": columns}
    linker = module.SchemaLinking(model=llm.model, max_workers=1, cache_dir=None)
    linker._generate_tables_dict_from_db = lambda *args, **kwargs: tables
    return module, linker, tables


def apex_selection(schema, result):
    raw = {t: info.get("relevant_columns", []) for t, info in
           result.get("refined_schema", {}).items()}
    return selection(schema, normalize_selection(schema, raw), details=result)


def apex_sql(schema, llm, db, question, evidence):
    _, linker, _ = apex_runtime(schema, llm, db)
    result = linker.run_schema_linking(question, "runtime", evidence=evidence,
                                       sqlite_path=db.path, enable_pruning=True)
    return apex_selection(schema, result)


ADAPTERS = {"full-schema": full_schema, "din-sql": din_sql, "c3-sql": c3_sql,
            "ta-sql": ta_sql, "rsl-sql": rsl_sql, "schemagraphsql": schemagraphsql,
            "pv-sql": pv_sql, "apex-sql": apex_sql}
