"""Shared, metadata-isolated runtime for the paper's baseline adaptations."""
from dataclasses import asdict
import copy
import threading
from gws.execute import run_query
from gws.commit import clean_final_sql
from gws.schema import Schema


def schema_columns(schema, tables=None):
    return {f"{t}.{c.name}" for t in schema.table_names() if tables is None or t in tables
            for c in schema.tables[t]}


def normalize_selection(schema, raw):
    valid = {x.lower(): x for x in schema_columns(schema)}
    tables = {t.lower(): t for t in schema.table_names()}
    output = set()
    for table, columns in raw.items():
        table = str(table).removeprefix(schema.name + ".").strip('`"')
        actual = tables.get(table.lower())
        if actual is None or not isinstance(columns, list):
            continue
        for col in columns:
            if isinstance(col, dict):
                col = col.get("column_name", "")
            if isinstance(col, str):
                key = f"{actual}.{col.strip(chr(96) + chr(34))}".lower()
                if key in valid:
                    output.add(valid[key])
    return output


def selection(schema, columns, **extra):
    valid = schema_columns(schema)
    columns = sorted(set(columns) & valid)
    return {"columns": columns,
            "tables": [t for t in schema.table_names()
                       if any(c.startswith(t + ".") for c in columns)], **extra}


def selected_schema(schema, columns):
    chosen = copy.deepcopy(schema)
    keep = set(columns)
    chosen.tables = {t: [c for c in cs if f"{t}.{c.name}" in keep]
                     for t, cs in chosen.tables.items()}
    chosen.tables = {t: cs for t, cs in chosen.tables.items() if cs}
    chosen.declared_fks = [f for f in chosen.declared_fks
                          if f"{f.src_table}.{f.src_column}" in keep
                          and f"{f.ref_table}.{f.ref_column}" in keep]
    chosen.raw_ddl = {}
    # A subset of a composite key is not itself a primary key.
    for table, cs in chosen.tables.items():
        original_keys = {c.name for c in schema.tables[table] if c.pk}
        if not original_keys <= {c.name for c in cs}:
            for c in cs:
                c.pk = False
    return chosen


def selected_ddl(schema, columns):
    return selected_schema(schema, columns).to_ddl_text(trusted_er=True, annotate=False)


class SafeDB:
    def __init__(self, path, schema, budget=None):
        self.path, self.schema, self.budget = path, schema, budget
        self.queries = []
        self.lock = threading.Lock()

    def execute(self, sql, timeout=5., cap=20, params=None):
        result = run_query(self.path, sql, timeout=min(timeout, 5.), max_rows=min(cap, 2000),
                           params=params, budget=self.budget, stage="baseline_read")
        with self.lock:
            self.queries.append({"sql": sql, "params": params, **result})
        return result.get("columns", []), result.get("rows", []), result.get("error")

    def schema_overview(self):
        return self.schema.to_ddl_text(trusted_er=True, annotate=False)

    def get_foreign_keys(self):
        result = {}
        for f in self.schema.declared_fks:
            result.setdefault(f.src_table, []).append({
                "from_table": f.src_table, "from_column": f.src_column,
                "to_table": f.ref_table, "to_column": f.ref_column})
        return result


class FixedBackend:
    """Inject shared request settings, overriding released sampling defaults."""
    def __init__(self, llm, stage):
        self.llm, self.stage = llm, stage
        self.model = llm.model
        self.temperature = 0

    def complete(self, system, user, temperature=None, **kwargs):
        return self.llm.complete(system, user, temperature=0, seed=42,
                                 stage=kwargs.get("stage", self.stage))

    def complete_messages(self, messages, temperature=None, **kwargs):
        return self.complete_messages_with_metadata(messages).text

    def complete_messages_with_metadata(self, messages, temperature=None, **kwargs):
        return self.llm.complete_messages_with_metadata(
            messages, temperature=0, seed=42, stage=self.stage)

    def __call__(self, messages, temperature=0):
        return self.complete_messages(messages)


GEN_SYSTEM = """You are an expert SQLite text-to-SQL generator.
Write exactly one SQL query answering the question using the supplied schema.
Return only SQL. Return only the requested output columns.
Use the provided evidence. Ensure floating-point arithmetic for ratios.
Visible relationship metadata may be incomplete or noisy; use it as a candidate,
not proof that a join is relevant to the question."""


def generate(schema, linked, question, evidence, llm, stage="generation:baseline"):
    prompt = (f"Schema:\n{selected_ddl(schema, linked['columns'])}\n\n"
              f"Question: {question}\nEvidence: {evidence}\n\nSQL:")
    return clean_final_sql(llm.complete(GEN_SYSTEM, prompt, temperature=0, seed=42, stage=stage))


# Compatibility for external callers.
clean_sql = clean_final_sql
full_ddl = lambda schema, tables=None: schema.to_ddl_text(tables=tables, trusted_er=True)
compact_schema = lambda schema, tables=None: schema.compact_text(tables=tables)
