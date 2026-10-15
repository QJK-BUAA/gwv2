"""Schema extraction from a SQLite database.

Produces a dialect-agnostic ``Schema`` holding tables, columns, types, declared
foreign keys and lightweight per-column statistics (distinct count, null count,
approximate uniqueness). The statistics feed the white-box belief scores in
``belief.py`` (uniqueness -> "looks like a key", value hits -> filter values).
"""
from __future__ import annotations

import re
import sqlite3
import json
from dataclasses import dataclass, field
from .sqlite_access import read_only_uri


@dataclass
class Column:
    name: str
    type: str
    pk: bool = False
    # Filled lazily by ``collect_column_stats``.
    n_distinct: int | None = None
    n_rows: int | None = None
    n_null: int | None = None

    @property
    def uniqueness(self) -> float:
        """Distinct / non-null rows in [0, 1]; ~1.0 means key-like."""
        if not self.n_rows:
            return 0.0
        denom = self.n_rows - (self.n_null or 0)
        if denom <= 0:
            return 0.0
        return min(1.0, (self.n_distinct or 0) / denom)


@dataclass
class ForeignKey:
    src_table: str
    src_column: str
    ref_table: str
    ref_column: str

    def undirected_key(self) -> tuple:
        return tuple(sorted([self.src_table, self.ref_table]))


@dataclass
class Schema:
    name: str
    tables: dict[str, list[Column]] = field(default_factory=dict)
    declared_fks: list[ForeignKey] = field(default_factory=list)
    raw_ddl: dict[str, str] = field(default_factory=dict)
    # {table_lower: {col_lower: {"desc": str, "value_desc": str}}}, loaded from a
    # benchmark's column-description files (e.g. BIRD database_description/*.csv).
    descriptions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def has_declared_fks(self) -> bool:
        return len(self.declared_fks) > 0

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def column(self, table: str, column: str) -> Column | None:
        if not isinstance(table, str) or not isinstance(column, str):
            return None
        actual = next((t for t in self.tables if t.lower() == table.lower()), table)
        for c in self.tables.get(actual, []):
            if c.name.lower() == column.lower():
                return c
        return None

    def to_ddl_text(
        self,
        tables: list[str] | None = None,
        annotate: bool = True,
        trusted_er: bool = False,
    ) -> str:
        names = tables if tables is not None else self.table_names()
        chunks = []
        for t in names:
            if t not in self.tables:
                continue
            if trusted_er:
                chunks.append(self._trusted_table_ddl(t, names))
            elif self.raw_ddl.get(t):
                chunks.append(self.raw_ddl[t].strip().rstrip(";") + ";")
            else:
                cols = ",\n    ".join(f'"{c.name}" {c.type}' for c in self.tables[t])
                chunks.append(f'CREATE TABLE "{t}" (\n    {cols}\n);')
            # Attach human column meanings + value notes when available (e.g. BIRD
            # database_description). This is the main lever for correct column
            # binding (which column "Low Grade"/"District Name" refer to).
            if annotate and self.descriptions:
                notes = self._column_notes(t)
                if notes:
                    chunks.append(notes)
        return "\n\n".join(chunks)

    def _trusted_table_ddl(
        self,
        table: str,
        visible_tables: list[str],
    ) -> str:
        def quote(identifier: str) -> str:
            return '"' + identifier.replace('"', '""') + '"'

        definitions = [
            f"{quote(column.name)} {column.type}"
            for column in self.tables[table]
        ]
        primary_keys = [
            quote(column.name)
            for column in self.tables[table]
            if column.pk
        ]
        if primary_keys:
            definitions.append(
                "PRIMARY KEY (" + ", ".join(primary_keys) + ")"
            )
        visible = set(visible_tables)
        for foreign_key in self.declared_fks:
            if (
                foreign_key.src_table != table
                or foreign_key.ref_table not in visible
            ):
                continue
            definitions.append(
                f"FOREIGN KEY ({quote(foreign_key.src_column)}) "
                f"REFERENCES {quote(foreign_key.ref_table)} "
                f"({quote(foreign_key.ref_column)})"
            )
        body = ",\n    ".join(definitions)
        return f"CREATE TABLE {quote(table)} (\n    {body}\n);"

    def _column_notes(self, table: str) -> str:
        d = self.descriptions.get(table.lower())
        if not d:
            return ""
        lines = []
        for c in self.tables.get(table, []):
            info = d.get(c.name.lower())
            if not info:
                continue
            desc, vd = info.get("desc", ""), info.get("value_desc", "")
            if not desc and not vd:
                continue
            line = f'--   "{c.name}": {desc}'.rstrip()
            if vd:
                line += f" | values: {vd}"
            lines.append(line)
        return (f"-- column meanings for {table}:\n" + "\n".join(lines)) if lines else ""

    def compact_text(self, tables: list[str] | None = None) -> str:
        names = tables if tables is not None else self.table_names()
        lines = []
        for t in names:
            cols = ", ".join(f"{c.name}:{c.type}" for c in self.tables.get(t, []))
            lines.append(f"{t}({cols})")
        return "\n".join(lines)


def load_foreign_key_registry(schema, path):
    """Benchmark-provided visible FK catalog, independent of query gold SQL."""
    records = json.loads(open(path).read())
    record = next((r for r in records if r.get("db_id") == schema.name), None)
    if record is None:
        raise ValueError("Database absent from benchmark catalog")
    tables = record.get("table_names_original", record.get("table_names"))
    columns = record.get("column_names_original", record.get("column_names"))
    actual_tables = {t.lower(): t for t in schema.tables}
    fks = []
    for i, j in record.get("foreign_keys", []):
        ti, ci = columns[i]
        tj, cj = columns[j]
        a, b = actual_tables.get(tables[ti].lower()), actual_tables.get(tables[tj].lower())
        if a is None or b is None or schema.column(a, ci) is None or schema.column(b, cj) is None:
            raise ValueError("Invalid benchmark FK endpoints")
        fks.append(ForeignKey(a, schema.column(a, ci).name, b, schema.column(b, cj).name))
    schema.declared_fks = fks
    schema.raw_ddl = {t: schema._trusted_table_ddl(t, schema.table_names()) for t in schema.tables}


def extract_schema(sqlite_path: str, name: str | None = None) -> Schema:
    con = sqlite3.connect(read_only_uri(sqlite_path), uri=True)
    cur = con.cursor()
    schema = Schema(name=name or sqlite_path.split("/")[-1].replace(".sqlite", ""))

    rows = cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for tname, tsql in rows:
        schema.raw_ddl[tname] = tsql or ""
        cols: list[Column] = []
        quoted = tname.replace('"', '""')
        for _cid, cname, ctype, _nn, _dflt, pk in cur.execute(
            f'PRAGMA table_info("{quoted}")'
        ):
            cols.append(Column(name=cname, type=(ctype or "TEXT").upper(), pk=bool(pk)))
        schema.tables[tname] = cols

        for fk in cur.execute(f'PRAGMA foreign_key_list("{quoted}")'):
            ref_table, from_col, to_col = fk[2], fk[3], fk[4]
            if to_col is None:
                pkcols = [c.name for c in schema.tables.get(ref_table, []) if c.pk]
                to_col = pkcols[0] if pkcols else from_col
            schema.declared_fks.append(
                ForeignKey(src_table=tname, src_column=from_col,
                           ref_table=ref_table, ref_column=to_col)
            )
    con.close()
    # Implicit referenced PKs can point to a table loaded later.
    for fk in schema.declared_fks:
        if schema.column(fk.ref_table, fk.ref_column) is None:
            keys = [c.name for c in schema.tables.get(fk.ref_table, []) if c.pk]
            if len(keys) == 1:
                fk.ref_column = keys[0]
    return schema


def collect_column_stats(sqlite_path: str, schema: Schema,
                         max_tables: int = 60, timeout: float = 15.0) -> None:
    """Populate n_rows / n_distinct / n_null for each column (best-effort).

    Bounded so it stays cheap on large schemas; failures leave stats as None,
    in which case belief scoring falls back to name-only signals.
    """
    import threading
    con = sqlite3.connect(read_only_uri(sqlite_path), uri=True)
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        cur = con.cursor()
        for i, (tname, cols) in enumerate(schema.tables.items()):
            if i >= max_tables:
                break
            try:
                n_rows = cur.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
            except Exception:  # noqa: BLE001
                continue
            for c in cols:
                c.n_rows = n_rows
                try:
                    c.n_distinct = cur.execute(
                        f'SELECT COUNT(DISTINCT "{c.name}") FROM "{tname}"'
                    ).fetchone()[0]
                    c.n_null = cur.execute(
                        f'SELECT COUNT(*) FROM "{tname}" WHERE "{c.name}" IS NULL'
                    ).fetchone()[0]
                except Exception:  # noqa: BLE001
                    c.n_distinct = c.n_null = None
    finally:
        timer.cancel()
        con.close()


def _clean_desc(text: str, max_len: int = 140) -> str:
    if not text:
        return ""
    t = text.replace("commonsense evidence:", " ").replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip().strip('"').strip()
    return t[:max_len]


def load_column_descriptions(desc_dir: str) -> dict:
    """Parse a directory of BIRD-style ``<table>.csv`` column-description files.

    Each CSV has columns original_column_name, column_name, column_description,
    data_format, value_description. Returns {table_lower: {col_lower: {desc,
    value_desc}}}. Robust to the mixed encodings BIRD ships (utf-8/latin-1/gbk).
    """
    import csv
    import glob
    import os

    out: dict[str, dict] = {}
    if not os.path.isdir(desc_dir):
        return out
    for path in sorted(glob.glob(os.path.join(desc_dir, "*.csv"))):
        table = os.path.basename(path)[:-4]
        rows = None
        for enc in ("utf-8-sig", "latin-1", "gbk"):
            try:
                with open(path, encoding=enc, newline="") as f:
                    rows = list(csv.DictReader(f))
                break
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            continue
        colmap: dict[str, dict] = {}
        for r in rows:
            oc = (r.get("original_column_name") or "").strip()
            if not oc:
                continue
            desc = _clean_desc(r.get("column_description") or "")
            vd = _clean_desc(r.get("value_description") or "")
            if desc or vd:
                colmap[oc.lower()] = {"desc": desc, "value_desc": vd}
        if colmap:
            out[table.lower()] = colmap
    return out
