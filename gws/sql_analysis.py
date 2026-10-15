"""Scope-aware physical SQL objects; no evaluation reference enters inference."""
import re
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope


def legal_columns(schema, values):
    index = {f"{t}.{c.name}".lower(): f"{t}.{c.name}"
             for t, cs in schema.tables.items() for c in cs}
    return sorted({index[v.replace('"', "").replace("`", "").lower()]
                   for v in values if isinstance(v, str)
                   and v.replace('"', "").replace("`", "").lower() in index})


def legal_joins(schema, values):
    result = {}
    for item in values or []:
        if isinstance(item, str) and " = " in item:
            a, b = item.split(" = ", 1)
            item = {"left": a, "right": b}
        if not isinstance(item, dict):
            continue
        left = item.get("left") or f"{item.get('src_table', '')}.{item.get('src_column', '')}"
        right = item.get("right") or f"{item.get('dst_table', '')}.{item.get('dst_column', '')}"
        a, b = legal_columns(schema, [left]), legal_columns(schema, [right])
        if a and b and a != b:
            key = tuple(sorted((a[0], b[0])))
            result[key] = {"left": key[0], "right": key[1],
                           "reason": str(item.get("reason", ""))}
    return [result[k] for k in sorted(result)]


def validate_sql(sql):
    trees = sqlglot.parse(sql, read="sqlite")
    if len(trees) != 1 or not isinstance(trees[0], exp.Query):
        raise ValueError("Only one SELECT/CTE query is permitted")
    tree = trees[0]
    if any(t.name.lower().startswith(("sqlite_", "pragma_")) for t in tree.find_all(exp.Table)):
        raise ValueError("Metadata access disabled")
    # A Query may contain an UPDATE/DELETE CTE in other dialects.
    if any(isinstance(n, (exp.Insert, exp.Update, exp.Delete, exp.Create,
                          exp.Drop, exp.Command)) for n in tree.walk()):
        raise ValueError("Write statements disabled")
    if any(isinstance(n, exp.Anonymous) and n.name.lower() in
           {"load_extension", "readfile", "writefile"} for n in tree.walk()):
        raise ValueError("Extension functions disabled")
    return tree


def resolve_sql_schema(sql, schema, representative=True):
    """Port of the archived lexical-scope resolver, with diagnostics retained."""
    output = {"tables": [], "columns": [], "joins": [], "diagnostics": [], "sql": sql}
    match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.I | re.S)
    sql = (match.group(1) if match else sql).strip().rstrip(";")
    output["sql"] = sql
    try:
        tree = validate_sql(sql)
        try:
            tree = qualify(tree.copy(), dialect="sqlite",
                           schema={t: {c.name: c.type or "TEXT" for c in cs}
                                   for t, cs in schema.tables.items()},
                           validate_qualify_columns=False, infer_schema=False,
                           allow_partial_qualification=True)
        except Exception as exc:
            output["diagnostics"].append(f"qualification: {type(exc).__name__}")
        lookup = {t.lower(): t for t in schema.tables}
        valid = {f"{t}.{c.name}".lower(): f"{t}.{c.name}"
                 for t, cs in schema.tables.items() for c in cs}
        tables, columns, joins = set(), set(), set()

        def physical(scope, col):
            cursor = scope
            while cursor is not None:
                sources = {str(a).lower(): s for a, (_, s) in cursor.selected_sources.items()}
                if col.table:
                    source = sources.get(col.table.lower())
                    if isinstance(source, exp.Table):
                        actual = lookup.get(source.name.lower())
                        return valid.get(f"{actual}.{col.name}".lower()) if actual else None
                    if source is not None:
                        return None
                else:
                    hits, derived = [], False
                    for source in sources.values():
                        if isinstance(source, exp.Table):
                            actual = lookup.get(source.name.lower())
                            key = f"{actual}.{col.name}".lower()
                            if key in valid:
                                hits.append(valid[key])
                        else:
                            derived = True
                    if len(hits) == 1 and not derived:
                        return hits[0]
                    if hits or derived:
                        output["diagnostics"].append(f"ambiguous/derived: {col.sql()}")
                        return None
                cursor = cursor.parent
            return None

        for scope in traverse_scope(tree):
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table):
                    actual = lookup.get(source.name.lower())
                    if actual:
                        tables.add(actual)
                    else:
                        output["diagnostics"].append(f"unknown table: {source.name}")
            for col in scope.columns:
                resolved = physical(scope, col)
                if resolved:
                    columns.add(resolved)
                elif col.table and isinstance(scope.sources.get(col.table), exp.Table):
                    output["diagnostics"].append(f"unknown physical column: {col.sql()}")
            for node in scope.walk():
                if isinstance(node, exp.EQ) and isinstance(node.left, exp.Column) and isinstance(node.right, exp.Column):
                    a, b = physical(scope, node.left), physical(scope, node.right)
                    if a and b and a != b:
                        joins.add(tuple(sorted((a, b))))
            for node in scope.expression.expressions:
                if isinstance(node, exp.Star) or isinstance(node, exp.Column) and node.is_star:
                    for alias, (_, source) in scope.selected_sources.items():
                        if isinstance(node, exp.Column) and node.table.lower() != alias.lower():
                            continue
                        if isinstance(source, exp.Table):
                            actual = lookup.get(source.name.lower())
                            columns.update(v for k, v in valid.items()
                                           if actual and k.startswith(actual.lower() + "."))
        owners = {f"{t}.{c.name}": t for t, cs in schema.tables.items() for c in cs}
        if representative:
            for t in sorted(tables - {owners[c] for c in columns}):
                cs = schema.tables[t]
                if cs:
                    c = next((c for c in cs if c.pk), cs[0])
                    columns.add(f"{t}.{c.name}")
        output.update(tables=sorted(tables), columns=sorted(columns),
                      joins=[{"left": a, "right": b, "reason": "independent SQL draft"}
                             for a, b in sorted(joins)])
    except Exception as exc:
        output["diagnostics"].append(f"parse: {type(exc).__name__}: {exc}")
    return output
