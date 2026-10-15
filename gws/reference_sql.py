"""Evaluation-only physical-object extractor, independent of the agent parser.

Derived-table outputs are recursively traced through their defining expressions.
This module is never imported by Ground, Explore, Commit, or baseline linkers.
"""
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope


def extract_reference(sql, schema):
    result = {"tables": [], "columns": [], "edges": [], "diagnostics": [], "status": "error"}
    try:
        trees = sqlglot.parse(sql, read="sqlite")
        if len(trees) != 1 or not isinstance(trees[0], exp.Query):
            raise ValueError("Expected a single reference query")
        tree = qualify(trees[0], dialect="sqlite",
            schema={t: {c.name: c.type for c in cs} for t, cs in schema.tables.items()},
            infer_schema=False, validate_qualify_columns=False)
        scopes = list(traverse_scope(tree))
        valid = {f"{t}.{c.name}".lower(): f"{t}.{c.name}".lower()
                 for t, cs in schema.tables.items() for c in cs}
        names = {t.lower(): t.lower() for t in schema.tables}
        tables, columns, edges, active = set(), set(), set(), set()

        def resolve(scope, col):
            if not isinstance(col, exp.Column) or col.is_star:
                return set()
            marker = (id(scope), col.table.lower(), col.name.lower())
            if marker in active:
                return set()
            active.add(marker)
            try:
                cursor = scope
                while cursor:
                    sources = {k.lower(): v for k, (_, v) in cursor.selected_sources.items()}
                    matches = ([sources[col.table.lower()]] if col.table.lower() in sources
                               else list(sources.values()) if not col.table else [])
                    found = set()
                    for source in matches:
                        if isinstance(source, exp.Table):
                            key = f"{source.name}.{col.name}".lower()
                            if key in valid:
                                found.add(valid[key])
                        elif isinstance(source, Scope):
                            for projected in source.expression.selects:
                                if projected.alias_or_name.lower() == col.name.lower():
                                    for nested in projected.find_all(exp.Column):
                                        found.update(resolve(source, nested))
                    if matches:
                        if found:
                            return found
                        # Aggregate output aliases have no physical source column.
                        if any(isinstance(x, Scope) for x in matches):
                            return set()
                        result["diagnostics"].append(f"unresolved column: {col.sql()}")
                        return set()
                    cursor = cursor.parent
                result["diagnostics"].append(f"unresolved column: {col.sql()}")
                return set()
            finally:
                active.remove(marker)

        def add_edges(left, right):
            for a in left:
                for b in right:
                    # Include self joins after physical-name normalization.
                    edges.add(tuple(sorted((a, b))))

        for scope in scopes:
            for _, source in scope.selected_sources.values():
                if isinstance(source, exp.Table):
                    name = source.name.lower()
                    if name in names:
                        tables.add(names[name])
                    else:
                        result["diagnostics"].append(f"unknown table: {source.name}")
            for col in scope.columns:
                columns.update(resolve(scope, col))
            for node in scope.walk():
                if isinstance(node, exp.EQ) and isinstance(node.left, exp.Column) and isinstance(node.right, exp.Column):
                    left, right = resolve(scope, node.left), resolve(scope, node.right)
                    if node.left.table.lower() != node.right.table.lower():
                        add_edges(left, right)
                if isinstance(node, exp.In) and isinstance(node.this, exp.Column) and node.args.get("query"):
                    query = node.args["query"]
                    nested = next((s for s in scopes if s.expression is query.this
                                   or s.expression is query), None)
                    if nested and len(nested.expression.selects) == 1:
                        right = set()
                        for col in nested.expression.selects[0].find_all(exp.Column):
                            right.update(resolve(nested, col))
                        add_edges(resolve(scope, node.this), right)
        owners = {f"{t}.{c.name}".lower(): t.lower() for t, cs in schema.tables.items() for c in cs}
        # COUNT(*) selects a representative entity key, never all table columns.
        for table in tables - {owners[c] for c in columns}:
            cs = next(cs for t, cs in schema.tables.items() if t.lower() == table)
            if cs:
                c = next((c for c in cs if c.pk), cs[0])
                columns.add(f"{table}.{c.name}".lower())
        result.update(tables=sorted(tables), columns=sorted(columns), edges=sorted(edges),
                      status="ok" if not result["diagnostics"] else "undecidable")
    except Exception as exc:
        result["diagnostics"].append(f"{type(exc).__name__}: {exc}")
    return result
