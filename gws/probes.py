"""The four read-only probe templates (Explore stage).

Each probe compiles to a single ``SELECT EXISTS(...)`` returning one Boolean,
mapped to HIT (1) / MISS (0). Templates (Appendix Tab. "Four probe templates"):

  equals     -- at least one row has column = bound literal.
  nonnull    -- at least one row has a non-null value in the column.
  duplicate  -- grouping non-null values yields a group with count > 1.
  overlap    -- two columns share at least one non-null value (exact equality).

Identifiers are validated and double-quoted; literals are bound as SQL
parameters (never string-formatted into the query). The compiler accepts only
these four kinds and rejects anything else with a diagnostic.
"""
from __future__ import annotations

import re

from .types import Probe

_UNSAFE = re.compile(r"[\x00-\x1f;\"`]|--|/\*|\*/")


def _validate_identifier(identifier: str) -> None:
    if not identifier or _UNSAFE.search(identifier):
        raise ValueError(f"unsafe identifier: {identifier!r}")


def _quote(identifier: str) -> str:
    _validate_identifier(identifier)
    return '"' + identifier.replace('"', '""') + '"'


def compile_probe(probe: Probe) -> tuple[str, tuple]:
    """Return (sql, params) for a probe, or raise ValueError if invalid.

    ``params`` is a tuple bound positionally by sqlite3 (``?`` placeholders).
    """
    kind = probe.kind.lower()
    if kind == "equals":
        if not (probe.table and probe.column) or probe.literal is None:
            raise ValueError("equals probe requires table, column, literal")
        t, c = _quote(probe.table), _quote(probe.column)
        sql = f'SELECT EXISTS(SELECT 1 FROM {t} WHERE {c} = ?) AS hit'
        return sql, (probe.literal,)

    if kind == "nonnull":
        if not (probe.table and probe.column):
            raise ValueError("nonnull probe requires table, column")
        t, c = _quote(probe.table), _quote(probe.column)
        sql = f'SELECT EXISTS(SELECT 1 FROM {t} WHERE {c} IS NOT NULL) AS hit'
        return sql, ()

    if kind == "duplicate":
        if not (probe.table and probe.column):
            raise ValueError("duplicate probe requires table, column")
        t, c = _quote(probe.table), _quote(probe.column)
        sql = (
            f'SELECT EXISTS(SELECT 1 FROM {t} WHERE {c} IS NOT NULL '
            f'GROUP BY {c} HAVING COUNT(*) > 1) AS hit'
        )
        return sql, ()

    if kind == "overlap":
        if not (probe.table and probe.column and probe.table2 and probe.column2):
            raise ValueError("overlap probe requires two table.column pairs")
        t1, c1 = _quote(probe.table), _quote(probe.column)
        t2, c2 = _quote(probe.table2), _quote(probe.column2)
        sql = (
            f'SELECT EXISTS(SELECT 1 FROM {t1} AS a JOIN {t2} AS b '
            f'ON a.{c1} = b.{c2} WHERE a.{c1} IS NOT NULL) AS hit'
        )
        return sql, ()

    raise ValueError(f"unknown probe kind: {probe.kind!r}")


VALID_KINDS = ("equals", "nonnull", "duplicate", "overlap")
