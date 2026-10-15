"""Execute compiled probes against a read-only SQLite database (Explore stage).

A probe runs with a short timeout (5s per Appendix Tab. 1). A valid observation
must be successful, complete, and contain a Boolean row; failures and timeouts
are recorded as *unavailable* observations (status error/timeout) that still
consume one active attempt but leave the posterior unchanged.
"""
from __future__ import annotations

import sqlite3
import threading
from .execute import run_query

from . import config
from .probes import compile_probe
from .types import Probe, ProbeOutcome


def run_probe(sqlite_path: str, probe: Probe,
              timeout: float = config.PROBE_TIMEOUT, budget=None) -> ProbeOutcome:
    try:
        sql, params = compile_probe(probe)
    except ValueError as exc:
        return ProbeOutcome(probe_id=probe.id, kind=probe.kind,
                            status="error", available=False, sql=str(exc))

    result = run_query(sqlite_path, sql, max_rows=1, timeout=timeout, params=params,
                       budget=budget, stage="active_probe")
    rows = result.get("rows", [])
    available = bool(result.get("ok") and result.get("complete") and rows
                     and len(rows[0]) == 1 and rows[0][0] in (0, 1))
    status = ("hit" if rows[0][0] else "miss") if available else result.get("status", "error")
    return ProbeOutcome(probe.id, probe.kind, status, available, sql, params,
                        result.get("error", ""))
