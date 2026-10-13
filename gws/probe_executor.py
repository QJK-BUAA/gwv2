"""Execute compiled probes against a read-only SQLite database (Explore stage).

A probe runs with a short timeout (5s per Appendix Tab. 1). A valid observation
must be successful, complete, and contain a Boolean row; failures and timeouts
are recorded as *unavailable* observations (status error/timeout) that still
consume one active attempt but leave the posterior unchanged.
"""
from __future__ import annotations

import sqlite3
import threading

from . import config
from .probes import compile_probe
from .types import Probe, ProbeOutcome


def run_probe(sqlite_path: str, probe: Probe,
              timeout: float = config.PROBE_TIMEOUT) -> ProbeOutcome:
    try:
        sql, params = compile_probe(probe)
    except ValueError as exc:
        return ProbeOutcome(probe_id=probe.id, kind=probe.kind,
                            status="error", available=False, sql=str(exc))

    con = None
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        timer = threading.Timer(timeout, con.interrupt)
        timer.start()
        try:
            cur = con.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
        finally:
            timer.cancel()
        if row is None or row[0] is None:
            return ProbeOutcome(probe_id=probe.id, kind=probe.kind,
                                status="error", available=False, sql=sql)
        hit = bool(row[0])
        return ProbeOutcome(probe_id=probe.id, kind=probe.kind,
                            status="hit" if hit else "miss",
                            available=True, sql=sql)
    except Exception as exc:  # noqa: BLE001
        status = "timeout" if "interrupt" in str(exc).lower() else "error"
        return ProbeOutcome(probe_id=probe.id, kind=probe.kind,
                            status=status, available=False, sql=sql)
    finally:
        if con is not None:
            con.close()
