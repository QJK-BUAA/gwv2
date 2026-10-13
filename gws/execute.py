"""SQLite execution helpers (read-only, watchdog-timed).

Used both for lightweight join *probes* during Explore and for the final SQL
execution during Commit. All connections are opened read-only; a watchdog timer
interrupts runaway queries (e.g. accidental cross joins on large tables).
"""
from __future__ import annotations

import sqlite3
import threading

from . import config


def run_query(sqlite_path: str, sql: str, max_rows: int = config.EXEC_MAX_ROWS,
              timeout: float = config.EXEC_TIMEOUT,
              params: dict | tuple | None = None) -> dict:
    """Execute ``sql`` read-only. Returns a structured result dict.

    ok=True  -> {"ok": True, "columns": [...], "rows": [...], "n_shown": int}
    ok=False -> {"ok": False, "error": "..."}
    """
    con = None
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        timer = threading.Timer(timeout, con.interrupt)
        timer.start()
        try:
            cur = con.cursor()
            cur.execute(sql, params or ())
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
        finally:
            timer.cancel()
        return {"ok": True, "columns": cols, "rows": rows, "n_shown": len(rows)}
    except Exception as e:  # noqa: BLE001
        message = str(e)
        lowered = message.lower()
        if "interrupted" in lowered:
            error_type = "timeout"
        elif any(token in lowered for token in ("readonly", "permission", "authorized")):
            error_type = "permission_denied"
        else:
            error_type = "error"
        return {"ok": False, "error": message, "error_type": error_type}
    finally:
        if con is not None:
            con.close()


def scalar(sqlite_path: str, sql: str, timeout: float = 10.0):
    """Return the first cell of the first row, or None on error/empty."""
    res = run_query(sqlite_path, sql, max_rows=1, timeout=timeout)
    if res.get("ok") and res.get("rows"):
        return res["rows"][0][0]
    return None


def result_signature(sqlite_path: str, sql: str, cap: int = 200,
                     timeout: float = config.EXEC_TIMEOUT) -> str | None:
    """Order-insensitive multiset signature of a query's result set.

    Returns None if the query fails, so callers can distinguish "ran and empty"
    from "did not run".
    """
    res = run_query(sqlite_path, sql, max_rows=cap, timeout=timeout)
    if not res.get("ok"):
        return None
    norm = sorted(tuple(str(x) for x in r) for r in res["rows"])
    return repr(norm)
