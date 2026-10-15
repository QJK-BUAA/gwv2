"""Portable read-only connections for static benchmark snapshots."""
from pathlib import Path
from urllib.parse import unquote, urlsplit


def read_only_uri(database):
    raw = str(database)
    if raw.startswith("file:"):
        raw = unquote(urlsplit(raw).path)
    path = Path(raw).resolve()
    # Published snapshots can have WAL-mode headers without WAL/SHM sidecars.
    # immutable=1 reads their checkpointed database without creating sidecars.
    # A real WAL sidecar must still be read by SQLite's normal WAL machinery.
    immutable = "&immutable=1" if not Path(str(path)+"-wal").exists() else ""
    return path.as_uri()+"?mode=ro"+immutable
