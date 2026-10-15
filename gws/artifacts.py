"""Portable JSON artifacts with validated fingerprints and typed restoration."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import tempfile
from .ground import GroundResult
from .types import Requirement, RelationCandidate, Path as JoinPath, Decision, Interpretation, Probe
from .explore import ExploreTrace


def digest(value):
    payload = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def encode_upstream(upstream):
    return {**upstream, "ground": asdict(upstream["ground"]),
            "decisions": [asdict(d) for d in upstream["decisions"]]}


def decode_upstream(raw):
    g = dict(raw["ground"])
    g["requirements"] = [Requirement(**r) for r in g["requirements"]]
    g["candidates"] = [RelationCandidate(**r) for r in g["candidates"]]
    g["paths"] = [JoinPath(p["tables"], [RelationCandidate(**e) for e in p["edges"]],
                          p["cost"]) for p in g["paths"]]
    decisions = []
    for d in raw["decisions"]:
        item = dict(d)
        item["interpretations"] = [Interpretation(**z) for z in d["interpretations"]]
        item["probes"] = [Probe(**p) for p in d["probes"]]
        decisions.append(Decision(**item))
    return {**raw, "ground": GroundResult(**g), "decisions": decisions}


def source_fingerprint(root):
    root = Path(root)
    paths = sorted(list((root/"gws").glob("*.py")) +
                   list((root/"baselines").rglob("*.py")) +
                   list((root/"manifests").glob("*.json")) +
                   [p for p in (root/"requirements.txt", root/"data_sources.json",
                    root/"vendor_sources.json", root/"reproduce.py", root/"run_experiments.py")
                    if p.is_file()])
    return digest({str(p.relative_to(root)): digest(p.read_bytes()) for p in paths})


def restore_trace(raw):
    return ExploreTrace(**raw)
