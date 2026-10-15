"""Verified public benchmark preparation (standard library only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import urllib.request
import zipfile
from . import config


ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT/"data_sources.json"


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = source["sha256"]
    if target.is_file() and sha256_file(target) == expected:
        return target
    partial = target.with_name(target.name+".part")
    print(f"Downloading {source['name']}...", flush=True)
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "GraphWalkerSQL-reproduction/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(source["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset and response.status == 206
                with partial.open("ab" if append else "wb") as stream:
                    last_report = time.monotonic()
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        stream.write(block)
                        if time.monotonic()-last_report >= 15:
                            print(f"  {source['name']}: {stream.tell()//(1024*1024)} MiB", flush=True)
                            last_report = time.monotonic()
            if sha256_file(partial) != expected:
                partial.unlink()
                raise ValueError(f"Checksum mismatch for {source['name']}")
            os.replace(partial, target)
            return target
        except (OSError, ValueError) as exc:
            if attempt == 3:
                raise RuntimeError(f"Cannot prepare {source['name']} ({type(exc).__name__})") from None
            print(f"  Retrying {source['name']} ({type(exc).__name__})...", flush=True)
            time.sleep(min(2**attempt, 8))


def extract_archive(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if "__MACOSX" in Path(member.filename).parts or Path(member.filename).name == ".DS_Store":
                continue
            target = (destination/member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive symlink is unsupported")
            bundle.extract(member, destination)


def install_archive(source, archive, parent, canonical_name, marker_name):
    parent = Path(parent)
    target = parent/canonical_name
    stamp = target/".source.sha256"
    if stamp.is_file() and stamp.read_text().strip() == source["sha256"]:
        return target
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        extract_archive(archive, temporary)
        candidates = [p.parent for p in Path(temporary).rglob(marker_name)]
        if len(candidates) != 1:
            raise ValueError(f"Unexpected layout for {source['name']}")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(candidates[0]), target)
    stamp.write_text(source["sha256"]+"\n")
    return target


def validate_dataset(root, dataset):
    root = Path(root)
    data = json.loads((root/"dev.json").read_text())
    expected_questions, expected_databases = (1534, 11) if dataset == "bird" else (1034, 20)
    if len(data) != expected_questions or len({r["db_id"] for r in data}) != expected_databases:
        raise ValueError(f"Unexpected {dataset} development split")
    database = root/("dev_databases" if dataset == "bird" else "database")
    for db in sorted({r["db_id"] for r in data}):
        path = database/db/(db+".sqlite")
        if not path.is_file():
            raise ValueError(f"Missing benchmark database: {db}")
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise ValueError(f"Invalid SQLite database: {db}")
    table_file = root/("dev_tables.json" if dataset == "bird" else "tables.json")
    tables = json.loads(table_file.read_text())
    if not {r["db_id"] for r in data} <= {t["db_id"] for t in tables}:
        raise ValueError("Incomplete table registry")
    if dataset == "spider":
        gold = root/"dev_gold.sql"
        if not gold.exists():
            gold.write_text("".join(r["query"].replace("\n", " ")+ "\t"+r["db_id"]+"\n" for r in data))
        lines = [line for line in gold.read_text().splitlines() if line.strip()]
        if len(lines) != len(data):
            raise ValueError("Gold SQL alignment mismatch")
        for row, line in zip(data, lines):
            sql, db = line.rsplit("\t", 1)
            if db != row["db_id"] or " ".join(sql.split()) != " ".join(row["query"].split()):
                raise ValueError("Gold SQL index mismatch")
        cohort = json.loads((ROOT/"manifests"/"spider200.json").read_text())
        if cohort["dev_sha256"] != sha256_file(root/"dev.json"):
            raise ValueError("Spider-200 manifest is for a different dev split")
        ids = cohort["source_indices"]
        if len(ids) != 200 or len(set(ids)) != 200 or any(i < 0 or i >= len(data) for i in ids):
            raise ValueError("Invalid Spider-200 manifest")
    print(f"Validated {dataset}: {len(data)} questions / {expected_databases} dev databases.", flush=True)
    return {"questions": len(data), "databases": expected_databases,
            "dev_sha256": sha256_file(root/"dev.json"), "tables_sha256": sha256_file(table_file)}


def prepare(dataset="both"):
    sources = json.loads(SOURCES.read_text())
    root = Path(config.DATA_ROOT)
    downloads = root/"downloads"
    report = {"sources": sources, "datasets": {}}
    need_bird = dataset in {"both", "bird"}
    need_spider = dataset in {"both", "spider1", "spider200"}
    if need_bird:
        source = sources["bird"]
        archive = download(source, downloads/"bird_dev.zip")
        data = install_archive(source, archive, root/"bird", "dev_20240627", "dev.json")
        if not (data/"dev_databases").is_dir():
            extract_archive(data/"dev_databases.zip", data)
        report["datasets"]["bird"] = validate_dataset(data, "bird")
        download(sources["bird_evaluator"], Path(config.BIRD_EVAL_SCRIPT))
        download(sources["bird_annotations"], root/"annotations"/"bird_schema.json")
        annotations = json.loads((root/"annotations"/"bird_schema.json").read_text())
        if len(annotations) != 1534:
            raise ValueError("Incomplete BIRD schema annotations")
    if need_spider:
        source = sources["spider"]
        archive = download(source, downloads/"spider_data.zip")
        data = install_archive(source, archive, root/"spider", "spider_data", "dev.json")
        report["datasets"]["spider"] = validate_dataset(data, "spider")
        source = sources["spider_evaluator"]
        archive = download(source, downloads/"spider_evaluator.zip")
        install_archive(source, archive, Path(config.SPIDER1_TESTSUITE).parent,
                        Path(config.SPIDER1_TESTSUITE).name, "exec_eval.py")
    root.mkdir(parents=True, exist_ok=True)
    (root/"preparation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", "bird", "spider1", "spider200"], default="both")
    prepare(parser.parse_args().dataset)
