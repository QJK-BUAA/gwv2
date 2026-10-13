"""Dataset loaders for the GraphWalkerSQL reproduction.

Two benchmarks reported in the paper, reduced to a common ``Example``:
  * BIRD-Dev      -- declared FKs; 1,534 questions / 11 databases; difficulty labels.
  * Spider 1.0-Dev -- declared FKs; 1,034 questions / 166 databases.
  * Spider-200    -- the paper's *fixed hash-selected* 200-question Spider-Dev
                     subset. Selection is a deterministic content hash (stable
                     across machines and runs), not a random draw, matching the
                     "fixed hash-selected" description in the paper's Setup.

``evidence`` carries BIRD's external knowledge (``e`` in the paper); Spider has
none. ``idx`` is the original source-file index, kept for gold alignment in the
official evaluators.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from . import config


@dataclass
class Example:
    idx: int                 # original index in the source file (for gold alignment)
    instance_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    sqlite_path: str
    difficulty: str = ""
    dialect: str = "SQLite"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_bird() -> list[Example]:
    data = json.load(open(config.BIRD_DEV_JSON))
    out: list[Example] = []
    for i, ex in enumerate(data):
        db = ex["db_id"]
        sp = os.path.join(config.BIRD_DB_DIR, db, f"{db}.sqlite")
        if not os.path.exists(sp):
            continue
        out.append(Example(
            idx=i, instance_id=f"bird_{ex['question_id']}", db_id=db,
            question=ex["question"], evidence=ex.get("evidence", ""),
            gold_sql=ex["SQL"], sqlite_path=sp,
            difficulty=ex.get("difficulty", ""),
        ))
    return out


def load_spider1() -> list[Example]:
    data = json.load(open(config.SPIDER1_DEV_JSON))
    gold_lines = [l.strip() for l in open(config.SPIDER1_DEV_GOLD) if l.strip()]
    out: list[Example] = []
    for i, ex in enumerate(data):
        db = ex["db_id"]
        sp = os.path.join(config.SPIDER1_DB_DIR, db, f"{db}.sqlite")
        if not os.path.exists(sp):
            continue
        gold = ex.get("query", "")
        # dev_gold.sql lines are "SQL\tdb_id"; keep them index-aligned for eval.
        if i < len(gold_lines) and "\t" in gold_lines[i]:
            gold = gold_lines[i].rsplit("\t", 1)[0]
        out.append(Example(
            idx=i, instance_id=f"spider1_{i}", db_id=db,
            question=ex["question"], evidence="", gold_sql=gold, sqlite_path=sp,
        ))
    return out


def load_spider200(
    size: int = config.SPIDER200_SIZE,
    salt: str = config.SPIDER200_SALT,
) -> list[Example]:
    """The paper's fixed, hash-selected 200-question Spider-Dev subset.

    Each example is scored by a SHA-256 hash of ``salt + db_id + question``. The
    lowest ``size`` hashes are kept, then re-sorted by original index for stable
    evaluator alignment. This is reproducible on any machine and never touches a
    PRNG, so the subset is literally "fixed" rather than seed-dependent.
    """
    examples = load_spider1()

    def _key(e: Example) -> str:
        payload = f"{salt}|{e.db_id}|{e.question.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    chosen = sorted(examples, key=_key)[: min(size, len(examples))]
    return sorted(chosen, key=lambda e: e.idx)


LOADERS = {
    "bird": load_bird,
    "spider1": load_spider1,
    "spider200": load_spider200,
}


# --------------------------------------------------------------------------- #
# Deterministic head/stratified subsampling (for smoke runs and ablations)
# --------------------------------------------------------------------------- #
def sample(examples: list[Example], limit: int | None,
           seed: int = config.DEFAULT_SEED) -> list[Example]:
    """Reproducible DB-interleaved subsample sorted by original index.

    Used for small development/ablation runs. The headline Spider result uses
    ``load_spider200`` instead, and the headline BIRD result uses the full set
    (``limit=None``).
    """
    if not limit or limit >= len(examples):
        return sorted(examples, key=lambda e: e.idx)
    import random
    rng = random.Random(seed)
    by_db: dict[str, list[Example]] = {}
    for e in examples:
        by_db.setdefault(e.db_id, []).append(e)
    for db in by_db:
        rng.shuffle(by_db[db])
    order = sorted(by_db.keys())
    picked: list[Example] = []
    while len(picked) < limit and any(by_db[db] for db in order):
        for db in order:
            if by_db[db]:
                picked.append(by_db[db].pop())
                if len(picked) >= limit:
                    break
    return sorted(picked, key=lambda e: e.idx)


def load_and_sample(dataset: str, limit: int | None,
                    seed: int = config.DEFAULT_SEED) -> list[Example]:
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(LOADERS)}")
    examples = LOADERS[dataset]()
    # spider200 is already a fixed subset; only further-trim if asked.
    if dataset == "spider200" and (not limit or limit >= len(examples)):
        return examples
    return sample(examples, limit, seed=seed)
