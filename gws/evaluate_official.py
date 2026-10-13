"""Evaluation harness wrapping the OFFICIAL benchmark evaluators.

Design principle (project rule): report execution accuracy (EX) using the
official scripts, never a bespoke approximate matcher.

  BIRD      -> 论文复现/.../evaluation/bird_evaluation_raw.py
               (official execute-both + round-6dp + set-equality per question)
  Spider1   -> .../test-suite-sql-eval/exec_eval.py  (execution match)
  Spider200 -> same Spider1 evaluator (the fixed 200-question subset reuses
               spider1_* instance ids and the Spider dev databases)

Gold SQL is carried on each ``Example`` and aligned by instance id, so partial
(smoke) runs score correctly without rebuilding whole-file gold lists.
"""
from __future__ import annotations

import importlib.util
import os
import sys

from . import config
from .datasets import Example


# --------------------------------------------------------------------------- #
# BIRD
# --------------------------------------------------------------------------- #
def evaluate_bird(examples: list[Example], pred_sql: dict[str, str],
                  workdir: str) -> dict:
    """Score with BIRD's OFFICIAL per-question comparator.

    We import ``execute_model`` from the official ``bird_evaluation_raw.py`` and
    call it per question (identical execute-both + round-floats-6dp + set-equality
    logic), then aggregate overall EX ourselves. This avoids the official
    ``compute_acc_by_diff`` division-by-zero when a difficulty bucket is empty in
    small samples, while keeping the exact official matching semantics.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    eval_dir = os.path.dirname(config.BIRD_EVAL_SCRIPT)
    shim = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "eval_shims")
    for p in (shim, eval_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("bird_eval_official",
                                                  config.BIRD_EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    per_diff: dict[str, list[int]] = {}
    results = []
    for i, ex in enumerate(examples):
        sql = (pred_sql.get(ex.instance_id) or "SELECT 1").replace("\n", " ").strip()
        r = mod.execute_model(sql, ex.gold_sql.strip(), ex.sqlite_path, i,
                              config.EXEC_TIMEOUT)
        res = int(r.get("res", 0))
        results.append({"instance_id": ex.instance_id, "res": res})
        per_diff.setdefault(ex.difficulty or "simple", []).append(res)

    n = len(examples)
    ex_overall = 100.0 * sum(x["res"] for x in results) / n if n else None
    out = {"n": n, "ex": round(ex_overall, 2) if ex_overall is not None else None,
           "correct": sum(x["res"] for x in results), "per_question": results}
    for d, xs in per_diff.items():
        out[d] = round(100.0 * sum(xs) / len(xs), 2)
    return out


# --------------------------------------------------------------------------- #
# Spider 1.0 / Spider-200 (test-suite evaluator, execution mode)
# --------------------------------------------------------------------------- #
def evaluate_spider1(examples: list[Example], pred_sql: dict[str, str],
                     workdir: str) -> dict:
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    exec_match = _load_spider1_exec_match()
    results = []
    for ex in examples:
        sql = (pred_sql.get(ex.instance_id) or "SELECT 1").replace("\n", " ").strip()
        score = int(exec_match(
            db=ex.sqlite_path,
            p_str=sql,
            g_str=ex.gold_sql.strip(),
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        ))
        results.append({"instance_id": ex.instance_id, "res": score})
    correct = sum(item["res"] for item in results)
    return {
        "n": len(examples),
        "ex": round(100.0 * correct / len(examples), 2) if examples else None,
        "correct": correct,
        "per_question": results,
    }


_SPIDER1_EXEC_MATCH = None


def _load_spider1_exec_match():
    global _SPIDER1_EXEC_MATCH
    if _SPIDER1_EXEC_MATCH is not None:
        return _SPIDER1_EXEC_MATCH
    eval_dir = os.path.abspath(config.SPIDER1_TESTSUITE)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    path = os.path.join(eval_dir, "exec_eval.py")
    spec = importlib.util.spec_from_file_location("spider1_exec_eval_official", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Spider1 evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SPIDER1_EXEC_MATCH = module.eval_exec_match
    return _SPIDER1_EXEC_MATCH


EVALUATORS = {
    "bird": evaluate_bird,
    "spider1": evaluate_spider1,
    "spider200": evaluate_spider1,
}


def evaluate(dataset: str, examples: list[Example], pred_sql: dict[str, str],
             workdir: str) -> dict:
    if dataset not in EVALUATORS:
        raise ValueError(f"no official evaluator for dataset {dataset!r}")
    return EVALUATORS[dataset](examples, pred_sql, workdir)
