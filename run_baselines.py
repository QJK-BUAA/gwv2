#!/usr/bin/env python3
"""Baseline runner (paper Table: End-to-End EX).

Runs one locally adapted baseline over a dataset on the shared backend and
scores it with the official evaluator -- the same datasets, model, and EX
evaluator GraphWalkerSQL uses.

Examples
--------
python run_baselines.py --method ta-sql --dataset bird --limit 5
python run_baselines.py --method full-schema --dataset spider200
python run_baselines.py --method all --dataset bird --limit 50   # every baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import BASELINES  # noqa: E402
from gws import config  # noqa: E402
from gws.datasets import load_and_sample  # noqa: E402
from gws.evaluate_official import evaluate  # noqa: E402
from gws.llm import LLM  # noqa: E402
from gws.pipeline import load_db  # noqa: E402
from gws.response_cache import ResponseCache  # noqa: E402


def run_one(method: str, dataset: str, args) -> dict:
    examples = load_and_sample(dataset, args.limit, seed=args.seed)
    fn = BASELINES[method]
    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{dataset}_{method}_{args.model}_{args.tag}_seed{args.seed}"
    cache = ResponseCache(os.path.join(args.outdir, f"{stem}.cache.jsonl"))

    print(f"[baseline] method={method} dataset={dataset} n={len(examples)} "
          f"model={args.model}")
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def work(ex):
        llm = LLM(model=args.model, seed=args.seed, response_cache=cache)
        schema = load_db(ex.sqlite_path)
        try:
            out = fn(llm, schema, ex.question, ex.evidence, ex.sqlite_path)
        except Exception as exc:  # noqa: BLE001
            out = {"pred_sql": "SELECT 1", "model_calls": llm.num_calls,
                   "reads": 0, "error": f"{type(exc).__name__}: {exc}"}
        rec = {"instance_id": ex.instance_id, "db_id": ex.db_id,
               "pred_sql": out.get("pred_sql", "SELECT 1"),
               "model_calls": out.get("model_calls"), "reads": out.get("reads"),
               "error": out.get("error", ""), "tokens": llm.stats()}
        with lock:
            results[ex.instance_id] = rec
            done = len(results)
        print(f"  [{done}/{len(examples)}] {ex.instance_id} "
              f"calls={rec['model_calls']} reads={rec['reads']}"
              f"{' ERR:'+rec['error'] if rec['error'] else ''}")
        return rec

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, examples))
    else:
        for ex in examples:
            work(ex)

    pred_sql = {iid: r["pred_sql"] for iid, r in results.items()}
    with open(os.path.join(args.outdir, f"{stem}.records.json"), "w") as f:
        json.dump({"config": vars(args), "method": method, "records": results},
                  f, indent=2)

    summary = {"method": method, "dataset": dataset}
    if not args.no_eval:
        score = evaluate(dataset, examples, pred_sql,
                         os.path.join(args.outdir, f"{stem}.eval"))
        summary.update(ex=score.get("ex"), correct=score.get("correct"),
                       n=score.get("n"))
        with open(os.path.join(args.outdir, f"{stem}.summary.json"), "w") as f:
            json.dump({**summary, "per_question": score.get("per_question", [])},
                      f, indent=2)
        print(f"[baseline] {method} EX = {score.get('ex')}%  "
              f"({score.get('correct')}/{score.get('n')})")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="GraphWalkerSQL baseline runner")
    ap.add_argument("--method", default="full-schema",
                    choices=list(BASELINES) + ["all"])
    ap.add_argument("--dataset", choices=["bird", "spider1", "spider200"],
                    default="bird")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "baselines"))
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    methods = list(BASELINES) if args.method == "all" else [args.method]
    summaries = [run_one(m, args.dataset, args) for m in methods]
    print("\n=== baseline summary ===")
    for s in sorted(summaries, key=lambda x: -(x.get("ex") or 0)):
        print(f"  {s['method']:16s} EX={s.get('ex')}%")


if __name__ == "__main__":
    main()
