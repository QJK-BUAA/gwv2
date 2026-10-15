#!/usr/bin/env python3
"""GraphWalkerSQL reproduction runner.

Examples
--------
# BIRD smoke run (5 questions), adaptive policy, official EX:
python run.py --dataset bird --limit 5 --policy adaptive

# Full BIRD Dev (1,534 q) headline result:
python run.py --dataset bird --policy adaptive

# Spider-200 (fixed hash-selected subset) headline result:
python run.py --dataset spider200 --policy adaptive

# Probe-policy ablation (Table: probe ranking and stopping):
python run.py --dataset bird --limit 150 --policy eig
python run.py --dataset bird --limit 150 --policy no-probe

The runner is deterministic given a fixed model backend: temperature 0, request
seed 42, and an on-disk response cache so re-runs reuse identical model outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gws import config  # noqa: E402
from gws.datasets import load_and_sample  # noqa: E402
from gws.evaluate_official import evaluate  # noqa: E402
from gws.llm import LLM  # noqa: E402
from gws.pipeline import AblationConfig, run_pipeline  # noqa: E402
from gws.response_cache import ResponseCache  # noqa: E402


def build_ablation(args) -> AblationConfig:
    return AblationConfig(
        probe_policy=args.policy,
        budget=args.budget,
        lam=args.lam,
        seed=args.probe_seed,
        include_draft=not args.no_draft,
        include_structural=not args.no_structural,
        include_model=not args.no_model,
        use_value_overlap=not args.no_overlap,
        include_beliefs=not args.no_beliefs,
        coverage_contract=not args.ordinary_commit,
        collect_stats=args.collect_stats,
        implication_filter=args.implication_filter,
        catalog_mode=args.catalog,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="GraphWalkerSQL reproduction runner")
    ap.add_argument("--dataset", choices=["bird", "spider1", "spider200"],
                    default="bird")
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample size (default: full set)")
    ap.add_argument("--policy", default="adaptive",
                    choices=["adaptive", "voi", "eig", "random", "no-probe"])
    ap.add_argument("--budget", type=int, default=config.ACTIVE_PROBE_BUDGET)
    ap.add_argument("--lam", type=float, default=config.LAMBDA_COST)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--probe-seed", type=int, default=42)
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"))
    # ablation toggles
    ap.add_argument("--no-draft", action="store_true")
    ap.add_argument("--no-structural", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--no-beliefs", action="store_true")
    ap.add_argument("--ordinary-commit", action="store_true")
    ap.add_argument("--collect-stats", action="store_true")
    ap.add_argument("--implication-filter", action="store_true")
    ap.add_argument("--catalog", choices=["full", "trimmed"], default="full")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip official EX scoring (predictions only)")
    args = ap.parse_args()

    examples = load_and_sample(args.dataset, args.limit, seed=args.seed)
    ablation = build_ablation(args)
    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.dataset}_{ablation.label()}_{args.model}_{args.tag}_seed{args.seed}"
    cache_path = os.path.join(args.outdir, f"{stem}.cache.jsonl")
    cache = ResponseCache(cache_path)

    print(f"[gwv2] dataset={args.dataset} n={len(examples)} "
          f"policy={args.policy} model={args.model} label={ablation.label()}")

    results: dict[str, dict] = {}
    lock = threading.Lock()

    def work(ex):
        # One LLM client per question so num_calls reflects the 6-call budget;
        # the shared cache keeps outputs identical across threads/re-runs.
        llm = LLM(model=args.model, seed=42, response_cache=cache)
        res = run_pipeline(llm, ex.instance_id, ex.db_id, ex.question,
                           ex.evidence, ex.sqlite_path, ablation)
        rec = asdict(res)
        with lock:
            results[ex.instance_id] = rec
            done = len(results)
        print(f"  [{done}/{len(examples)}] {ex.instance_id} "
              f"calls={res.model_calls} probes={res.probe_attempts} "
              f"stop={res.stop_reason}{' ERR:'+res.error if res.error else ''}")
        return rec

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, examples))
    else:
        for ex in examples:
            work(ex)

    pred_sql = {iid: r["pred_sql"] for iid, r in results.items()}
    out_records = os.path.join(args.outdir, f"{stem}.records.json")
    with open(out_records, "w") as f:
        json.dump({"config": vars(args), "records": results}, f, indent=2)
    print(f"[gwv2] wrote {out_records}")

    total_probes = sum(r["probe_attempts"] for r in results.values())
    mean_probes = total_probes / len(results) if results else 0.0
    print(f"[gwv2] mean active probes/q = {mean_probes:.2f}")

    if not args.no_eval:
        score = evaluate(args.dataset, examples, pred_sql,
                         os.path.join(args.outdir, f"{stem}.eval"),
                         statuses={i: r["status"] for i, r in results.items()})
        summary = {"ex": score.get("ex"), "correct": score.get("correct"),
                   "n": score.get("n"), "mean_probes": round(mean_probes, 2)}
        out_summary = os.path.join(args.outdir, f"{stem}.summary.json")
        with open(out_summary, "w") as f:
            json.dump({**summary, "per_question": score.get("per_question", [])},
                      f, indent=2)
        print(f"[gwv2] EX = {score.get('ex')}%  "
              f"({score.get('correct')}/{score.get('n')})  -> {out_summary}")


if __name__ == "__main__":
    main()
