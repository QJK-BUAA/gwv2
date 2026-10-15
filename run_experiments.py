#!/usr/bin/env python3
"""Run the frozen paper matrix after reproduce.py has prepared data/evaluators."""
import argparse
from pathlib import Path
from gws import config
from gws.experiments import run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["both", "bird", "spider200", "spider1"], default="both")
    parser.add_argument("--suite", choices=["all", "main", "relations", "policies", "mechanisms",
                        "commit", "perturbations", "tokens", "aligned", "thresholds"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--outdir", default=str(Path(config.PROJECT_ROOT)/"outputs"/"paper"))
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--experiments", nargs="+", help="Only these named configurations")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for dataset in ["bird", "spider200"] if args.dataset == "both" else [args.dataset]:
        run_suite(dataset, args.outdir, args.suite, args.limit, args.sample_seed,
                  args.model, args.no_eval, experiment_names=args.experiments)


if __name__ == "__main__":
    main()
