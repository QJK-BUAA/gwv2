# GraphWalkerSQL

Implementation of the paper's **Ground → Explore → Commit** algorithm,
eight shared-backend baselines, and the complete controlled experiment matrix.

Prepare your own **DeepSeek API key**. With Python 3.9+ and internet access,
one command installs dependencies, downloads and verifies benchmarks and official
evaluators, and runs all experiments:

```bash
python3 reproduce.py --api-key YOUR_DEEPSEEK_API_KEY
```

Alternatively set `DEEPSEEK_API_KEY` and run `python3 reproduce.py`.
The default is the paper's `deepseek-v4-flash` model, temperature 0, seed 42,
4096 output tokens. Use `--model` / `--base-url` for another compatible endpoint.

```bash
# Small end-to-end check of GraphWalkerSQL on both benchmarks
python3 reproduce.py --api-key YOUR_DEEPSEEK_API_KEY --limit 2 --suite main --experiments adaptive

# Data/environment preparation only; no API calls
python3 reproduce.py --prepare-only

# Offline verification after preparation
.venv/bin/python -m unittest discover -s tests -q
```

Full runs incur API charges and can take substantial time. Results, traces,
official EX scores, paired statistics, and resumable caches are saved in
`outputs/paper/`; data lives in `.data/`. Use `--dataset`, `--suite`, `--limit`,
`--data-dir`, or `--outdir` to adjust the run.

See [paper-to-code mapping](PAPER_ALIGNMENT.md), [source provenance](SOURCES.md),
and [verification](LIVE_VERIFICATION.md). The original Spider-200 IDs and
historical path-scoring implementation were not archived; the explicit
[reproduction cohort](manifests/spider200.json) and path rule are documented.
Exact historical scores are not claimed.
