# Release verification

Verified on macOS with Python 3.9 in a clean source export, a newly created
virtual environment, and no private workspace imports.

`python3 reproduce.py --prepare-only` installed the pinned dependencies,
verified/extracted the public benchmark archive cache, downloaded the official
evaluators and evaluation-only BIRD annotations, and validated:

- BIRD development: 1,534 questions, 11 databases, including the nested database archive.
- Spider development: 1,034 questions, 20 development databases.
- Spider-200 reproduction: 200 explicit IDs matched against the source-file hash.

The cached data archives matched the current public BIRD object's MD5 and
the pinned Spider mirror's SHA-256. The downloader also fetched the real
evaluator/annotation sources and verified their pinned SHA-256 hashes. The
large-archive network transfer was not completed from an empty cache during
this verification; archive extraction and validation were performed afresh.

`.venv/bin/python -m unittest discover -s tests -q` passed all **19 tests**,
including actual official BIRD/Spider scoring, incorrect/non-submission
checks, all 68 configurations on a real temporary SQLite database,
frozen artifacts/trajectories, resume without model calls, all eight baseline
procedures, aligned budgets, malformed probability models, safety, timeout,
WAL snapshots, cache/token accounting, and upstream-failure accounting.
The standalone offline checks passed **35/35**; compilation and
`git diff --check` also passed.

Live API verification used `deepseek-v4-flash`, temperature 0, seed 42,
4096 output tokens, and disabled thinking. The provider returned
`deepseek-flash`. Each run below used **one question** (`--limit 1`), with
official EX scoring. These sample scores only establish the functioning of
the pipeline and comparator.

| BIRD question `bird_81` | Calls | DB attempts | EX |
|---|---:|---:|---:|
| GraphWalkerSQL adaptive | 6 | 4 | 1 |
| Full-schema | 1 | 0 | 0 |
| DIN-SQL | 2 | 0 | 0 |
| C3-SQL | 3 | 0 | 0 |
| TA-SQL | 2 | 0 | 0 |
| RSL-SQL | 3 | 0 | 0 |
| SchemaGraphSQL | 2 | 0 | 0 |
| PV-SQL | 3 | 1 | 0 |
| APEX-SQL | 21 | 34 | 0 |
| Frozen downstream-belief removal | 6 | 4 | 0 |
| Implication filtering | 6 | 4 | 1 |
| Ordinary Commit + trimmed catalog | 6 | 4 | 0 |
| Resource-aligned TA | 4 | 1 | 0 |
| Resource-aligned APEX | 5 | 0 | 1 |

All 14 BIRD configurations produced valid single-query submissions. Main
APEX keeps its released exploration procedure; only the separately labelled
aligned controls have the six-call/five-read limit.

| Spider question `spider1_502` | Calls | DB attempts | EX |
|---|---:|---:|---:|
| GraphWalkerSQL adaptive | 6 | 3 | 1 |
| Full-schema | 1 | 0 | 1 |
| TA-SQL | 2 | 0 | 1 |
| No active probes | 6 | 1 | 1 |
| Frozen downstream-belief removal | 6 | 3 | 1 |
| Ordinary Commit + trimmed catalog | 6 | 3 | 1 |

All six Spider configurations produced valid submissions. Both benchmark
adaptive traces contain completed probes and Bayesian updates. Frozen
downstream-belief removal shares exactly the adaptive upstream and trajectory
hashes. The final verification reused actual provider responses from the
request cache where inputs were identical.

This verifies method functionality and automated preparation/run behavior.
The full benchmark matrix has not been rerun on the live API. The original
Spider-200 ID list and historical path extractor remain unavailable, as
documented in `PAPER_ALIGNMENT.md`; the paper's headline scores are not
claimed as newly reproduced results.
