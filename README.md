# gwv2 — GraphWalkerSQL Reproduction

A faithful, self-contained reproduction of **GraphWalkerSQL: Schema-Graph Walks
for Relational Grounding in Text-to-SQL**, plus the paper's eight comparison
baselines run on the same backend and official evaluators.

GraphWalkerSQL is a training-free **Ground → Explore → Commit** pipeline with
one-step **EVSI**-guided, read-only database probing:

- **Ground** — independent requirement analysis + SQL draft, then a candidate
  relation graph (structural + visible FKs + model-proposed) and bounded path search.
- **Explore** — a decision model proposes probes; selection ranks by one-step EVSI
  (0–1 utility) and stops when `EVSI ≤ λ·c(a)`. Four `SELECT EXISTS` probe templates.
- **Commit** — reconcile + coverage-audit produce a per-requirement coverage
  contract and a compact schema; one final SQL is generated (`K=1`).

Six logical model calls per question; probes do not add to the call count.

## Layout

```
gws/         the GraphWalkerSQL method (pipeline, EVSI/belief, probes, commit)
baselines/   8 adapted baselines: full-schema, DIN/C3/TA/RSL-SQL,
             SchemaGraphSQL, PV-SQL, APEX-SQL
run.py              GraphWalkerSQL runner (+ official EX)
run_baselines.py    baseline runner (+ official EX)
tests/              offline unit + mock E2E tests (no API)
```

## Setup

Python 3.9+, `openai` + `sqlglot`. Set an OpenAI-compatible key:

```bash
export DS_API_KEY="sk-..."            # or DEEPSEEK_API_KEY / OPENAI_API_KEY
export GWV2_MODEL="deepseek-v4-flash" # optional backend overrides
```

## Verify (no API cost)

```bash
python3 tests/test_offline.py        # deterministic unit tests
python3 tests/test_e2e_mock.py       # full pipeline on a real DB (mock LLM)
python3 tests/test_baselines_mock.py # all 8 baselines (mock LLM)
```

## Run

```bash
# GraphWalkerSQL headline settings
python3 run.py --dataset bird      --policy adaptive
python3 run.py --dataset spider200 --policy adaptive

# Probe-policy study / ablations
python3 run.py --dataset bird --policy eig          # or voi / random / no-probe
python3 run.py --dataset bird --no-draft            # relation / commit / belief ablations

# Baselines
python3 run_baselines.py --method all --dataset bird
```

Results and a response cache are written to `outputs/`. EX is always computed by
the official BIRD comparator and the Spider test-suite evaluator — never a
bespoke matcher. Reported numbers depend on the live model backend; the method,
not any hard-coded score, is what this repo reproduces.
