# Live Verification Log — gwv2 reproduction

This file records a **real** end-to-end run against the live model backend
(not the mock tests), proving the reproduction actually executes the paper's
pipeline and the baselines on the shared backend + official evaluator.

## Backend confirmed
- Requested model alias `deepseek-v4-flash` resolves to response model
  `deepseek-flash` on the endpoint — exactly the paper's Appendix Table 1 setup.
- The `gws.llm.LLM` wrapper disables reasoning (`thinking=disabled`) for v4
  models, so `deepseek-v4-flash` returns normal (non-empty) completions.
- Settings: temperature 0, request seed 42, 4096-token cap, read-only SQLite.

## GraphWalkerSQL — BIRD Dev, 5-question smoke (policy = adaptive)
Command:
```
python3 run.py --dataset bird --limit 5 --policy adaptive --workers 3 --tag smoke_real
```
Result:
- Every question used **exactly 6 logical model calls** (paper's call budget).
- Live read-only probes fired and updated beliefs; cost-aware stopping triggered
  (`stop=cost_threshold`) and candidate exhaustion (`stop=exhausted`) both observed.
- Example `bird_1100` (european_football_2): 58 candidate edges, 3 bounded paths,
  3 decisions, 2 probes (`duplicate` HIT, `overlap` HIT), cost-threshold stop,
  coherent join-spanning column selection, valid final SQL.
- Official BIRD evaluator **EX = 60.0% (3/5)**, mean active probes/q = 0.40.

## Baselines — same BIRD 5-question subset, shared backend + official EX
```
python3 run_baselines.py --method full-schema --dataset bird --limit 5 --tag smoke_real
python3 run_baselines.py --method ta-sql      --dataset bird --limit 5 --tag smoke_real
```
- full-schema: 1 model call/q, **EX = 60.0% (3/5)**.
- ta-sql: 2 model calls/q, **EX = 40.0% (2/5)**.

## Notes
- These are tiny-sample smoke numbers to prove the live wiring, **not** the
  paper's headline figures. Headline results require the full BIRD Dev (1,534 q)
  and the fixed Spider-200 subset:
  ```
  python3 run.py --dataset bird      --policy adaptive
  python3 run.py --dataset spider200 --policy adaptive
  python3 run_baselines.py --method all --dataset bird
  ```
- Response caches (`outputs/*.cache.jsonl`) make re-runs reuse identical model
  outputs, so scaling up resumes rather than repeats paid calls.
