# Paper-to-code mapping

This package implements the submitted **GraphWalkerSQL: Schema-Graph Walks
for Relational Grounding in Text-to-SQL** algorithm and its experimental
controls. A source-function mapping is provided below so each behavior can be
inspected independently of model outputs.

| Paper behavior | Implementation |
|---|---|
| Algorithm 1: Ground → Explore → Commit | `gws/pipeline.py`: `prepare_upstream`, `finish_upstream`, `run_pipeline` |
| Independent requirements and draft; no cross-conditioning | `gws/ground.py`: `ground`; `gws/paper_prompts.py` |
| Physical-column and equality-join extraction, including aliases/CTEs | `gws/sql_analysis.py`: `resolve_sql_schema` |
| Structural, visible-FK, requirement, and draft candidate sources | `gws/graph_builder.py`: `build_candidate_edges` |
| Structural name/type/key eligibility; source scores .65/.75 | `gws/graph_builder.py`: `_enumerate_structural`, `name_score`; `gws/config.py` |
| Table-simple bounded best-first paths: 3/pair, 12 total, 3 edges, 512 pops | `gws/graph_builder.py`: `_search_pair`, `search_paths` |
| Local options, normalized priors, fixed binary likelihoods | `gws/decision.py`: `parse_decisions_detailed`, `decision_model` |
| Four parameterized EXISTS templates: EQUALS/NONNULL/DUPLICATE/OVERLAP | `gws/probes.py`: `compile_probe`; `gws/probe_executor.py`: `run_probe` |
| One-step EVSI under 0–1 utility, natural-log EIG, Bayes update | `gws/belief.py`: `evsi`, `eig`, `update_belief`, `map_commitment` |
| Global selection; stop at EVSI ≤ λc; λ=.015, c=1, 4 active attempts | `gws/explore.py`: `explore`; `gws/config.py` |
| Failed/incomplete probes consume an attempt and preserve beliefs | `gws/explore.py`, `gws/probe_executor.py`, `gws/execute.py` |
| Exact probe deduplication, legal endpoints, diagnostics | `gws/decision.py`, `gws/probes.py` |
| Free implication-filter skips with unchanged posterior | `gws/explore.py`: `explore` |
| Reconciler plus coverage auditor; explicit coverage/replacement reasons | `gws/commit.py`: `commit`, `_apply_coverage_contract` |
| Missing coverage restores original columns and records unresolved needs | `gws/commit.py`: `_apply_coverage_contract` |
| Only Boolean `retain=true` permits unresolved retained columns | `gws/commit.py`: `_apply_coverage_contract` |
| Legal compact DDL, accepted visible FKs, inferred joins as comments | `gws/commit.py`: `render_ddl` |
| Full schema catalog versus selected-column catalog | `gws/commit.py`: `generate_sql`, `gws/ground.py`: `catalog` |
| One final SQL, K=1; no repair/voting/draft fallback | `gws/commit.py`: `generate_sql`, `clean_final_sql` |
| Six logical model calls and ≤5 inference database attempts | `gws/budget.py`: `Budget`; `gws/pipeline.py` |
| Shared backend: temperature 0, seed 42, non-thinking, 4096-token cap | `gws/llm.py`; `baselines/common.py`: `FixedBackend` |
| Exact-request caching; logical and paid token accounting | `gws/response_cache.py`, `gws/llm.py`, `gws/experiments.py` |
| Eight actual shared-backend linker adaptations | `baselines/source_adapters.py`, `baselines/vendor/`, `baselines/methods.py` |
| Separate budget-aligned TA/APEX verification/correction | `baselines/methods.py`: `_ta_aligned`, `_apex_aligned` |
| Gold confined to offline scoring | `gws/reference_sql.py`, `gws/metrics.py`, `gws/experiments.py` |
| Database-shared D100/N100 visible-schema corruption | `gws/perturbations.py`: `visible_schema` |
| N100 seeds 42/137/911; legal type-compatible noise; exclude gold relations | `gws/experiments.py`, `gws/perturbations.py` |
| Table/column/edge/path metrics, schema compliance, noise adoption | `gws/metrics.py`: `score_structure` |
| Official EX; invalid/non-submitted predictions score zero | `gws/evaluate_official.py`: `score_one` |
| Paired exact McNemar and Holm adjustment | `gws/metrics.py`: `mcnemar`, `holm` |
| API-key-only installation, data/evaluator preparation, run | `reproduce.py`, `gws/setup.py` |

The default `--suite all` has 68 configurations on each benchmark. Main
comparisons use each linker's own procedure; controlled comparisons share
immutable Ground/draft/decision artifacts. The downstream-belief and
Commit/catalog controls additionally reuse the adaptive probe trajectory.
Belief removal preserves candidate sources/scores, probe definitions and raw
observations, while removing the decision priors/posteriors, rankings, EVSI
scores and MAP recommendations from downstream inputs.

| Paper tables | Suite / exported content |
|---|---|
| 1–2 | `main`: EX and pre-generation grounding |
| 3 | `relations`: declared-only, no draft/structure/model/overlap |
| 4 | `policies`: adaptive, VOI, EIG, random 42/137/911, no-probe |
| 5 | `mechanisms`: frozen downstream-belief removal and implication filtering |
| 6 | `commit`: contract/ordinary × full/trimmed catalog |
| 7, 11 | `perturbations`: D100 and each N100 seed, means and EX differences |
| 8 | `tokens`: shared-backend logical/paid calls and tokens |
| 9 | `aligned`: resource-aligned TA/APEX |
| 15 | `thresholds`: λ=.005/.015/.030 and no cost stop |

Each dataset output contains the sampled IDs, population size, source/data/DB
fingerprints, per-question records, upstream artifacts, trajectories, CSV
metrics, table-ready JSON, and paired statistics. Resumption checks each
record's fingerprint before reusing it. Database hashes are computed by
streaming, without reading a large database into memory.

Metric definitions are macro-averaged per question. T-EMR is exact table-set
agreement; T-R is table recall; T-F6 is `37PR/(36P+R)`. C-SRR requires all
reference columns; NSR/NSP/NSF are column recall/precision/F1. Edge metrics use
normalized equality predicates, including reference join predicates recovered
from subqueries. Empty edge sets match only when both sides are empty; the
exports also report the nonempty-reference subset separately. Unresolved
parses/illegal annotation columns are explicitly unclassified, with metric
denominators saved. Noise adoption checks predicates in the final SQL.

Two historical details cannot currently be verified. The original Spider-200
source indices are unavailable; the committed manifest specifies a new fixed
reproduction cohort from the same public dev split. The historical path
extractor is also unavailable; `path_signature` explicitly compares all simple
table paths as ordered normalized predicate sequences, invariant to reversal.
This implemented path rule is not claimed to be identical to that extractor.
Official EX semantics are independently pinned and unaffected by this rule.

The full matrix is executable, but live API outputs can differ across backend
versions. The verification log records actual tests and sample sizes;
this release does not claim the paper's headline scores were rerun.
