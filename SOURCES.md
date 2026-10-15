# Source provenance

GraphWalkerSQL prompts and algorithm modules implement the submitted paper's
Ground, local decision model, binary probes, reconciliation, coverage audit,
and single-candidate SQL generation. The public package has no dependency on
private workspaces or historical output files.

The eight main comparisons use the frozen linker adaptations in
`baselines/source_adapters.py`, followed by the same final generator. They
are the paper's shared-backend comparisons, with the following retained
procedures:

| Comparison | Retained procedure |
|---|---|
| Full-schema | Full visible schema |
| DIN-SQL | Released schema-linking few-shot prompt and extraction |
| C3-SQL | Separate table recall and column recall |
| TA-SQL | Released dummy-SQL schema linking |
| RSL-SQL | Forward selection, preliminary SQL, schema union |
| SchemaGraphSQL | Source/destination linking and force-union shortest paths |
| PV-SQL | Released `_ground` exploration procedure |
| APEX-SQL | Released planning, exploration, pruning, selection, and refinement |

`vendor_sources.json` pins every bundled source file by SHA-256 and records
available upstream commits. `baselines/vendor/` retains PV-SQL's MIT license
and RSL-SQL's Apache-2.0 license; their authors are third-party contributors.
The SchemaGraphSQL and DIN/C3 adapters are archived adaptations with explicit
content hashes; no unavailable upstream commit is asserted.

The loader replaces backend/database imports with the shared model and read-only
database interfaces. It removes top-level demonstration code. APEX uses one
worker and deterministic result order; credentials, external state, and hidden
database calls are never inherited from an upstream client.

The resource-aligned TA/APEX controls are separate configurations. They preserve
candidate verification/correction while enforcing six logical model calls,
five database attempts, and one reserved final submission. Main linker
comparisons retain their original call counts.

`data_sources.json` pins public datasets, the official BIRD comparator, the
Spider test-suite evaluator, and BIRD's evaluation-only golden schema
annotations. BIRD uses set-of-rows comparison; Spider uses the pinned
`eval_exec_match` with `plug_value=False` and `keep_distinct=False`. Both run
on the development SQLite databases in isolated read-only workers with a
30-second limit. No generated SQL is scored with a replacement comparator.

`manifests/spider200.json` lists all 200 reproduction IDs and the exact source
hash. It is a new fixed cohort because the original paper's ID list is
unavailable. Historical numeric results and the unpublished original
path-scoring implementation are not reconstructed from guesses.
