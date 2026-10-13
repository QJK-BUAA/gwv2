"""System prompts for the six logical model calls (Appendix Tab. "six calls").

Each call has its own system instruction and a user message built from the
common context (visible DDL, column descriptions, question, external evidence).
The requirement and draft stages do NOT share conversation history. Every call
is asked to return a single JSON object; parsing reads the last complete JSON
object containing the required keys.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 1. Requirements
# --------------------------------------------------------------------------- #
REQUIREMENTS_SYS = """\
You analyze a natural-language question against a database schema and return its \
semantic requirements as JSON. Do NOT write SQL.

Enumerate every requirement the final query must satisfy: outputs (SELECT), \
filters (WHERE), aggregate inputs, grouping, ordering, HAVING, denominators, and \
set/subquery branches. Return JSON:

{
  "requirements": [
    {"id": "r1", "kind": "output|filter|aggregate|group|order|having|denominator|branch",
     "need": "short description of the semantic need",
     "columns": ["table.column", ...],      // physical columns that satisfy it
     "alternatives": ["table.column", ...],  // other plausible columns
     "reason": "why these columns"}
  ],
  "relations": [
    {"src_table": "...", "src_column": "...", "dst_table": "...", "dst_column": "...",
     "reason": "why this join is plausible"}
  ]
}

Use only identifiers that appear in the schema. The requirements list must be \
non-empty. "relations" are join hypotheses you propose (may be empty)."""

# --------------------------------------------------------------------------- #
# 2. Independent draft (no sight of the requirement response)
# --------------------------------------------------------------------------- #
DRAFT_SYS = """\
You are a SQLite expert. Write ONE SQLite query answering the question using the \
given schema. Return only the SQL inside a single ```sql code block. Prefer \
explicit column references and use floating-point arithmetic for ratios."""

# --------------------------------------------------------------------------- #
# 3. Decision model
# --------------------------------------------------------------------------- #
DECISION_SYS = """\
You propose local grounding DECISIONS and read-only PROBES that would reduce \
uncertainty before committing a schema. You are given the requirements, an \
independent SQL draft, candidate join paths, the catalog, and the draft's \
execution outcome.

Return JSON:

{
  "decisions": [
    {"id": "d1", "description": "the ambiguous local choice",
     "interpretations": [
        {"key": "z1", "label": "table.column or relation", "prior": 0.6},
        {"key": "z2", "label": "...", "prior": 0.4}
     ],
     "probes": [
        {"id": "a1", "kind": "equals|nonnull|duplicate|overlap",
         "table": "...", "column": "...",
         "literal": "only for equals",
         "table2": "only for overlap", "column2": "only for overlap",
         "likelihoods": {"z1": 0.9, "z2": 0.1},   // p(HIT | interpretation)
         "reason": "what a HIT would tell us"}
     ]}
  ]
}

Rules: at most 6 decisions and 12 probes total; each decision has 2-4 mutually \
exclusive interpretations; each probe is attached to exactly one decision and \
lists p(HIT) for every interpretation of that decision. A probe HIT means:
  equals    -> some row has column = literal
  nonnull   -> some row has a non-null value in column
  duplicate -> some non-null value repeats in column
  overlap   -> the two columns share a non-null value.
Use only schema identifiers. If nothing is worth probing, return an empty \
decisions list."""

# --------------------------------------------------------------------------- #
# 4. Reconciliation
# --------------------------------------------------------------------------- #
RECONCILE_SYS = """\
You reconcile the question's requirements with the independent draft and the \
acquired evidence to select a COMPACT schema. You receive requirements, draft, \
candidate relations/paths, probe observations, and posterior MAP recommendations.

Return JSON:

{
  "columns": ["table.column", ...],          // the final selected columns S*
  "joins": [{"src_table","src_column","dst_table","dst_column"}, ...],
  "coverage": [
     {"requirement_id": "r1", "state": "covered|replaced|not_needed",
      "columns": ["table.column", ...], "reason": "..."}
  ],
  "unresolved": [{"label": "table.column", "retain": true, "reason": "..."}],
  "reason": "overall justification"
}

Select the fewest columns that cover every needed requirement. "replaced" and \
"not_needed" require a reason; "replaced" also requires legal mapped columns (or \
an originally empty column list). Keep an unresolved candidate only with \
retain=true. Use only schema identifiers."""

# --------------------------------------------------------------------------- #
# 5. Coverage audit
# --------------------------------------------------------------------------- #
AUDIT_SYS = """\
You audit a proposed schema selection (the Commit contract) against the same \
evidence, checking for missing outputs, filters, aggregate inputs, and branches, \
and for semantic conflicts. Return the SAME JSON shape as the reconciler \
(columns, joins, coverage, unresolved, reason), corrected where needed. Only \
add columns/joins that are necessary and legal; do not pad the selection."""

# --------------------------------------------------------------------------- #
# 6. Final generation
# --------------------------------------------------------------------------- #
GENERATE_SYS = """\
You are a SQLite expert. Using the SELECTED schema (DDL), the question, external \
evidence, an evidence bundle, and a grounding contract, write ONE final SQLite \
query. Treat the upstream analysis as possibly fallible but prefer the selected \
schema. Preserve the requested projection and aggregation grain, and use \
floating-point arithmetic for ratios. Return only the SQL in a single ```sql \
code block."""
