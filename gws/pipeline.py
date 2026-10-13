"""End-to-end GraphWalkerSQL pipeline (Algorithm 1).

Six logical model calls per question:
  1. requirements        (Ground)
  2. independent draft    (Ground)
  3. decision model       (Explore)
  4. reconciliation       (Commit)
  5. coverage audit       (Commit)
  6. final generation     (Commit)
Database probes run between calls and do not add to the model-call count.

``AblationConfig`` exposes every lever the paper ablates: probe policy
(adaptive / voi / eig / random / no-probe), relation-construction toggles
(independent draft, structural / model-proposed candidates, value-overlap
evidence), the downstream-belief control, and the ordinary two-round Commit
filter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import commit as commit_mod
from . import config
from .decision import decision_model
from .execute import run_query
from .ground import ground as run_ground
from .explore import explore
from .probe_executor import run_probe
from .schema import (Schema, collect_column_stats, extract_schema,
                     load_column_descriptions)
from .types import Decision


@dataclass
class AblationConfig:
    probe_policy: str = "adaptive"            # adaptive|voi|eig|random|no-probe
    budget: int = config.ACTIVE_PROBE_BUDGET
    lam: float = config.LAMBDA_COST
    seed: int = config.DEFAULT_SEED
    # relation-construction ablations
    include_draft: bool = True
    include_structural: bool = True
    include_model: bool = True
    use_value_overlap: bool = True            # allow overlap probes
    # commit / belief controls
    include_beliefs: bool = True              # downstream-belief control
    coverage_contract: bool = True            # vs ordinary two-round filter
    collect_stats: bool = False               # populate uniqueness for structural

    def label(self) -> str:
        tags = [self.probe_policy]
        if not self.include_draft:
            tags.append("nodraft")
        if not self.include_structural:
            tags.append("nostruct")
        if not self.include_model:
            tags.append("nomodel")
        if not self.use_value_overlap:
            tags.append("noverlap")
        if not self.include_beliefs:
            tags.append("nobelief")
        if not self.coverage_contract:
            tags.append("ordinary")
        return "+".join(tags)


@dataclass
class GWSResult:
    instance_id: str
    db_id: str
    question: str
    pred_sql: str
    draft_sql: str = ""
    selected_columns: list[str] = field(default_factory=list)
    n_candidates: int = 0
    n_paths: int = 0
    n_decisions: int = 0
    probe_attempts: int = 0
    stop_reason: str = ""
    explore_steps: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    model_calls: int = 0
    error: str = ""


# --------------------------------------------------------------------------- #
# Schema loading (with BIRD column descriptions when present)
# --------------------------------------------------------------------------- #
_SCHEMA_CACHE: dict[str, Schema] = {}


def load_db(sqlite_path: str, collect_stats: bool = False) -> Schema:
    key = f"{sqlite_path}|{collect_stats}"
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]
    schema = extract_schema(sqlite_path)
    desc_dir = os.path.join(os.path.dirname(sqlite_path), "database_description")
    if os.path.isdir(desc_dir):
        schema.descriptions = load_column_descriptions(desc_dir)
    if collect_stats:
        collect_column_stats(sqlite_path, schema)
    _SCHEMA_CACHE[key] = schema
    return schema


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _draft_exec_summary(sqlite_path: str, draft_sql: str) -> str:
    if not draft_sql:
        return "no draft"
    res = run_query(sqlite_path, draft_sql, max_rows=5)
    if not res.get("ok"):
        return f"error: {res.get('error', '')[:160]}"
    rows = res.get("rows") or []
    preview = rows[0] if rows else "()"
    return f"ok, {len(rows)} row(s); first={preview}"


def _run_probes(sqlite_path: str, decisions: list[Decision],
                ablation: AblationConfig):
    """Run the Explore loop; drop overlap probes if value-overlap is disabled."""
    if not ablation.use_value_overlap:
        for d in decisions:
            d.probes = [p for p in d.probes if p.kind != "overlap"]
    return explore(
        sqlite_path, decisions,
        policy=ablation.probe_policy, budget=ablation.budget,
        lam=ablation.lam, seed=ablation.seed, probe_runner=run_probe,
    )


def run_pipeline(llm, instance_id: str, db_id: str, question: str,
                 evidence: str, sqlite_path: str,
                 ablation: AblationConfig | None = None) -> GWSResult:
    ablation = ablation or AblationConfig()
    result = GWSResult(instance_id=instance_id, db_id=db_id, question=question,
                       pred_sql="")
    try:
        schema = load_db(sqlite_path, collect_stats=ablation.collect_stats)

        # --- Ground (calls 1-2) ------------------------------------------- #
        g = run_ground(llm, schema, question, evidence,
                       include_structural=ablation.include_structural,
                       include_model=ablation.include_model,
                       include_draft=ablation.include_draft)
        result.draft_sql = g.draft_sql
        result.n_candidates = len(g.candidates)
        result.n_paths = len(g.paths)

        draft_exec = _draft_exec_summary(sqlite_path, g.draft_sql)

        # --- Explore (call 3 + probes) ------------------------------------ #
        decisions, _rejected = decision_model(llm, schema, question, evidence,
                                              g, draft_exec)
        result.n_decisions = len(decisions)
        trace = _run_probes(sqlite_path, decisions, ablation)
        result.probe_attempts = trace.attempts
        result.stop_reason = trace.stop_reason
        result.explore_steps = trace.steps
        observations = [
            {"probe_id": s["probe_id"], "kind": s["kind"], "status": s["status"]}
            for s in trace.steps
        ]
        result.observations = observations

        # --- Commit (calls 4-6) ------------------------------------------- #
        contract = commit_mod.commit(
            llm, schema, question, evidence, g, decisions,
            trace.posteriors, trace.map_recommendations, observations,
            include_beliefs=ablation.include_beliefs,
            coverage_contract=ablation.coverage_contract,
        )
        result.selected_columns = list(contract.columns)
        pred = commit_mod.generate_sql(llm, schema, question, evidence, g,
                                       observations, contract)
        result.pred_sql = pred or "SELECT 1"
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        if not result.pred_sql:
            result.pred_sql = "SELECT 1"
    result.model_calls = getattr(llm, "num_calls", 0)
    return result
