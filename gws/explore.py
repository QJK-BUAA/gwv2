"""Explore policy: EVSI-ranked, cost-gated probe selection and stopping.

Implements Algorithm 1 lines 6--13 and Eqs. (select)/(stop). Given the decision
model's decisions (each with interpretations, priors, and attached probes), the
policy repeatedly selects the highest-scoring remaining probe, tests the cost
threshold, executes it, and updates only the associated decision's belief.

Policies (Appendix "implementation identifiers"):
  adaptive -- EVSI ranking + cost-aware stopping (the full GraphWalkerSQL).
  voi      -- EVSI ranking, stopping disabled (budget / exhaustion only).
  eig      -- entropy-reduction ranking, no cost stop.
  random   -- uniform selection with a fixed seed, no cost stop.
  no-probe -- no active acquisition at all.

All policies share the budget cap (4 active attempts) and candidate exhaustion;
failed/timed-out probes still consume an attempt.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import belief as B
from . import config
from .probe_executor import run_probe
from .types import Decision, Probe, ProbeOutcome


@dataclass
class ExploreTrace:
    """Auditable record of the Explore loop for one question."""
    policy: str
    steps: list[dict] = field(default_factory=list)
    attempts: int = 0
    stop_reason: str = ""
    # Final per-decision posterior MAP + belief, consumed by Commit.
    posteriors: dict[str, dict[str, float]] = field(default_factory=dict)
    map_recommendations: dict[str, str] = field(default_factory=dict)


def _score(policy: str, belief: dict[str, float], probe: Probe) -> float:
    if policy == "eig":
        return B.eig(belief, probe)
    return B.evsi(belief, probe)   # adaptive / voi / random all rank by EVSI value when needed


def explore(
    sqlite_path: str,
    decisions: list[Decision],
    policy: str = "adaptive",
    budget: int = config.ACTIVE_PROBE_BUDGET,
    lam: float = config.LAMBDA_COST,
    cost: float = config.PROBE_COST,
    seed: int = config.DEFAULT_SEED,
    probe_runner=run_probe,
) -> ExploreTrace:
    """Run the Explore loop and return beliefs + an auditable trace.

    ``probe_runner`` is injectable so the loop can be unit-tested with a
    deterministic stub instead of a live SQLite probe.
    """
    policy = policy.lower()
    trace = ExploreTrace(policy=policy)

    # Initialize each decision's belief from its normalized priors.
    beliefs: dict[str, dict[str, float]] = {
        d.id: B.normalize_prior(d) for d in decisions
    }
    # Remaining probe pool: (decision_id, probe).
    pool: list[tuple[str, Probe]] = [
        (d.id, p) for d in decisions for p in d.probes
    ]

    if policy == "no-probe":
        trace.stop_reason = "no_probe_policy"
        _finalize(trace, decisions, beliefs)
        return trace

    rng = random.Random(seed)

    while pool and trace.attempts < budget:
        # --- selection (Eq. select) -------------------------------------- #
        if policy == "random":
            idx = rng.randrange(len(pool))
            best_did, best_probe = pool[idx]
            best_score = _score("evsi", beliefs[best_did], best_probe)
        else:
            scored = [
                (_score(policy, beliefs[did], pr), did, pr)
                for did, pr in pool
            ]
            # Ties broken deterministically by probe id (Appendix).
            best_score, best_did, best_probe = max(
                scored, key=lambda s: (s[0], _neg_id(s[2].id))
            )
            idx = next(i for i, (did, pr) in enumerate(pool)
                       if did == best_did and pr.id == best_probe.id)

        # --- cost-aware stopping (Eq. stop), adaptive only ---------------- #
        if policy == "adaptive" and best_score <= lam * cost:
            trace.stop_reason = "cost_threshold"
            break

        # --- execute probe, update only its decision's belief ------------- #
        belief_before = dict(beliefs[best_did])
        outcome: ProbeOutcome = probe_runner(sqlite_path, best_probe)
        trace.attempts += 1
        pool.pop(idx)

        if outcome.available:
            beliefs[best_did] = B.update_belief(
                beliefs[best_did], best_probe, outcome.is_hit
            )
        trace.steps.append({
            "attempt": trace.attempts,
            "decision_id": best_did,
            "probe_id": best_probe.id,
            "kind": best_probe.kind,
            "score": round(best_score, 6),
            "score_type": "eig" if policy == "eig" else "evsi",
            "status": outcome.status,
            "available": outcome.available,
            "belief_before": {k: round(v, 4) for k, v in belief_before.items()},
            "belief_after": {k: round(v, 4) for k, v in beliefs[best_did].items()},
        })

    if not trace.stop_reason:
        trace.stop_reason = "budget" if trace.attempts >= budget else "exhausted"

    _finalize(trace, decisions, beliefs)
    return trace


def _finalize(trace: ExploreTrace, decisions: list[Decision],
              beliefs: dict[str, dict[str, float]]) -> None:
    for d in decisions:
        b = beliefs.get(d.id, {})
        trace.posteriors[d.id] = {k: round(v, 6) for k, v in b.items()}
        trace.map_recommendations[d.id] = B.map_commitment(b)


def _neg_id(probe_id: str) -> tuple:
    """Deterministic tiebreak key: smaller probe id wins (lexicographic)."""
    return tuple(-ord(c) for c in probe_id)
