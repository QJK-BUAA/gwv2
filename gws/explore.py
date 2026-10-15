"""Global EVSI/EIG ranking with attempt costs and optional implication skips."""
import random
from dataclasses import dataclass, field, asdict
from . import belief as B, config
from .probe_executor import run_probe
from .probes import compile_probe


@dataclass
class ExploreTrace:
    policy: str
    steps: list = field(default_factory=list)
    attempts: int = 0
    stop_reason: str = ""
    posteriors: dict = field(default_factory=dict)
    map_recommendations: dict = field(default_factory=dict)
    skips: list = field(default_factory=list)


def explore(sqlite_path, decisions, policy="adaptive", budget=config.ACTIVE_PROBE_BUDGET,
            lam=config.LAMBDA_COST, cost=config.PROBE_COST, seed=42, probe_runner=run_probe,
            implication_filter=False, resource_budget=None):
    if policy not in {"adaptive", "voi", "eig", "random", "no-probe"}:
        raise ValueError("Unknown probe policy")
    trace = ExploreTrace(policy)
    beliefs = {d.id: B.normalize_prior(d) for d in decisions}
    pool, seen = [], set()
    for d in decisions:
        for p in d.probes:
            sql, params = compile_probe(p)
            fact = (sql, params)
            if fact in seen:
                trace.skips.append({"probe_id": p.id, "reason": "duplicate_fact"})
                continue
            seen.add(fact)
            pool.append((d.id, p))
    pool.sort(key=lambda x: x[1].id)
    if policy == "no-probe":
        trace.stop_reason = "no_probe_policy"
    else:
        rng, nonnull = random.Random(seed), set()
        while pool and trace.attempts < budget:
            scores = sorted([{"probe_id": p.id, "decision_id": did,
                              "evsi": B.evsi(beliefs[did], p), "eig": B.eig(beliefs[did], p)}
                             for did, p in pool],
                            key=lambda x: (-x["eig" if policy == "eig" else "evsi"], x["probe_id"]))
            aid = rng.choice(pool)[1].id if policy == "random" else scores[0]["probe_id"]
            idx = next(i for i, (_, p) in enumerate(pool) if p.id == aid)
            did, probe = pool[idx]
            if implication_filter and probe.kind == "nonnull" and (probe.table, probe.column) in nonnull:
                trace.skips.append({"probe_id": probe.id, "action": asdict(probe),
                                    "reason": "implied_nonnull_hit", "free": True})
                pool.pop(idx)
                continue
            score = next(s for s in scores if s["probe_id"] == aid)["eig" if policy == "eig" else "evsi"]
            if policy == "adaptive" and score <= lam * cost:
                trace.stop_reason = "cost_threshold"
                break
            before = dict(beliefs[did])
            if resource_budget is None:
                outcome = probe_runner(sqlite_path, probe)
            else:
                outcome = probe_runner(sqlite_path, probe, budget=resource_budget)
            trace.attempts += 1
            pool.pop(idx)
            surprise = False
            if outcome.available:
                surprise = B.predicted_obs_prob(before, probe, outcome.is_hit) <= config.MIN_OBS_PROB
                beliefs[did] = B.update_belief(before, probe, outcome.is_hit)
            if outcome.is_hit:
                if probe.kind in {"duplicate", "nonnull"} or probe.kind == "equals" and probe.literal is not None:
                    nonnull.add((probe.table, probe.column))
                if probe.kind == "overlap":
                    nonnull.update({(probe.table, probe.column), (probe.table2, probe.column2)})
            trace.steps.append({"attempt": trace.attempts, "decision_id": did,
                "probe_id": probe.id, "kind": probe.kind, "action": asdict(probe),
                "sql": outcome.sql, "parameters": list(outcome.parameters),
                "score": score, "ranked_scores": scores,
                "score_type": "eig" if policy == "eig" else "evsi",
                "status": outcome.status, "available": outcome.available,
                "error": outcome.error, "surprise": surprise,
                "belief_before": before, "belief_after": dict(beliefs[did]),
                "cost": cost, "budget_remaining": budget - trace.attempts})
    if not trace.stop_reason:
        trace.stop_reason = "budget" if trace.attempts >= budget else "exhausted"
    trace.posteriors = beliefs
    trace.map_recommendations = {d.id: B.map_commitment(beliefs[d.id]) for d in decisions}
    return trace
