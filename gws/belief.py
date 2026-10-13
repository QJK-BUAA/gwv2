"""Belief updates and one-step EVSI (Explore stage, Sec. EVSI / Appendix).

A *decision* carries a categorical belief ``b`` over 2--4 mutually exclusive
interpretations. Each attached *probe* has binary likelihoods p(hit | z) (and
p(miss | z) = 1 - p(hit | z)). We implement:

  * Bayesian belief update (Eq. belief) given an observed hit/miss.
  * One-step EVSI (Eq. EVSI) under 0--1 utility U(k,z) = 1[k == z], for which
    V(b) = max_k b(k) and the local commitment is the posterior MAP.
  * The EIG control (Eq. EIG): expected entropy reduction, natural-log units.

All functions are deterministic and LLM-free, so the whole decision layer is
unit-testable without any API access.
"""
from __future__ import annotations

import math

from . import config
from .types import Decision, Probe

# ``p(o | z, a)`` with o in {hit, miss}.
def _obs_prob(probe: Probe, interp_key: str, observed_hit: bool) -> float:
    p_hit = float(probe.likelihoods.get(interp_key, 0.0))
    p_hit = min(1.0, max(0.0, p_hit))
    return p_hit if observed_hit else (1.0 - p_hit)


def predicted_obs_prob(belief: dict[str, float], probe: Probe,
                       observed_hit: bool) -> float:
    """p(o | a, b) = sum_z p(o | z, a) b(z)."""
    return sum(_obs_prob(probe, z, observed_hit) * bz
               for z, bz in belief.items())


def update_belief(belief: dict[str, float], probe: Probe,
                  observed_hit: bool) -> dict[str, float]:
    """Posterior b_{t+1}(z) ∝ p(o | z, a) b_t(z) (Eq. belief).

    If the predicted probability of the observation is <= MIN_OBS_PROB the
    belief is returned unchanged (the paper's "surprises leave beliefs
    unchanged" rule, which also guards against divide-by-zero).
    """
    denom = predicted_obs_prob(belief, probe, observed_hit)
    if denom <= config.MIN_OBS_PROB:
        return dict(belief)
    return {z: (_obs_prob(probe, z, observed_hit) * bz) / denom
            for z, bz in belief.items()}


def value(belief: dict[str, float]) -> float:
    """V(b) = max_k b(k) under 0--1 utility."""
    return max(belief.values()) if belief else 0.0


def map_commitment(belief: dict[str, float]) -> str:
    """argmax_k b(k); ties broken by interpretation key for determinism."""
    if not belief:
        return ""
    return max(sorted(belief.keys()), key=lambda k: belief[k])


def evsi(belief: dict[str, float], probe: Probe) -> float:
    """One-step EVSI (Eq. EVSI) under 0--1 utility.

    EVSI(a) = E_{o|a,b}[ max_k b_{t+1}^{a,o}(k) ] - max_k b_t(k)
            = sum_o p(o|a,b) * V(update(b, a, o)) - V(b).
    """
    base = value(belief)
    expected = 0.0
    for observed_hit in (True, False):
        p_o = predicted_obs_prob(belief, probe, observed_hit)
        if p_o <= 0.0:
            continue
        expected += p_o * value(update_belief(belief, probe, observed_hit))
    return expected - base


def _entropy(belief: dict[str, float]) -> float:
    """Shannon entropy in natural-log units (EIG control uses nats)."""
    h = 0.0
    for p in belief.values():
        if p > 0.0:
            h -= p * math.log(p)
    return h


def eig(belief: dict[str, float], probe: Probe) -> float:
    """Expected information gain (Eq. EIG): H(b) - E_o[H(b_{t+1})]."""
    base = _entropy(belief)
    expected = 0.0
    for observed_hit in (True, False):
        p_o = predicted_obs_prob(belief, probe, observed_hit)
        if p_o <= 0.0:
            continue
        expected += p_o * _entropy(update_belief(belief, probe, observed_hit))
    return base - expected


def normalize_prior(decision: Decision) -> dict[str, float]:
    """Initial belief b_0: decision-model scores normalized over interpretations.

    Negative / non-finite priors become zero; an empty or zero-sum vector falls
    back to a uniform distribution so the decision stays well-formed.
    """
    keys = [itp.key for itp in decision.interpretations]
    raw = []
    for itp in decision.interpretations:
        v = itp.prior
        if not math.isfinite(v) or v < 0.0:
            v = 0.0
        raw.append(v)
    total = sum(raw)
    if total <= 0.0:
        n = len(keys)
        return {k: 1.0 / n for k in keys} if n else {}
    return {k: v / total for k, v in zip(keys, raw)}
