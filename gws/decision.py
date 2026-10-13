"""Decision-model call: parse local decisions, interpretations, and probes.

The decision model considers at most the first six decisions and twelve actions.
Each probe is linked to exactly one decision and lists p(HIT) per interpretation.
Likelihood vectors are sanitized (negative/non-finite -> 0, then normalized per
binary row); invalid probes (bad kind, unknown identifiers, missing args) are
dropped with a count. The result feeds the EVSI Explore loop.
"""
from __future__ import annotations

import math

from . import config, prompts
from .ground import GroundResult
from .parsing import extract_json
from .probes import VALID_KINDS
from .schema import Schema
from .types import Decision, Interpretation, Probe


def _ctx(schema: Schema, question: str, evidence: str, ground: GroundResult,
         draft_exec: str) -> str:
    lines = [schema.to_ddl_text(annotate=True), "", f"Question: {question}"]
    if evidence:
        lines.append(f"External evidence: {evidence}")
    lines.append("")
    lines.append("Requirements:")
    for r in ground.requirements:
        lines.append(f"  [{r.id}] {r.kind}: {r.need} -> {r.columns} "
                     f"(alts {r.alternatives})")
    lines.append("")
    lines.append(f"Independent draft SQL:\n{ground.draft_sql or '(none)'}")
    lines.append(f"Draft execution: {draft_exec}")
    if ground.paths:
        lines.append("")
        lines.append("Candidate join paths:")
        for p in ground.paths[:config.MAX_PATHS_OVERALL]:
            lines.append(f"  {p.path_id}")
    return "\n".join(lines)


def _sanitize_likelihoods(raw, interp_keys: list[str]) -> dict[str, float] | None:
    """p(HIT|z) per interpretation; drop if dimensions/values are invalid."""
    if not isinstance(raw, dict):
        return None
    vec: dict[str, float] = {}
    for k in interp_keys:
        v = raw.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        if not math.isfinite(v) or v < 0.0:
            v = 0.0
        vec[k] = min(1.0, v)
    if not vec or len(vec) != len(interp_keys):
        return None
    return vec


def parse_decisions(text: str, schema: Schema) -> tuple[list[Decision], int]:
    """Return (decisions, n_rejected_probes). Enforces the paper's caps."""
    obj = extract_json(text, ("decisions",))
    if not obj:
        return [], 0
    decisions: list[Decision] = []
    rejected = 0
    total_probes = 0
    for drec in (obj.get("decisions") or [])[: config.MAX_DECISIONS]:
        if not isinstance(drec, dict):
            continue
        interps_raw = drec.get("interpretations") or []
        interps: list[Interpretation] = []
        for irec in interps_raw:
            if not isinstance(irec, dict):
                continue
            key = str(irec.get("key") or f"z{len(interps)+1}")
            try:
                prior = float(irec.get("prior", 0.0))
            except (TypeError, ValueError):
                prior = 0.0
            interps.append(Interpretation(key=key,
                                          label=str(irec.get("label") or ""),
                                          prior=prior))
        if not (2 <= len(interps) <= 4):
            continue
        interp_keys = [i.key for i in interps]
        did = str(drec.get("id") or f"d{len(decisions)+1}")

        probes: list[Probe] = []
        for prec in (drec.get("probes") or []):
            if total_probes >= config.MAX_PROBE_ACTIONS:
                break
            if not isinstance(prec, dict):
                continue
            kind = str(prec.get("kind") or "").lower()
            if kind not in VALID_KINDS:
                rejected += 1
                continue
            like = _sanitize_likelihoods(prec.get("likelihoods") or {}, interp_keys)
            if like is None:
                rejected += 1
                continue
            probe = Probe(
                id=str(prec.get("id") or f"a{total_probes+1}"),
                decision_id=did, kind=kind,
                table=str(prec.get("table") or ""),
                column=str(prec.get("column") or ""),
                literal=prec.get("literal"),
                table2=str(prec.get("table2") or ""),
                column2=str(prec.get("column2") or ""),
                likelihoods=like, reason=str(prec.get("reason") or ""),
            )
            if not _probe_identifiers_ok(probe, schema):
                rejected += 1
                continue
            probes.append(probe)
            total_probes += 1
        decisions.append(Decision(id=did, description=str(drec.get("description") or ""),
                                  interpretations=interps, probes=probes))
    return decisions, rejected


def _probe_identifiers_ok(probe: Probe, schema: Schema) -> bool:
    if schema.column(probe.table, probe.column) is None:
        return False
    if probe.kind == "equals" and probe.literal is None:
        return False
    if probe.kind == "overlap":
        if schema.column(probe.table2, probe.column2) is None:
            return False
    return True


def decision_model(llm, schema: Schema, question: str, evidence: str,
                   ground: GroundResult, draft_exec: str
                   ) -> tuple[list[Decision], int]:
    text = llm.complete(prompts.DECISION_SYS,
                        _ctx(schema, question, evidence, ground, draft_exec))
    return parse_decisions(text, schema)
