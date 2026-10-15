"""Validated finite decision models and globally deduplicated binary actions."""
import json
from dataclasses import asdict
from . import config, paper_prompts as prompts
from .belief import normalize
from .ground import _common_context
from .parsing import extract_json
from .probes import compile_probe
from .sql_analysis import legal_columns
from .types import Decision, Interpretation, Probe


def parse_decisions_detailed(text, schema):
    obj = extract_json(text, ("decisions",))
    if obj is None:
        raise ValueError("Invalid decision-model response")
    decisions, diagnostics, action_ids, facts = {}, [], set(), set()
    raw_decisions = obj.get("decisions", [])
    if not isinstance(raw_decisions, list):
        raise ValueError("decisions must be a list")
    for i, item in enumerate(raw_decisions[:config.MAX_DECISIONS]):
        try:
            did = str(item.get("id") or f"d{i+1}")
            options = item.get("options", item.get("interpretations", []))
            if did in decisions or not isinstance(options, list) or not 2 <= len(options) <= 4:
                raise ValueError("Duplicate id or invalid alternatives")
            keys = [str(x.get("id", x.get("key", f"h{j+1}"))) for j, x in enumerate(options)]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate option ids")
            p = normalize([x["prior"] for x in options])
            interps = [Interpretation(k, str(x.get("interpretation", x.get("label", ""))), v,
                                     legal_columns(schema, x.get("columns", [])))
                       for k, x, v in zip(keys, options, p)]
            decisions[did] = Decision(did, str(item.get("need", item.get("description", ""))), interps)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            diagnostics.append(f"decision {i}: {exc}")
    actions = obj.get("actions")
    if actions is None:
        actions = [{**p, "decision": d.get("id", f"d{i+1}")}
                   for i, d in enumerate(raw_decisions) if isinstance(d, dict)
                   for p in d.get("probes", []) if isinstance(p, dict)]
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")
    for i, item in enumerate(actions[:config.MAX_PROBE_ACTIONS]):
        try:
            did = str(item.get("decision", item.get("decision_id", "")))
            if did not in decisions:
                raise ValueError("Unknown decision")
            aid = str(item.get("id") or f"a{i+1}")
            if aid in action_ids:
                raise ValueError("Duplicate action id")
            action_ids.add(aid)
            keys = [o.key for o in decisions[did].interpretations]
            if "likelihood" in item:
                rows = item["likelihood"]
                if not isinstance(rows, list) or len(rows) != len(keys):
                    raise ValueError("Likelihood dimension mismatch")
                rows = [normalize(r) for r in rows]
                if any(len(r) != 2 for r in rows):
                    raise ValueError("Binary likelihood requires two outcomes")
                like = dict(zip(keys, [r[1] for r in rows]))
            else:
                raw = item.get("likelihoods", {})
                if not isinstance(raw, dict) or set(raw) != set(keys):
                    raise ValueError("Likelihood keys mismatch")
                like = {k: float(raw[k]) for k in keys}
                for v in like.values():
                    normalize([1 - v, v])
            dotted = item.get("column", "")
            if item.get("table"):
                dotted = f"{item['table']}.{dotted}"
            col = legal_columns(schema, [dotted])
            if not col:
                raise ValueError("Unknown action column")
            t, c = col[0].split(".", 1)
            ot, oc = "", ""
            if item.get("kind", "").lower() == "overlap":
                other = item.get("other_column", "")
                if item.get("table2"):
                    other = f"{item['table2']}.{item.get('column2', '')}"
                cols = legal_columns(schema, [other])
                if not cols:
                    raise ValueError("Unknown overlap endpoint")
                ot, oc = cols[0].split(".", 1)
            probe = Probe(aid, did, str(item.get("kind", "")).lower(), t, c,
                          item.get("value", item.get("literal")), ot, oc, like,
                          str(item.get("reason", "")), str(item.get("caveat", "")))
            sql, params = compile_probe(probe)
            key = json.dumps([sql, params], ensure_ascii=False, sort_keys=True)
            if key in facts:
                diagnostics.append(f"action {aid}: duplicate SQL/parameter pair")
                continue
            facts.add(key)
            decisions[did].probes.append(probe)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            diagnostics.append(f"action {i}: {exc}")
    return list(decisions.values()), diagnostics


def parse_decisions(text, schema):
    decisions, diagnostics = parse_decisions_detailed(text, schema)
    return decisions, len(diagnostics)


def decision_model(llm, schema, question, evidence, ground, draft_exec):
    payload = {"requirements": [asdict(r) for r in ground.requirements],
               "independent_draft": ground.draft_analysis,
               "candidate_paths": [asdict(p) for p in ground.paths],
               "draft_execution": draft_exec}
    text = llm.complete(prompts.DECISION_SYS, _common_context(schema, question, evidence)
                        + "\nGrounding evidence:\n" + json.dumps(payload, ensure_ascii=False),
                        stage="decision_model")
    decisions, diagnostics = parse_decisions_detailed(text, schema)
    ground.decision_diagnostics = diagnostics
    return decisions, len(diagnostics)
