"""Offline per-question set metrics, schema compliance, and paired statistics."""
import json
import math
from pathlib import Path
from .reference_sql import extract_reference


def normalized(values):
    return {str(x).replace('`', '').replace('"', '').lower() for x in values}


def edge_keys(values):
    out = set()
    for x in values:
        if isinstance(x, str):
            parts = x.split(" = ")
            if len(parts) != 2:
                continue
            a, b = parts
        elif isinstance(x, dict):
            a = x.get("left") or f"{x.get('src_table', '')}.{x.get('src_column', '')}"
            b = x.get("right") or f"{x.get('dst_table', x.get('ref_table', ''))}.{x.get('dst_column', x.get('ref_column', ''))}"
        else:
            a, b = x
        a, b = next(iter(normalized([a]))), next(iter(normalized([b])))
        out.add(tuple(sorted((a, b))))
    return out


def prf(pred, gold, symmetric_empty=False):
    if symmetric_empty and (not pred or not gold):
        value = float(not pred and not gold)
        return value, value, value, value
    hit = len(pred & gold)
    p = hit / len(pred) if pred else float(not gold)
    r = hit / len(gold) if gold else 1.
    return p, r, (2*p*r/(p+r) if p+r else 0.), (37*p*r/(36*p+r) if p+r else 0.)


def load_annotations(path):
    if not path or not Path(path).is_file():
        return {}
    rows = json.loads(Path(path).read_text())
    return {f"bird_{r['question_id']}": {
        "tables": normalized(t["table_name"] for t in r["schema"]),
        "columns": normalized(f"{t['table_name']}.{c}" for t in r["schema"] for c in t["columns"]),
        "db_id": r["db_id"]} for r in rows}


def path_signature(edges):
    """Exact normalized simple column-predicate paths, invariant to reversal.

    Includes every simple table path of the reference graph and its predicates;
    no claim of identity to an unavailable historical path evaluator is made.
    """
    adjacency = {}
    for a, b in edges:
        ta, tb = a.split(".", 1)[0], b.split(".", 1)[0]
        adjacency.setdefault(ta, []).append((tb, (a, b)))
        adjacency.setdefault(tb, []).append((ta, (a, b)))
    paths = set()
    def visit(table, seen, predicates):
        for nxt, edge in adjacency.get(table, []):
            if nxt in seen:
                continue
            sequence = predicates + (edge,)
            paths.add(min(sequence, tuple(reversed(sequence))))
            visit(nxt, seen | {nxt}, sequence)
    for t in sorted(adjacency):
        visit(t, {t}, ())
    return paths


def score_structure(record, gold_sql, schema, annotation=None, injected=()):
    ref = extract_reference(gold_sql, schema)
    valid = normalized(f"{t}.{c.name}" for t, cs in schema.tables.items() for c in cs)
    cols = normalized(record.get("selected_columns", []))
    owners = {f"{t}.{c.name}".lower(): t.lower() for t, cs in schema.tables.items() for c in cs}
    tabs = {owners[c] for c in cols if c in owners}
    reference_cols = set(annotation["columns"] if annotation else ref["columns"])
    reference_tabs = set(annotation["tables"] if annotation else ref["tables"])
    reference_valid = annotation is not None or ref["status"] == "ok"
    annotation_valid = reference_valid and reference_cols <= valid
    tp, tr, _, tf = prf(tabs, reference_tabs)
    cp, cr, cf, _ = prf(cols, reference_cols)
    gold_edges = {tuple(e) for e in ref["edges"]}
    selected = edge_keys(record.get("selected_joins", []))
    candidates = edge_keys(record.get("upstream", {}).get("ground", {}).get("candidates", []))
    ep, er, ef, _ = prf(selected, gold_edges, symmetric_empty=True)
    _, candidate_recall, _, _ = prf(candidates, gold_edges, symmetric_empty=True)
    scores = {"T-EMR": float(tabs == reference_tabs) if reference_valid else None,
              "T-R": tr if reference_valid else None, "T-F6": tf if reference_valid else None,
              "C-SRR": float(reference_cols <= cols) if annotation_valid else None,
              "NSR": cr if annotation_valid else None, "NSP": cp if annotation_valid else None,
              "NSF": cf if annotation_valid else None, "#Cols": len(cols),
              "Edge-EMR": float(selected == gold_edges) if ref["status"] == "ok" else None,
              "Edge-P": ep if ref["status"] == "ok" else None,
              "Edge-R": er if ref["status"] == "ok" else None,
              "Edge-F1": ef if ref["status"] == "ok" else None,
              "Candidate-Edge-R": candidate_recall if ref["status"] == "ok" else None,
              "Path-EMR": float(path_signature(selected) == path_signature(gold_edges))
                          if ref["status"] == "ok" else None,
              "reference_has_joins": bool(gold_edges), "annotation_valid": annotation_valid,
              "reference_parse_status": ref["status"]}
    used = extract_reference(record.get("pred_sql", ""), schema)
    scores["sql_parse_status"] = used["status"]
    scores["out_of_selection"] = (bool(set(used["columns"]) - cols)
                                   if used["status"] == "ok" else None)
    scores["out_of_selection_columns"] = (sorted(set(used["columns"]) - cols)
                                           if used["status"] == "ok" else [])
    scores["noise_adopted"] = (bool(set(used["edges"]) & {tuple(e) for e in injected})
                             if used["status"] == "ok" else None)
    return scores


def macro(rows, field):
    values = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
    return sum(values)/len(values) if values else None


def mcnemar(first, second):
    shared = sorted(set(first) & set(second))
    b = sum(bool(first[i]) and not second[i] for i in shared)
    c = sum(not first[i] and bool(second[i]) for i in shared)
    n = b+c
    # Exact two-sided binomial test, stable even for the full BIRD cohort.
    tail = sum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                        - n*math.log(2)) for k in range(min(b, c)+1)) if n else .5
    return {"n": len(shared), "first_only": b, "second_only": c,
            "p": min(1., 2*tail)}


def holm(comparisons):
    ordered = sorted(enumerate(comparisons), key=lambda item: item[1]["p"])
    result = [dict(c) for c in comparisons]
    previous = 0.
    for rank, (index, row) in enumerate(ordered):
        previous = min(1., max(previous, row["p"]*(len(ordered)-rank)))
        result[index]["p_holm"] = previous
        result[index]["significant_0.05"] = previous < .05
    return result
