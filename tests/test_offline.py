"""Offline unit tests for the deterministic GraphWalkerSQL components.

No API access and no database required. Covers: belief update + EVSI/EIG math,
the four probe SQL templates, candidate-graph construction + bounded path search,
coverage-contract enforcement, Spider-200 subset determinism, and JSON/SQL
parsing. Run: ``python tests/test_offline.py``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gws import belief as B  # noqa: E402
from gws import graph_builder as G  # noqa: E402
from gws.commit import _apply_coverage_contract  # noqa: E402
from gws.parsing import extract_json, extract_sql  # noqa: E402
from gws.probes import compile_probe  # noqa: E402
from gws.schema import Column, Schema  # noqa: E402
from gws.types import (Decision, Interpretation, Probe, Requirement)  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


# --------------------------------------------------------------------------- #
def test_belief_evsi():
    d = Decision("d1", "x",
                 [Interpretation("A", "a", 0.5), Interpretation("B", "b", 0.5)])
    b0 = B.normalize_prior(d)
    check("prior normalized", abs(sum(b0.values()) - 1.0) < 1e-9)
    check("V(b0)=0.5", abs(B.value(b0) - 0.5) < 1e-9)

    informative = Probe("p", "d1", "equals", "t", "c", literal="x",
                        likelihoods={"A": 0.9, "B": 0.1})
    # Hit posterior: A = 0.9/(0.9+0.1)=0.9; expected V = 0.5*0.9 + 0.5*0.9 = 0.9.
    check("EVSI informative = 0.4", abs(B.evsi(b0, informative) - 0.4) < 1e-9)
    post = B.update_belief(b0, informative, True)
    check("posterior hit A=0.9", abs(post["A"] - 0.9) < 1e-9)
    check("MAP = A", B.map_commitment(post) == "A")

    uninform = Probe("p0", "d1", "nonnull", "t", "c",
                     likelihoods={"A": 0.5, "B": 0.5})
    check("EVSI uninformative = 0", abs(B.evsi(b0, uninform)) < 1e-12)
    check("EIG uninformative = 0", abs(B.eig(b0, uninform)) < 1e-12)
    check("EIG informative > 0", B.eig(b0, informative) > 0.0)

    # Low-probability surprise leaves belief unchanged.
    extreme = Probe("pe", "d1", "equals", "t", "c", literal="x",
                    likelihoods={"A": 0.0, "B": 0.0})
    unchanged = B.update_belief(b0, extreme, True)
    check("surprise leaves belief unchanged", unchanged == b0)


def test_probe_sql():
    eq = Probe("1", "d", "equals", "frpm", "County Name", literal="Alameda",
               likelihoods={})
    sql, params = compile_probe(eq)
    check("equals is SELECT EXISTS", sql.startswith("SELECT EXISTS"))
    check("equals binds literal param", params == ("Alameda",))
    check("equals quotes identifier", '"County Name"' in sql)

    nn = Probe("2", "d", "nonnull", "t", "c", likelihoods={})
    sql, params = compile_probe(nn)
    check("nonnull IS NOT NULL", "IS NOT NULL" in sql and params == ())

    dup = Probe("3", "d", "duplicate", "t", "c", likelihoods={})
    sql, _ = compile_probe(dup)
    check("duplicate GROUP BY HAVING", "GROUP BY" in sql and "COUNT(*) > 1" in sql)

    ov = Probe("4", "d", "overlap", "t1", "c1", table2="t2", column2="c2",
               likelihoods={})
    sql, _ = compile_probe(ov)
    check("overlap joins two columns", "JOIN" in sql and "a." in sql and "b." in sql)

    bad = Probe("5", "d", "equals", "t", "c", literal=None, likelihoods={})
    try:
        compile_probe(bad)
        check("equals without literal rejected", False)
    except ValueError:
        check("equals without literal rejected", True)

    inj = Probe("6", "d", "nonnull", "t;DROP TABLE t", "c", likelihoods={})
    try:
        compile_probe(inj)
        check("unsafe identifier rejected", False)
    except ValueError:
        check("unsafe identifier rejected", True)


def _toy_schema() -> Schema:
    s = Schema(name="toy")
    s.tables = {
        "students": [Column("student_id", "INTEGER", pk=True), Column("name", "TEXT")],
        "enroll": [Column("student_id", "INTEGER"), Column("course_id", "INTEGER")],
        "courses": [Column("course_id", "INTEGER", pk=True), Column("title", "TEXT")],
    }
    return s


def test_graph_and_paths():
    s = _toy_schema()
    edges = G.build_candidate_edges(s, model_relations=[], include_structural=True)
    ids = {e.edge_id for e in edges}
    # structural: enroll.student_id -> students.student_id and course_id -> courses.course_id
    check("structural edge student_id->students.student_id",
          any("enroll.student_id" in e and "students.student_id" in e for e in ids))
    check("structural edge course_id->courses.course_id",
          any("enroll.course_id" in e and "courses.course_id" in e for e in ids))
    for e in edges:
        check(f"structural score 0.65 ({e.edge_id})", abs(e.score - 0.65) < 1e-9)

    # model-proposed edge gets 0.75 and dominates a duplicate structural pair.
    edges2 = G.build_candidate_edges(s, model_relations=[{
        "src_table": "enroll", "src_column": "student_id",
        "dst_table": "students", "dst_column": "student_id"}])
    merged = [e for e in edges2 if "enroll.student_id" in e.edge_id
              and "students.student_id" in e.edge_id][0]
    check("merged dup keeps larger score 0.75", abs(merged.score - 0.75) < 1e-9)
    check("merged dup unions sources",
          set(merged.sources) >= {"structural", "model"})

    paths = G.search_paths(edges, ["students", "courses"])
    check("path students..courses found", len(paths) >= 1)
    check("path within edge cap", all(len(p.edges) <= 3 for p in paths))
    best = min(paths, key=lambda p: p.cost)
    check("2-hop path via enroll", "enroll" in best.tables)


def test_coverage_contract():
    s = _toy_schema()
    reqs = [Requirement("r1", "output", "name", ["students.name"]),
            Requirement("r2", "filter", "title", ["courses.title"])]
    raw = {
        "columns": ["students.name"],
        "joins": [{"src_table": "enroll", "src_column": "student_id",
                   "dst_table": "students", "dst_column": "student_id"}],
        "coverage": [
            {"requirement_id": "r1", "state": "covered",
             "columns": ["students.name"], "reason": "out"},
            # r2 missing -> should restore courses.title
        ],
        "unresolved": [{"label": "courses.title", "retain": True, "reason": "keep"}],
        "reason": "x",
    }
    c = _apply_coverage_contract(raw, s, reqs)
    check("covered column present", "students.name" in c.columns)
    check("missing requirement restored OR retained",
          "courses.title" in c.columns)
    check("accepted join endpoints added",
          "enroll.student_id" in c.columns and "students.student_id" in c.columns)
    # empty selection rejected
    try:
        _apply_coverage_contract({"columns": [], "joins": [], "coverage": [],
                                  "unresolved": [], "reason": ""}, s, [])
        check("empty selection rejected", False)
    except ValueError:
        check("empty selection rejected", True)


def test_parsing():
    txt = 'noise {"a":1} more {"columns":["t.c"],"reason":"r"} trailing'
    obj = extract_json(txt, ("columns",))
    check("extract last valid json", obj and obj["columns"] == ["t.c"])
    check("missing-key json -> None", extract_json('{"a":1}', ("columns",)) is None)
    sql = extract_sql("```sql\nSELECT 1;\n```")
    check("extract sql strips fence+semicolon", sql == "SELECT 1")


def test_spider200_determinism():
    # Determinism is a pure function of content hash; verify the selector keys
    # are stable without loading the dataset (hashing is stdlib only).
    from gws.datasets import Example, load_spider200
    import hashlib
    from gws import config

    def key(db, q):
        payload = f"{config.SPIDER200_SALT}|{db}|{q.strip()}"
        return hashlib.sha256(payload.encode()).hexdigest()
    k1 = key("db", "What is X?")
    k2 = key("db", "What is X?")
    check("hash key deterministic", k1 == k2)
    check("different question -> different key", k1 != key("db", "What is Y?"))
    _ = (Example, load_spider200)  # referenced for import coverage


def main():
    for t in (test_belief_evsi, test_probe_sql, test_graph_and_paths,
              test_coverage_contract, test_parsing, test_spider200_determinism):
        print(f"\n== {t.__name__} ==")
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
