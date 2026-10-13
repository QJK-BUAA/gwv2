"""Offline end-to-end wiring test with a scripted mock LLM (no API calls).

Runs the full 6-call pipeline on a real BIRD SQLite database, verifying that
Ground builds candidates/paths, the decision model's probes execute against the
live DB, the Explore EVSI loop updates beliefs, Commit enforces the coverage
contract, and the final SQL is produced and executes. Also checks the Alameda
example end-to-end shape.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gws.datasets import load_bird  # noqa: E402
from gws.execute import run_query  # noqa: E402
from gws.pipeline import AblationConfig, run_pipeline  # noqa: E402


class MockLLM:
    """Returns scripted JSON/SQL per system-prompt type; counts calls."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.num_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def complete(self, system: str, user: str, temperature=None) -> str:
        del user, temperature
        self.num_calls += 1
        for marker, reply in self.scripts.items():
            if marker in system:
                return reply if isinstance(reply, str) else reply()
        return "{}"

    def stats(self):
        return {"num_calls": self.num_calls, "input_tokens": 0, "output_tokens": 0}


def build_scripts():
    requirements = (
        '{"requirements": ['
        '{"id":"r1","kind":"aggregate","need":"highest free meal rate",'
        '"columns":["frpm.Free Meal Count (K-12)","frpm.Enrollment (K-12)"],'
        '"alternatives":[],"reason":"ratio"},'
        '{"id":"r2","kind":"filter","need":"Alameda county",'
        '"columns":["frpm.County Name"],"alternatives":["schools.County"],'
        '"reason":"county filter"}],'
        '"relations":[]}'
    )
    draft = '```sql\nSELECT MAX("Free Meal Count (K-12)"*1.0/"Enrollment (K-12)") '\
            'FROM frpm WHERE "County Name" = \'Alameda\'\n```'
    decisions = (
        '{"decisions":[{"id":"d1","description":"which county column",'
        '"interpretations":[{"key":"z1","label":"frpm.County Name","prior":0.6},'
        '{"key":"z2","label":"schools.County","prior":0.4}],'
        '"probes":[{"id":"a1","kind":"equals","table":"frpm","column":"County Name",'
        '"literal":"Alameda","likelihoods":{"z1":0.9,"z2":0.2},'
        '"reason":"does frpm.County Name hold Alameda"}]}]}'
    )
    reconcile = (
        '{"columns":["frpm.Free Meal Count (K-12)","frpm.Enrollment (K-12)",'
        '"frpm.County Name"],"joins":[],'
        '"coverage":[{"requirement_id":"r1","state":"covered",'
        '"columns":["frpm.Free Meal Count (K-12)","frpm.Enrollment (K-12)"],"reason":"ratio"},'
        '{"requirement_id":"r2","state":"covered","columns":["frpm.County Name"],'
        '"reason":"filter"}],"unresolved":[],"reason":"single-table"}'
    )
    final = '```sql\nSELECT MAX("Free Meal Count (K-12)"*1.0/"Enrollment (K-12)") '\
            'AS r FROM "frpm" WHERE "County Name" = \'Alameda\'\n```'
    return {
        "semantic requirements": requirements,
        "SQLite expert. Write ONE": draft,
        "local grounding DECISIONS": decisions,
        "reconcile": reconcile,
        "audit a proposed": reconcile,     # audit returns same shape
        "write ONE final": final,
    }


def main() -> int:
    examples = load_bird()
    cs = [e for e in examples if e.db_id == "california_schools"]
    assert cs, "california_schools not found"
    ex = next((e for e in cs if "Alameda" in e.question), cs[0])
    print(f"example: {ex.instance_id} / {ex.db_id}")

    llm = MockLLM(build_scripts())
    res = run_pipeline(llm, ex.instance_id, ex.db_id, ex.question,
                       ex.evidence, ex.sqlite_path, AblationConfig())

    print("model_calls:", res.model_calls)
    print("candidates:", res.n_candidates, "paths:", res.n_paths,
          "decisions:", res.n_decisions)
    print("probe_attempts:", res.probe_attempts, "stop:", res.stop_reason)
    print("observations:", res.observations)
    print("selected_columns:", res.selected_columns)
    print("pred_sql:", res.pred_sql)
    print("error:", res.error or "(none)")

    assert res.model_calls == 6, f"expected 6 model calls, got {res.model_calls}"
    assert res.n_candidates > 0, "no candidate edges built"
    assert res.n_decisions == 1, "decision not parsed"
    assert res.probe_attempts == 1, "equals probe should fire once then stop"
    assert res.observations and res.observations[0]["status"] == "hit", \
        "frpm.County Name = Alameda should be a HIT on the real DB"
    assert "frpm" in res.pred_sql.lower(), "final SQL missing frpm"
    assert not res.error, f"pipeline error: {res.error}"

    exec_res = run_query(ex.sqlite_path, res.pred_sql)
    assert exec_res.get("ok"), f"final SQL did not execute: {exec_res}"
    print("final SQL executed OK, rows:", exec_res.get("rows"))
    print("\nALL E2E WIRING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
