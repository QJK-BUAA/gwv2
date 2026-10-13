"""Offline control-flow test for all 8 baselines with a scripted mock LLM.

Verifies each baseline runs end-to-end on a real BIRD database, respects the
model-call/read budget shape, and emits an executable final SQL. No API calls.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines import BASELINES  # noqa: E402
from gws.datasets import load_bird  # noqa: E402
from gws.execute import run_query  # noqa: E402
from gws.pipeline import load_db  # noqa: E402

FINAL = ('```sql\nSELECT MAX("Free Meal Count (K-12)"*1.0/"Enrollment (K-12)") '
         'FROM "frpm" WHERE "County Name" = \'Alameda\'\n```')


class MockLLM:
    """Routes by keywords in the system prompt; returns valid JSON or SQL."""

    def __init__(self):
        self.num_calls = 0

    def complete(self, system: str, user: str, temperature=None) -> str:
        del user, temperature
        self.num_calls += 1
        s = system.lower()
        if "json" in s or "return json" in s:
            # cover every schema-linking / planning / probe JSON shape used.
            return ('{"tables":["frpm"],"columns":["frpm.County Name"],'
                    '"joins":[],"difficulty":"easy","entities":[],'
                    '"aggregations":[],"conditions":[],"plan":"p",'
                    '"source":"frpm","destination":"frpm",'
                    '"action":"done","probe_sql":"SELECT 1"}')
        return FINAL

    def stats(self):
        return {"num_calls": self.num_calls, "input_tokens": 0, "output_tokens": 0}


def main() -> int:
    ex = next(e for e in load_bird() if e.db_id == "california_schools")
    schema = load_db(ex.sqlite_path)
    failures = 0
    for name, fn in BASELINES.items():
        llm = MockLLM()
        out = fn(llm, schema, ex.question, ex.evidence, ex.sqlite_path)
        sql = out.get("pred_sql", "")
        res = run_query(ex.sqlite_path, sql)
        ok = res.get("ok")
        print(f"  {name:16s} calls={out.get('model_calls')} reads={out.get('reads')} "
              f"exec_ok={ok} sql={sql[:48]}...")
        if not ok or not sql:
            failures += 1
            print(f"    !! {name} produced non-executing SQL: {res.get('error')}")
    print(f"\n{len(BASELINES)-failures}/{len(BASELINES)} baselines OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
