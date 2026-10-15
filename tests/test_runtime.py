"""Portable regressions: real SQLite reads, scripted models, full experiment matrix."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.methods import BASELINES, ALIGNED_BASELINES
from baselines.common import SafeDB
from gws.artifacts import encode_upstream, decode_upstream, digest
from gws.budget import Budget, BudgetExceeded
from gws.commit import _apply_coverage_contract, evidence_bundle, render_ddl
from gws.decision import parse_decisions_detailed
from gws.datasets import Example
from gws.execute import run_query
from gws.experiments import run_suite, experiment_matrix
from gws.explore import explore
from gws.ground import ground
from gws.metrics import score_structure, prf, mcnemar, holm
from gws.perturbations import visible_schema
from gws.pipeline import AblationConfig, prepare_upstream, run_pipeline
from gws.reference_sql import extract_reference
from gws.schema import extract_schema, Schema, Column
from gws.types import Decision, Interpretation, Probe, ProbeOutcome, Requirement, CommitContract
from gws.llm import LLM
from gws.response_cache import ResponseCache


class ScriptedLLM:
    def __init__(self):
        self.num_calls = 0
        self.calls = []
        self.inputs = []
        self.budget = None
        self.model = "scripted"

    def stats(self):
        return {"num_calls": self.num_calls, "input_tokens": self.num_calls*10,
                "output_tokens": self.num_calls*3, "paid_calls": 0,
                "paid_input_tokens": 0, "paid_output_tokens": 0}

    def complete(self, system, user, **kwargs):
        return self.complete_messages_with_metadata([
            {"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs).text

    def complete_messages_with_metadata(self, messages, stage="", temperature=0, seed=42, **kwargs):
        if self.budget:
            self.budget.charge_call(stage, final=stage.startswith("generation"))
        self.num_calls += 1
        self.calls.append({"stage": stage, "cached": False})
        self.inputs.append(messages)
        text = "\n".join(m["content"] for m in messages)
        if stage.startswith("generation") or stage in {"independent_draft", "ta_candidate", "ta_correction"}:
            response = "SELECT name FROM users WHERE name='Alice'"
        elif stage == "requirements":
            response = json.dumps({"requirements": [
                {"id": "r1", "kind": "output", "need": "user name", "columns": ["users.name"]}],
                "joins": [{"left": "users.id", "right": "orders.user_id"}]})
        elif stage == "decision_model":
            response = json.dumps({"decisions": [{"id": "d1", "need": "name encoding",
                "options": [{"id": "h1", "interpretation": "Alice", "prior": .5},
                            {"id": "h2", "interpretation": "ALICE", "prior": .5}]}],
                "actions": [{"id": "a1", "decision": "d1", "kind": "equals",
                             "column": "users.name", "value": "Alice",
                             "likelihood": [[.1, .9], [.9, .1]]}]})
        elif stage in {"reconcile", "coverage_audit", "selection_audit"}:
            response = json.dumps({"columns": ["users.name"], "joins": [],
                "coverage": [{"id": "r1", "status": "covered", "columns": ["users.name"],
                              "reason": "requested output"}], "unresolved": []})
        elif stage == "ta_sr":
            response = "res = df.where(users.name, 'Alice').select(users.name)"
        elif "plan_gen" in stage:
            response = '{"logical_steps":["Select user names"]}'
        elif "plan_merge" in stage:
            response = "1. Select user names"
        elif "pruning_sample" in stage:
            response = '{"obviously_irrelevant_columns":[],"obviously_irrelevant_tables":[]}'
        elif "selection_sample" in stage:
            response = '{"relevant_tables":["users"],"relevant_columns":[{"table":"users","columns":["name"]}]}'
        elif "summarize_init" in stage:
            response = '{"query_specific_content_analysis":"User names","table_functions":{"users":"target"}}'
        elif "exp_" in stage:
            response = "```sql\nSELECT name FROM users LIMIT 1\n```"
        elif "link_" in stage:
            response = '{"relevant":true,"relevant_columns":[{"column_name":"name","relevance_reason":"output"}],"table_summary":"names"}'
        elif "global_refine" in stage:
            response = '{"refined_schema":{"users":{"relevant_columns":["name"]}},"status":"[CONFIRM]"}'
        elif "Schema_links:" in text:
            response = "Schema_links: [users.name]"
        elif stage == "c3_table_recall":
            response = '{"tables":["users"]}'
        elif stage == "c3_column_recall":
            response = '{"users":["name"]}'
        elif stage == "schemagraphsql:linking":
            response = "src=users, dst=orders"
        elif stage == "ta-sql:linking":
            response = "SELECT users.name FROM users"
        elif stage == "pv-sql:linking":
            response = '{"action":"done","relevant_columns":{"users":["name"]}}'
        elif stage == "rsl-sql:linking":
            response = '{"tables":["users"],"columns":["users.name"],"sql":"SELECT name FROM users"}'
        else:
            raise AssertionError("Unscripted stage: "+stage)
        return SimpleNamespace(text=response)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name)/"toy.sqlite")
        con = sqlite3.connect(self.db)
        con.executescript("""
            CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE orders(id INTEGER PRIMARY KEY,user_id INTEGER,
                                amount REAL,FOREIGN KEY(user_id) REFERENCES users(id));
            INSERT INTO users VALUES(1,'Alice'),(2,'Bob');
            INSERT INTO orders VALUES(1,1,10),(2,1,20);
        """)
        con.close()
        self.schema = extract_schema(self.db)
        self.example = Example(0, "toy_0", "toy", "Show Alice's name", "",
                               "SELECT name FROM users WHERE name='Alice'", self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_pipeline_complete_contract_and_reads(self):
        result = run_pipeline(ScriptedLLM(), "toy_0", "toy", self.example.question, "", self.db)
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.model_calls, 6)
        self.assertEqual(result.reads, 2)
        self.assertEqual(result.probe_attempts, 1)
        self.assertEqual(result.explore_steps[0]["status"], "hit")
        self.assertEqual(run_query(self.db, result.pred_sql)["rows"], [("Alice",)])

    def test_full_matrix_frozen_artifacts_and_resume(self):
        records = run_suite("bird", Path(self.temp.name)/"out", no_eval=True,
            llm_factory=ScriptedLLM, examples=[self.example], schemas={"toy": self.schema})
        self.assertEqual(len(records), len(experiment_matrix()))
        self.assertTrue(all(r["status"] == "ok" for r in records),
                        [(r["experiment"], r.get("error")) for r in records if r["status"] != "ok"])
        by_name = {r["experiment"]: r for r in records}
        self.assertEqual(by_name["adaptive"]["upstream_hash"], by_name["policy-eig"]["upstream_hash"])
        self.assertEqual(by_name["adaptive"]["trajectory_hash"], by_name["no-downstream-beliefs"]["trajectory_hash"])
        def forbidden():
            raise AssertionError("Resume must not request a model")
        resumed = run_suite("bird", Path(self.temp.name)/"out", no_eval=True,
            llm_factory=forbidden, examples=[self.example], schemas={"toy": self.schema})
        self.assertEqual([r["fingerprint"] for r in records], [r["fingerprint"] for r in resumed])

    def test_failed_upstream_keeps_call_and_token_accounting(self):
        class InvalidRequirements(ScriptedLLM):
            def complete_messages_with_metadata(self, *args, **kwargs):
                result = super().complete_messages_with_metadata(*args, **kwargs)
                if kwargs.get("stage") == "requirements":
                    return SimpleNamespace(text='{"requirements":[]}')
                return result
        records = run_suite("bird", Path(self.temp.name)/"failed", no_eval=True,
            llm_factory=InvalidRequirements, examples=[self.example], schemas={"toy": self.schema},
            experiment_names=["adaptive"])
        self.assertEqual(records[0]["status"], "failure")
        self.assertEqual(records[0]["model_calls"], 1)
        self.assertEqual(records[0]["tokens"]["input_tokens"], 10)
        self.assertEqual(len(records[0]["model_trace"]), 1)

    def test_baselines_use_distinct_linking_procedures(self):
        for name, function in (BASELINES | ALIGNED_BASELINES).items():
            model = ScriptedLLM()
            record = function(model, self.schema, self.example.question, "", self.db)
            self.assertEqual(record["status"], "ok", (name, record["error"]))
            self.assertTrue(record["selected_columns"])
            if name.endswith("-aligned"):
                self.assertLessEqual(record["model_calls"], 6)
                self.assertLessEqual(record["reads"], 5)
            if name == "schemagraphsql":
                self.assertIn("orders", record["selected_tables"])
            if name == "ta-sql":
                self.assertEqual(record["selected_columns"], ["users.name"])

    def test_readonly_limits_and_parameter_safety(self):
        before = Path(self.db).read_bytes()
        for query in ("DELETE FROM users", "SELECT 1;SELECT 2",
                      "SELECT * FROM sqlite_master", "SELECT * FROM pragma_table_info('users')",
                      "SELECT load_extension('x')"):
            self.assertFalse(run_query(self.db, query)["ok"])
        value = "'; DROP TABLE users;--"
        result = run_query(self.db, "SELECT name FROM users WHERE name=?", params=(value,))
        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(before, Path(self.db).read_bytes())
        timeout = run_query(self.db,
            "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x", timeout=.01)
        self.assertEqual(timeout["status"], "timeout")

    def test_checkpointed_wal_snapshot_without_sidecars(self):
        con = sqlite3.connect(self.db)
        con.execute("PRAGMA journal_mode=WAL")
        con.close()
        self.assertEqual(Path(self.db).read_bytes()[18:20], b"\x02\x02")
        self.assertFalse(Path(self.db+"-wal").exists())
        result = run_query(self.db, "SELECT name FROM users ORDER BY id")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["rows"], [("Alice",), ("Bob",)])
        self.assertFalse(Path(self.db+"-wal").exists())
        self.assertFalse(Path(self.db+"-shm").exists())

    def test_read_budget_rejects_before_database_attempt(self):
        budget = Budget()
        database = SafeDB(self.db, self.schema, budget)
        for _ in range(5):
            database.execute("SELECT name FROM users")
        with self.assertRaises(BudgetExceeded):
            database.execute("SELECT name FROM users")
        self.assertEqual(budget.reads, 5)
        self.assertEqual(len(database.queries), 5)

    def test_reference_scopes_stars_and_subqueries(self):
        q = "WITH x AS (SELECT name FROM users) SELECT name FROM x"
        self.assertEqual(extract_reference(q, self.schema)["columns"], ["users.name"])
        self.assertEqual(extract_reference("SELECT COUNT(*) FROM users", self.schema)["columns"], ["users.id"])
        self.assertEqual(extract_reference("SELECT * FROM users", self.schema)["columns"], ["users.id", "users.name"])
        q = "SELECT u.name FROM users u WHERE u.id IN (SELECT user_id FROM orders)"
        self.assertIn(("orders.user_id", "users.id"), extract_reference(q, self.schema)["edges"])
        q = "SELECT u.name FROM users u WHERE EXISTS(SELECT 1 FROM orders u WHERE u.amount=10 AND u.user_id=1)"
        parsed = extract_reference(q, self.schema)
        self.assertIn("orders.amount", parsed["columns"])
        self.assertNotIn("users.amount", parsed["columns"])

    def test_contract_missing_coverage_and_strict_boolean(self):
        reqs = [Requirement("r1", "output", "name", ["users.name"])]
        raw = {"columns": ["users.id"], "coverage": [],
               "unresolved": [{"columns": ["orders.amount"], "retain": "true"}]}
        result = _apply_coverage_contract(raw, self.schema, reqs)
        self.assertIn("users.name", result.columns)
        self.assertNotIn("orders.amount", result.columns)
        self.assertEqual(result.coverage[0].state, "unresolved")
        schema = Schema("x", {"t": [Column("a", "INT", True), Column("b", "INT", True)]})
        self.assertNotIn("PRIMARY KEY", render_ddl(schema, CommitContract(["t.a"])))

    def test_malformed_likelihood_and_dedup(self):
        text = ScriptedLLM().complete("", "", stage="decision_model")
        raw = json.loads(text)
        raw["actions"].append({**raw["actions"][0], "id": "a2"})
        decisions, errors = parse_decisions_detailed(json.dumps(raw), self.schema)
        self.assertEqual(len(decisions[0].probes), 1)
        self.assertTrue(errors)
        raw["actions"] = raw["actions"][:1]
        raw["actions"][0]["likelihood"] = [[.2, .8]]
        decisions, errors = parse_decisions_detailed(json.dumps(raw), self.schema)
        self.assertEqual(len(decisions[0].probes), 0)
        self.assertTrue(any("dimension" in e for e in errors))

    def test_implication_filter_is_free(self):
        options = [Interpretation("a", "", .5), Interpretation("b", "", .5)]
        probes = [Probe("1", "d", "equals", "users", "name", "Alice", likelihoods={"a": .99, "b": .01}),
                  Probe("2", "d", "nonnull", "users", "name", likelihoods={"a": .99, "b": .01})]
        calls = []
        def runner(path, probe):
            calls.append(probe.id)
            return ProbeOutcome(probe.id, probe.kind, "hit", True)
        trace = explore(self.db, [Decision("d", "", options, probes)], policy="voi",
                        implication_filter=True, probe_runner=runner)
        self.assertEqual(calls, ["1"])
        self.assertEqual(trace.attempts, 1)
        self.assertTrue(trace.skips[0]["free"])

    def test_belief_removal_preserves_candidates_and_definitions(self):
        model = ScriptedLLM()
        upstream = prepare_upstream(model, self.schema, self.example.question, "", self.db)
        raw = encode_upstream(upstream)
        self.assertEqual(digest(raw), digest(encode_upstream(decode_upstream(raw))))
        g, decisions = upstream["ground"], upstream["decisions"]
        full = evidence_bundle(self.schema, g, decisions, {"d1": {"h1": .9}}, {"d1": "h1"}, [], True)
        stripped = evidence_bundle(self.schema, g, decisions, {}, {}, [], False)
        self.assertEqual(full["candidate_relations"], stripped["candidate_relations"])
        self.assertEqual(full["candidate_paths"], stripped["candidate_paths"])
        self.assertEqual(full["decisions"][0]["probes"], stripped["decisions"][0]["probes"])
        self.assertNotIn("prior", stripped["decisions"][0]["interpretations"][0])
        self.assertNotIn("posteriors", stripped)

    def test_perturbation_reuses_seed_and_preserves_bytes(self):
        before = Path(self.db).read_bytes()
        deleted, audit = visible_schema(self.schema, "D100")
        self.assertFalse(deleted.declared_fks)
        self.assertNotIn("FOREIGN KEY", deleted.to_ddl_text())
        first, a = visible_schema(self.schema, "N100", excluded=[("orders.id", "users.id")])
        second, b = visible_schema(self.schema, "N100", excluded=[("orders.id", "users.id")])
        self.assertEqual(a, b)
        self.assertNotIn(("orders.id", "users.id"), a["injected_edges"])
        self.assertEqual(before, Path(self.db).read_bytes())

    def test_metrics_empty_macro_and_statistics(self):
        self.assertEqual(prf(set(), set(), True), (1., 1., 1., 1.))
        self.assertEqual(prf({"a"}, set(), True), (0., 0., 0., 0.))
        metrics = score_structure({"selected_columns": ["users.id"],
            "pred_sql": "SELECT name FROM users", "selected_joins": []},
            "SELECT name FROM users", self.schema)
        self.assertTrue(metrics["out_of_selection"])
        self.assertEqual(metrics["NSR"], 0.)
        self.assertEqual(mcnemar({"1": 1}, {"1": 1})["p"], 1.)
        values = holm([{"p": .01}, {"p": .04}, {"p": .03}])
        self.assertEqual(values[0]["p_holm"], .03)

    def test_cached_calls_output_limit_and_submission_reserve(self):
        requests = []
        def create(**request):
            requests.append(request)
            return SimpleNamespace(id="response", model="fake",
                choices=[SimpleNamespace(message=SimpleNamespace(content="SELECT 1"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test-key"}), patch("gws.llm.OpenAI", return_value=client):
            model = LLM(response_cache=ResponseCache(str(Path(self.temp.name)/"cache.jsonl")))
            model.budget = Budget()
            for _ in range(5):
                model.complete("system", "question", stage="requirements")
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["max_tokens"], 4096)
            self.assertEqual(model.stats()["input_tokens"], 50)
            self.assertEqual(model.stats()["paid_input_tokens"], 10)
            with self.assertRaises(BudgetExceeded):
                model.complete("system", "question", stage="requirements")
            model.complete("system", "question", stage="generation:0")
            self.assertEqual(model.budget.calls, 6)
            self.assertNotIn("offline-test-key", (Path(self.temp.name)/"cache.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
