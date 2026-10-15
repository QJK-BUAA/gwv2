"""Frozen, resumable paper experiments; reference data is confined to scoring."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import csv
import json
import time
from baselines.methods import BASELINES, ALIGNED_BASELINES
from .artifacts import digest, file_digest, dump, encode_upstream, decode_upstream, source_fingerprint
from .commit import contract_payload
from .datasets import LOADERS, sample
from .evaluate_official import score_one
from .explore import ExploreTrace
from .llm import LLM
from .metrics import score_structure, extract_reference, load_annotations, macro, mcnemar, holm
from .perturbations import visible_schema
from .pipeline import AblationConfig, load_db, prepare_upstream, finish_upstream
from .response_cache import ResponseCache
from . import config


@dataclass
class Experiment:
    name: str
    suite: str
    method: str = "graphwalker"
    condition: str = "clean"
    perturbation_seed: int = 42
    ablation: AblationConfig = None
    frozen_adaptive: bool = False

    def __post_init__(self):
        if self.ablation is None:
            self.ablation = AblationConfig()


def experiment_matrix(suite="all"):
    rows = [Experiment("adaptive", "main")]
    rows += [Experiment(name, "main", method=name) for name in BASELINES]
    rows += [Experiment("declared-fk-only", "relations", ablation=AblationConfig(
        include_draft=False, include_structural=False, include_model=False))]
    for label, key in (("no-draft", "include_draft"), ("no-structural", "include_structural"),
                       ("no-model-relations", "include_model"), ("no-overlap", "use_value_overlap")):
        rows.append(Experiment(label, "relations", ablation=replace(AblationConfig(), **{key: False})))
    rows += [Experiment("policy-" + p, "policies", ablation=AblationConfig(probe_policy=p))
             for p in ("voi", "eig", "no-probe")]
    rows += [Experiment(f"random-{s}", "policies", ablation=AblationConfig(probe_policy="random", seed=s))
             for s in config.RANDOM_SEEDS]
    rows += [Experiment("no-downstream-beliefs", "mechanisms",
                        ablation=AblationConfig(include_beliefs=False), frozen_adaptive=True),
             Experiment("implication-filter", "mechanisms", ablation=AblationConfig(implication_filter=True))]
    rows += [Experiment(f"commit-{commit}-{catalog}", "commit", frozen_adaptive=True,
                        ablation=AblationConfig(coverage_contract=commit == "contract", catalog_mode=catalog))
             for commit in ("contract", "ordinary") for catalog in ("full", "trimmed")]
    rows += [Experiment(f"{m}-D100", "perturbations", method=m, condition="D100")
             for m in ("graphwalker", *BASELINES)]
    rows += [Experiment(f"{m}-N100-{s}", "perturbations", method=m, condition="N100", perturbation_seed=s)
             for m in ("graphwalker", *BASELINES) for s in config.N100_SEEDS]
    rows += [Experiment(m, "aligned", method=m) for m in ALIGNED_BASELINES]
    rows += [Experiment(f"lambda-{lam:.3f}", "thresholds", ablation=AblationConfig(lam=lam))
             for lam in (.005, .015, .030)]
    rows.append(Experiment("lambda-no-stop", "thresholds", ablation=AblationConfig(probe_policy="voi")))
    if suite == "all":
        return rows
    if suite == "tokens":
        return [r for r in rows if r.suite == "main"]
    selected = [r for r in rows if r.suite == suite]
    if not selected:
        raise ValueError("Unknown suite")
    # Every controlled suite includes the adaptive and relevant main controls.
    return [r for r in rows if r.suite == "main"] + selected


def _upstream_key(ex, schema, ablation, runtime):
    return digest({"question": ex.question, "evidence": ex.evidence, "schema": asdict(schema),
                   "draft": ablation.include_draft, "structural": ablation.include_structural,
                   "model_relations": ablation.include_model, "runtime": runtime})


def _combine_tokens(upstream, downstream, upstream_paid):
    result = dict(downstream)
    for key in ("num_calls", "input_tokens", "output_tokens"):
        result[key] = result.get(key, 0) + upstream.get(key, 0)
    if upstream_paid:
        for key in ("paid_calls", "paid_input_tokens", "paid_output_tokens"):
            result[key] = result.get(key, 0) + upstream.get(key, 0)
    result["frozen_upstream_calls"] = upstream.get("num_calls", 0)
    return result


def run_suite(dataset, outdir, suite="all", limit=None, sample_seed=42, model=None,
              no_eval=False, llm_factory=None, examples=None, schemas=None, experiment_names=None):
    root = Path(__file__).resolve().parent.parent
    outdir = Path(outdir)
    all_examples = examples if examples is not None else LOADERS[dataset]()
    chosen = sample(all_examples, limit, sample_seed)
    if not chosen:
        raise ValueError("No examples")
    schemas = schemas or {ex.db_id: load_db(ex.sqlite_path) for ex in all_examples}
    model = model or config.DEFAULT_MODEL
    runtime = {"source": source_fingerprint(root), "model": model, "model_seed": 42,
               "temperature": 0, "max_tokens": config.MAX_OUTPUT_TOKENS,
               "backend_hash": digest(config.DEFAULT_BASE_URL.rstrip("/")),
               "settings": {k: getattr(config, k) for k in (
                   "SCORE_STRUCTURAL", "SCORE_VISIBLE_FK", "SCORE_MODEL_PROPOSED",
                   "STRUCTURAL_UNIQUENESS", "NAME_SCORE_MIN", "MAX_PATHS_PER_PAIR",
                   "MAX_PATHS_OVERALL", "MAX_PATH_EDGES", "MAX_HEAP_REMOVALS",
                   "MAX_DECISIONS", "MAX_PROBE_ACTIONS", "PROBE_COST", "PROBE_TIMEOUT",
                   "MIN_OBS_PROB", "EXEC_TIMEOUT", "EXEC_MAX_ROWS")}}
    cache = ResponseCache(str(outdir/"responses.jsonl"))
    factory = llm_factory or (lambda: LLM(model=model, seed=42, response_cache=cache))
    annotations = load_annotations(str(Path(config.DATA_ROOT)/"annotations"/"bird_schema.json")) if dataset == "bird" else {}
    exclusions = {db: set() for db in schemas}
    for ex in all_examples:
        exclusions[ex.db_id].update(tuple(e) for e in extract_reference(ex.gold_sql, schemas[ex.db_id])["edges"])
    data_hash = digest([{"id": e.instance_id, "db_id": e.db_id, "question": e.question,
                         "evidence": e.evidence, "gold_hash": digest(e.gold_sql)}
                        for e in all_examples])
    db_hashes = {db: file_digest(next(e.sqlite_path for e in all_examples if e.db_id == db))
                 for db in schemas}
    matrix = experiment_matrix(suite)
    if experiment_names:
        unknown = set(experiment_names)-{e.name for e in matrix}
        if unknown:
            raise ValueError("Unknown configuration(s) in suite: "+", ".join(sorted(unknown)))
        matrix = [e for e in matrix if e.name in experiment_names]
    manifest = {"runtime": runtime, "data": data_hash, "databases": db_hashes,
                "dataset": dataset, "sample_seed": sample_seed, "requested_limit": limit,
                "ids": [e.instance_id for e in chosen], "population": len(all_examples),
                "experiments": [asdict(e) for e in matrix]}
    # Each record has its own fingerprint, so extending a suite/sample is safe.
    dump(outdir/dataset/"manifest.json", manifest)
    views, upstream_cache, adaptive_cache, records = {}, {}, {}, []
    for experiment in matrix:
        for index, ex in enumerate(chosen):
            base = schemas[ex.db_id]
            view_key = (ex.db_id, experiment.condition, experiment.perturbation_seed)
            if view_key not in views:
                views[view_key] = visible_schema(base, experiment.condition,
                    experiment.perturbation_seed, excluded=exclusions[ex.db_id])
            schema, perturbation = views[view_key]
            key = digest({"runtime": runtime, "data": data_hash, "db": db_hashes[ex.db_id],
                          "experiment": asdict(experiment), "id": ex.instance_id})
            path = outdir/dataset/"records"/experiment.name/(ex.instance_id+".json")
            if path.is_file():
                row = json.loads(path.read_text())
                if row.get("fingerprint") == key and ("ex" in row or no_eval):
                    records.append(row)
                    continue
            started = time.monotonic()
            llm = factory()
            upstream, up_llm, newly_paid = None, None, False
            row = {"instance_id": ex.instance_id, "db_id": ex.db_id, "dataset": dataset,
                   "experiment": experiment.name, "suite": experiment.suite,
                   "method": experiment.method, "condition": experiment.condition,
                   "perturbation_seed": experiment.perturbation_seed,
                   "fingerprint": key, "status": "failure", "pred_sql": "",
                   "perturbation": perturbation}
            try:
                if experiment.method != "graphwalker":
                    runner = {**BASELINES, **ALIGNED_BASELINES}[experiment.method]
                    row.update(runner(llm, schema, ex.question, ex.evidence, ex.sqlite_path))
                else:
                    up_key = _upstream_key(ex, schema, experiment.ablation, runtime)
                    upstream_path = outdir/dataset/"upstream"/(up_key+".json")
                    newly_paid = False
                    if up_key not in upstream_cache:
                        if upstream_path.is_file():
                            saved = json.loads(upstream_path.read_text())
                            if saved["fingerprint"] != up_key or digest(saved["artifact"]) != saved["artifact_hash"]:
                                raise ValueError("Corrupted frozen upstream artifact")
                            upstream_cache[up_key] = decode_upstream(saved["artifact"])
                        else:
                            up_llm = factory()
                            upstream = prepare_upstream(up_llm, schema, ex.question, ex.evidence,
                                                        ex.sqlite_path, experiment.ablation)
                            encoded = encode_upstream(upstream)
                            dump(upstream_path, {"fingerprint": up_key, "artifact_hash": digest(encoded),
                                                 "artifact": encoded})
                            upstream_cache[up_key] = upstream
                            newly_paid = True
                    upstream = upstream_cache[up_key]
                    frozen = None
                    trajectory_path = outdir/dataset/"trajectories"/(up_key+".json")
                    if experiment.frozen_adaptive:
                        if up_key not in adaptive_cache:
                            if trajectory_path.is_file():
                                saved = json.loads(trajectory_path.read_text())
                                if saved["upstream_hash"] != up_key or digest(saved["trace"]) != saved["trace_hash"]:
                                    raise ValueError("Corrupted frozen trajectory")
                                adaptive_cache[up_key] = ExploreTrace(**saved["trace"])
                            else:
                                # Obtain only adaptive observations; no redundant Commit/generation.
                                from .pipeline import _run_probes
                                from .budget import Budget
                                budget = Budget(calls=upstream["calls"], reads=upstream["reads"])
                                adaptive_cache[up_key] = _run_probes(
                                    ex.sqlite_path, copy.deepcopy(upstream["decisions"]),
                                    AblationConfig(), budget)
                        frozen = adaptive_cache[up_key]
                    g, decisions, trace, contract, pred = finish_upstream(
                        llm, schema, ex.question, ex.evidence, ex.sqlite_path,
                        upstream, experiment.ablation, frozen_trace=frozen)
                    if experiment.name == "adaptive":
                        adaptive_cache[up_key] = trace
                        dump(trajectory_path, {"upstream_hash": up_key,
                             "trace_hash": digest(asdict(trace)), "trace": asdict(trace)})
                    row.update(pred_sql=pred, status="ok" if pred else "failure",
                        error="" if pred else "Invalid final SQL; no fallback",
                        selected_columns=contract.columns, selected_joins=[asdict(j) for j in contract.joins],
                        contract=contract_payload(contract), upstream=encode_upstream(upstream),
                        upstream_hash=up_key, trajectory=asdict(trace),
                        trajectory_hash=digest(asdict(trace)), probe_attempts=trace.attempts,
                        reads=llm.budget.reads, model_calls=llm.budget.calls,
                        tokens=_combine_tokens(upstream["tokens"], llm.stats(), newly_paid),
                        model_trace=upstream["model_trace"]+llm.calls)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                active = up_llm if upstream is None and up_llm is not None else llm
                row.setdefault("model_calls", getattr(getattr(active, "budget", None), "calls", active.num_calls))
                row.setdefault("reads", getattr(getattr(active, "budget", None), "reads", 0))
                row.setdefault("tokens", _combine_tokens(upstream["tokens"], llm.stats(), newly_paid)
                               if upstream is not None else active.stats())
                row.setdefault("model_trace", upstream["model_trace"]+llm.calls
                               if upstream is not None else active.calls)
            row["seconds"] = time.monotonic()-started
            row["metrics"] = score_structure(row, ex.gold_sql, base, annotations.get(ex.instance_id),
                                              perturbation["injected_edges"])
            if not no_eval:
                row["evaluation"] = score_one(dataset, ex, row["pred_sql"], row["status"])
                row["ex"] = row["evaluation"]["res"]
            dump(path, row)
            records.append(row)
            print(f"[{dataset}/{experiment.name}] {index+1}/{len(chosen)} "
                  f"{ex.instance_id}: {row['status']}, calls={row.get('model_calls')}, "
                  f"reads={row.get('reads')}", flush=True)
    summarize(records, outdir/dataset, expected=len(chosen), population=len(all_examples))
    return records


def summarize(records, directory, expected=None, population=None):
    directory = Path(directory)
    groups = {}
    for row in records:
        groups.setdefault(row["experiment"], []).append(row)
    summaries, comparisons = [], []
    for name, rows in groups.items():
        first = rows[0]
        out = {"experiment": name, "suite": first["suite"], "method": first["method"],
               "condition": first["condition"], "perturbation_seed": first["perturbation_seed"],
               "n": len(rows), "expected": expected, "population": population,
               "complete_requested_sample": len(rows) == expected, "full_population": len(rows) == population,
               "failures": sum(r["status"] != "ok" for r in rows),
               "EX": 100*macro(rows, "ex") if any("ex" in r for r in rows) else None,
               "model_calls": macro(rows, "model_calls"), "reads": macro(rows, "reads"),
               "active_probes": macro(rows, "probe_attempts")}
        metrics = [r["metrics"] for r in rows]
        for key in ("T-EMR", "T-R", "T-F6", "C-SRR", "NSR", "NSP", "NSF", "#Cols",
                    "Edge-EMR", "Edge-P", "Edge-R", "Edge-F1", "Candidate-Edge-R", "Path-EMR",
                    "out_of_selection", "noise_adopted"):
            value = macro(metrics, key)
            out[key] = value*(1 if key == "#Cols" else 100) if value is not None else None
            out[key+"_n"] = sum(isinstance(m.get(key), (int, float)) for m in metrics)
        for key in ("Edge-EMR", "Edge-P", "Edge-R", "Edge-F1", "Path-EMR"):
            subset = [m for m in metrics if m["reference_has_joins"]]
            value = macro(subset, key)
            out[key+"-nonempty"] = 100*value if value is not None else None
            out[key+"-nonempty_n"] = len(subset)
        for key in ("input_tokens", "output_tokens", "paid_calls", "paid_input_tokens", "paid_output_tokens"):
            out[key] = macro([r.get("tokens", {}) for r in rows], key)
        for status, value in (("correct", 1), ("incorrect", 0)):
            subset = [r for r in rows if r.get("ex") == value
                      and isinstance(r["metrics"].get("out_of_selection"), bool)]
            out[f"out_of_selection_{status}_n"] = sum(r["metrics"]["out_of_selection"] for r in subset)
            out[f"classifiable_{status}_n"] = len(subset)
        summaries.append(out)
        if name != "adaptive" and first["condition"] == "clean" and "adaptive" in groups:
            primary = {r["instance_id"]: r["ex"] for r in groups["adaptive"] if "ex" in r}
            alternative = {r["instance_id"]: r["ex"] for r in rows if "ex" in r}
            if primary and alternative:
                comparisons.append({"comparison": "adaptive vs "+name, **mcnemar(primary, alternative)})
    dump(directory/"summary.json", summaries)
    dump(directory/"statistics.json", holm(comparisons))
    with (directory/"records.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, default=str)+"\n")
    if summaries:
        with (directory/"metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    # Table-ready records retain per-seed entries and explicit sample sizes.
    tables = {"1": "main", "2": "main", "3": "relations", "4": "policies",
              "5": "mechanisms", "6": "commit", "7": "perturbations", "8": "main",
              "9": "aligned", "11": "perturbations", "15": "thresholds"}
    paper_tables = {table: [s for s in summaries
         if s["suite"] == group or s["experiment"] == "adaptive"] for table, group in tables.items()}
    perturbation_means = []
    for method in ("graphwalker", *BASELINES):
        noise = [s for s in summaries if s["method"] == method and s["condition"] == "N100"]
        deleted = next((s for s in summaries if s["method"] == method and s["condition"] == "D100"), {})
        clean = next((s for s in summaries if s["method"] == method and s["suite"] == "main"), {})
        if noise:
            mean_ex, mean_adoption = macro(noise, "EX"), macro(noise, "noise_adopted")
            clean_ex = clean.get("EX")
            perturbation_means.append({"method": method, "seed_count": len(noise),
                "seeds": [s["perturbation_seed"] for s in noise], "Clean-EX": clean_ex,
                "D100-EX": deleted.get("EX"), "N100-mean-EX": mean_ex,
                "N100-mean-noise-adoption": mean_adoption,
                "Delta-D100": deleted.get("EX")-clean_ex if deleted.get("EX") is not None and clean_ex is not None else None,
                "Delta-N100": mean_ex-clean_ex if mean_ex is not None and clean_ex is not None else None})
    paper_tables["7_means"] = perturbation_means
    dump(directory/"paper_tables.json", paper_tables)
    return summaries
