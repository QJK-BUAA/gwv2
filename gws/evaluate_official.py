"""Pinned official EX semantics in read-only, 30-second worker processes."""
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time
from . import config
from .sql_analysis import validate_sql
from .sqlite_access import read_only_uri


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_worker(pipe, dataset, sql, gold, db, index, evaluator):
    try:
        validate_sql(sql)
        original_connect = sqlite3.connect
        def readonly(database, *args, **kwargs):
            kwargs["uri"] = True
            con = original_connect(read_only_uri(database), *args, **kwargs)
            started = time.monotonic()
            con.set_progress_handler(lambda: int(time.monotonic()-started > 30), 1000)
            if hasattr(con, "enable_load_extension"):
                con.enable_load_extension(False)
            return con
        sqlite3.connect = readonly
        if dataset == "bird":
            sys.path.insert(0, str(Path(evaluator).parent))
            module = _load(evaluator, "official_bird")
            result = module.execute_model(sql, gold, db, index, 30)
            value = int(result.get("res", 0))
        else:
            sys.path.insert(0, evaluator)
            module = _load(str(Path(evaluator)/"exec_eval.py"), "official_spider")
            module.TIMEOUT = 30
            value = int(module.eval_exec_match(
                db=db, p_str=sql, g_str=gold, plug_value=False, keep_distinct=False,
                progress_bar_for_each_datapoint=False))
        pipe.send({"res": value})
    except Exception as exc:
        pipe.send({"res": 0, "evaluation_error": f"{type(exc).__name__}: {exc}"})
    finally:
        pipe.close()


def score_one(dataset, example, sql, status="ok"):
    if status != "ok" or not sql:
        return {"res": 0, "status": "non_submission"}
    evaluator = config.BIRD_EVAL_SCRIPT if dataset == "bird" else config.SPIDER1_TESTSUITE
    if not Path(evaluator).exists():
        raise FileNotFoundError("Official evaluator missing; run reproduce.py --prepare-only")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_score_worker, args=(
        child, dataset, sql, example.gold_sql, example.sqlite_path, example.idx, evaluator))
    process.start()
    child.close()
    process.join(config.EXEC_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(5)
        result = {"res": 0, "status": "timeout"}
    else:
        result = parent.recv() if parent.poll() else {"res": 0, "status": "worker_failure"}
    parent.close()
    return result


def evaluate(dataset, examples, pred_sql, workdir, statuses=None):
    if dataset not in {"bird", "spider1", "spider200"}:
        raise ValueError("Unknown evaluator")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    results, per_diff = [], {}
    for ex in examples:
        value = score_one(dataset, ex, pred_sql.get(ex.instance_id, ""),
                          (statuses or {}).get(ex.instance_id, "ok"))
        results.append({"instance_id": ex.instance_id, **value})
        per_diff.setdefault(ex.difficulty or "simple", []).append(value["res"])
    correct = sum(x["res"] for x in results)
    summary = {"n": len(examples), "correct": correct,
               "ex": round(100*correct/len(examples), 2) if examples else None,
               "per_question": results}
    summary.update({d: round(100*sum(xs)/len(xs), 2) for d, xs in per_diff.items()})
    return summary


evaluate_bird = lambda examples, pred_sql, workdir: evaluate("bird", examples, pred_sql, workdir)
evaluate_spider1 = lambda examples, pred_sql, workdir: evaluate("spider1", examples, pred_sql, workdir)
