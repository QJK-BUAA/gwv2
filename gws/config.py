"""Central configuration for the GraphWalkerSQL (ARR October 2026) reproduction.

Every default here is traced to the paper (main.tex / appendix.tex). Where the
paper gives an explicit number we use that number; where it names a rule we
implement the rule. Paths resolve against the surrounding workspace (``研二下``)
so the code runs on this machine without further setup, and every path/number
can be overridden with an environment variable of the same name.

Paper references (section / table):
  * Spider-200 fixed hash-selected subset ....... Sec. Setup / Appendix Tab. 1
  * deepseek-v4-flash, temp 0, seed 42, 4096 tok . Appendix Tab. 1
  * probe timeout 5s / eval timeout 30s ......... Appendix Tab. 1
  * structural/FK score 0.65, model-proposed 0.75 Appendix Tab. 1 / Sec. Ground
  * cost c(a)=1, threshold lambda=0.015 ......... Sec. EVSI / Appendix Tab. 7
  * active-probe budget 4 ....................... Sec. EVSI / Appendix Tab. 7
  * path caps: 3/pair, 12 overall, 3 edges,
    512 heap removals per pair .................. Sec. Ground / Appendix
  * uniqueness >= 0.95 structural eligibility ... Appendix Tab. 1
  * random/N100 seeds 42, 137, 911 .............. Appendix Tab. 1
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Workspace roots
# --------------------------------------------------------------------------- #
# .../研二下/gwv2/gws/config.py -> .../研二下
_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))


def _p(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


# --------------------------------------------------------------------------- #
# Dataset locations (reuse the extracted benchmarks already on this machine)
# --------------------------------------------------------------------------- #
_ITS = _p("论文复现", "Interactive-Text-to-SQL", "dataset", "_extracted")

# BIRD-Dev (declared FKs): 1,534 questions / 11 databases.
BIRD_ROOT = os.environ.get("GWV2_BIRD_ROOT", os.path.join(_ITS, "bird_dev", "dev_20240627"))
BIRD_DEV_JSON = os.path.join(BIRD_ROOT, "dev.json")
BIRD_DEV_GOLD = os.path.join(BIRD_ROOT, "dev.sql")
BIRD_DB_DIR = os.path.join(BIRD_ROOT, "dev_databases")
BIRD_TABLES_JSON = os.path.join(BIRD_ROOT, "dev_tables.json")

# Spider 1.0-Dev (declared FKs): 1,034 questions / 166 databases.
SPIDER1_ROOT = os.environ.get("GWV2_SPIDER1_ROOT", os.path.join(_ITS, "spider_data"))
SPIDER1_DEV_JSON = os.path.join(SPIDER1_ROOT, "dev.json")
SPIDER1_DEV_GOLD = os.path.join(SPIDER1_ROOT, "dev_gold.sql")
SPIDER1_DB_DIR = os.path.join(SPIDER1_ROOT, "database")
SPIDER1_TABLES = os.path.join(SPIDER1_ROOT, "tables.json")

# Spider-200: the paper's fixed hash-selected 200-question Spider 1.0-Dev subset.
SPIDER200_SIZE = int(os.environ.get("GWV2_SPIDER200_SIZE", "200"))
SPIDER200_SALT = os.environ.get("GWV2_SPIDER200_SALT", "graphwalkersql-spider200")

# --------------------------------------------------------------------------- #
# Official evaluators (reused, never a bespoke matcher)
# --------------------------------------------------------------------------- #
BIRD_EVAL_SCRIPT = _p("论文复现", "Interactive-Text-to-SQL", "evaluation",
                      "bird_evaluation_raw.py")
SPIDER1_TESTSUITE = _p("论文复现", "Interactive-Text-to-SQL", "dataset", "_external",
                       "test-suite-sql-eval")

# --------------------------------------------------------------------------- #
# Model backend (Appendix Table 1)
# --------------------------------------------------------------------------- #
# The paper requests ``deepseek-v4-flash`` (non-thinking) through a shared
# backend. On an endpoint without that alias, override GWV2_MODEL (e.g.
# ``deepseek-chat``). The LLM wrapper disables reasoning for v4/glm models.
DEFAULT_MODEL = os.environ.get("GWV2_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.environ.get("GWV2_BASE_URL", "https://api.deepseek.com")
DEFAULT_TEMPERATURE = float(os.environ.get("GWV2_TEMPERATURE", "0.0"))
DEFAULT_SEED = int(os.environ.get("GWV2_SEED", "42"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GWV2_MAX_OUTPUT_TOKENS", "4096"))

# --------------------------------------------------------------------------- #
# Ground: candidate graph and bounded path search
# --------------------------------------------------------------------------- #
# Source-based path-ranking priors (Appendix Tab. 1 / Sec. Ground).
SCORE_STRUCTURAL = float(os.environ.get("GWV2_SCORE_STRUCTURAL", "0.65"))
SCORE_VISIBLE_FK = float(os.environ.get("GWV2_SCORE_VISIBLE_FK", "0.65"))
SCORE_MODEL_PROPOSED = float(os.environ.get("GWV2_SCORE_MODEL", "0.75"))

# Structural-enumeration eligibility: PK, name ``id``, or uniqueness >= this.
STRUCTURAL_UNIQUENESS = float(os.environ.get("GWV2_STRUCTURAL_UNIQUENESS", "0.95"))
# Name-score filter for structural candidates (Appendix: scores < 0.55 dropped).
NAME_SCORE_MIN = float(os.environ.get("GWV2_NAME_SCORE_MIN", "0.55"))

# Bounded best-first search over simple paths (Sec. Ground / Appendix).
MAX_PATHS_PER_PAIR = int(os.environ.get("GWV2_MAX_PATHS_PER_PAIR", "3"))
MAX_PATHS_OVERALL = int(os.environ.get("GWV2_MAX_PATHS_OVERALL", "12"))
MAX_PATH_EDGES = int(os.environ.get("GWV2_MAX_PATH_EDGES", "3"))
MAX_HEAP_REMOVALS = int(os.environ.get("GWV2_MAX_HEAP_REMOVALS", "512"))

# --------------------------------------------------------------------------- #
# Explore: decision model, probes, and one-step EVSI
# --------------------------------------------------------------------------- #
MAX_DECISIONS = int(os.environ.get("GWV2_MAX_DECISIONS", "6"))       # <= 6 decisions
MAX_PROBE_ACTIONS = int(os.environ.get("GWV2_MAX_PROBE_ACTIONS", "12"))  # <= 12 probes
ACTIVE_PROBE_BUDGET = int(os.environ.get("GWV2_ACTIVE_PROBE_BUDGET", "4"))  # 4 attempts
PROBE_COST = float(os.environ.get("GWV2_PROBE_COST", "1.0"))        # c(a) = 1
LAMBDA_COST = float(os.environ.get("GWV2_LAMBDA_COST", "0.015"))    # lambda = 0.015
PROBE_TIMEOUT = float(os.environ.get("GWV2_PROBE_TIMEOUT", "5.0"))  # 5-second probes
# Observation with predicted probability <= this leaves the posterior unchanged.
MIN_OBS_PROB = float(os.environ.get("GWV2_MIN_OBS_PROB", "1e-12"))

# Random probe-policy seeds and N100 injection seeds (Appendix Tab. 1).
RANDOM_SEEDS = (42, 137, 911)
N100_SEEDS = (42, 137, 911)

# --------------------------------------------------------------------------- #
# Commit / final generation
# --------------------------------------------------------------------------- #
FINAL_SQL_CANDIDATES = int(os.environ.get("GWV2_FINAL_CANDIDATES", "1"))  # K = 1

# --------------------------------------------------------------------------- #
# Execution (Appendix Tab. 1)
# --------------------------------------------------------------------------- #
EXEC_TIMEOUT = float(os.environ.get("GWV2_EXEC_TIMEOUT", "30.0"))   # 30-second eval
EXEC_MAX_ROWS = int(os.environ.get("GWV2_EXEC_MAX_ROWS", "2000"))
