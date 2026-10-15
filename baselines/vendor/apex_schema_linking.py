"""
Schema Linking Module for Text2SQL.

This module implements an iterative agentic schema linking workflow:
1. Logical Planning: Generate abstract logical steps based ONLY on the query.
2. Negative Pruning (Optional, Default OFF): Use the logical plan to remove 100% irrelevant columns.
3. Semantic Linking (One-time): Initial database understanding and query grounding.
4. Exploration (One-time): Agentic exploration of potentially relevant tables.
5. Global Refinement (Iterative):
   - Round 0: Synthesize exploration results into an initial schema.
   - Round 1..N: Pure gap analysis (Integrity/Connectivity) to recover missing columns without new exploration.
"""

import json
import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import concurrent.futures
import threading
import time
import argparse
import json_repair
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from pathlib import Path
from tqdm import tqdm

# Assuming these modules exist in your project
from chat import GPTChat
from sql import SqlEnv
from utils import extract_all_blocks, get_table_schema_info
from preprocess import generate_table_info

# --- GLOBAL HYPERPARAMETERS ---
PLAN_SAMPLES = 2              # Samples for logical planning
PRUNE_SAMPLES = 2             # Samples for negative pruning/selection (if enabled)
MAX_REFINE_ROUNDS = 2         # Max iterations for global refinement loop
DEFAULT_MAX_WORKERS = 1       # Default threads
PLAN_TEMP = 0.8               # Temperature for planning/pruning diversity
STD_TEMP = 0.2                # Temperature for standard deterministic tasks
DEBUG = False                 # Set to False to run full pipeline
FAIL_ON_ERROR = True          # Set to True to raise an error if any step fails

CORE_RULES = (
    "*** CRITICAL RULES ***\n"
    "1. **Strict Naming**: Use EXACT table/column names from schema. NEVER replace spaces with underscores.\n"
    "2. **Topic Relevance**: Focus on tables matching the query's Topic.\n"
    "3. **Keep Discriminators**: Always keep ID/Code/Name/Type/FK columns for relevant tables.\n"
    "4. **When Unsure, KEEP**: If a column might be relevant (even 1% chance), keep it.\n"
)

EXTENDED_RULES = CORE_RULES + (
    "5. **Identify ALL Potential Columns**: Do NOT try to decide which column is 'THE only one' for a step. Identify ALL potential columns that could be part of the reasoning process. For example,\n"
    "\t- DO NOT guess the type of an unspecified entity even you have some prior knowledge, e.g., if the query contains location entity like Riverside, then ALL columns related to location (e.g., County, District, etc.) should be kept.\n"
    "\t- If the query requires filtering entities by a specific attribute (e.g., account holders), then ALL columns related to that attribute should be kept; if counting distinct entities, then ALL columns related to entity identification (e.g., ID, Code, Name, Type, etc.) should be kept.\n"
    "\t- Pay attention to columns sharing similar names (xx_id, xx_id2, xx_id3).\n"
    "\t- Pay attention to Foreign Key columns. Sometimes we need other tables containing the same column name as a bridge or filter table.\n"
    "\t- If a column has no description but its name is highly related to the query, keep it for safety.\n"
    "6. **Table Caution**: If a table contains ANY relevant keys/IDs, be very careful about removing its other attributes unless they are clearly irrelevant to the topic.\n"
)

CRITICAL_RULES = EXTENDED_RULES

class SchemaLinking:
    """Schema Linking class for identifying relevant schema elements from queries."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = STD_TEMP, max_workers: int = DEFAULT_MAX_WORKERS, cache_dir: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
        
        self.io_lock = threading.Lock()
        
        self.cache_dir = cache_dir
    
    def _extract_and_parse_json(self, response: str) -> Dict[str, Any]:
        try:
            decoded_object = json_repair.repair_json(response, return_objects=True)
            return decoded_object
        except Exception as e:
            try:
                # Fallback: try to locate JSON-like content
                match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
                if match:
                    return json_repair.repair_json(match.group(1), return_objects=True)
            except:
                pass
            raise ValueError(f"Failed to parse JSON: {str(e)}")
    
    def _validate_refinement_output_format(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate and auto-fix the refinement output format.
        Auto-fixes common deviations to avoid unnecessary retries.
        return: (is_valid, error_message)
        """
        if not isinstance(result, dict):
            return False, "Result must be a dictionary"

        if "refined_schema" not in result:
            return False, "Missing required field 'refined_schema'"

        refined_schema = result["refined_schema"]
        if not isinstance(refined_schema, dict):
            return False, "'refined_schema' must be a dictionary"
        
        errors = []
        for table_name, table_data in refined_schema.items():
            # Auto-fix: model gave a list directly instead of {"relevant_columns": [...]}
            if isinstance(table_data, list):
                refined_schema[table_name] = {"relevant_columns": table_data}
                table_data = refined_schema[table_name]
            
            if not isinstance(table_data, dict):
                errors.append(f"The data of '{table_name}' must be a dictionary")
                continue
            
            if "relevant_columns" not in table_data:
                # Auto-fix: model might have used "columns" instead of "relevant_columns"
                if "columns" in table_data:
                    table_data["relevant_columns"] = table_data.pop("columns")
                else:
                    errors.append(f"Table '{table_name}' missing 'relevant_columns' field")
                    continue
            
            relevant_columns = table_data["relevant_columns"]
            if not isinstance(relevant_columns, list):
                errors.append(f"The 'relevant_columns' of {table_name} must be a list")
                continue
            
            for idx, col in enumerate(relevant_columns):
                # Auto-fix: model gave a string instead of {"column_name": "..."}
                if isinstance(col, str):
                    relevant_columns[idx] = {"column_name": col, "relevance_reason": ""}
                    continue
                if not isinstance(col, dict):
                    errors.append(f"The 'relevant_columns[{idx}]' of {table_name} has invalid type")
                    continue
                if "column_name" not in col:
                    # Auto-fix: model might have used "name" instead of "column_name"
                    if "name" in col:
                        col["column_name"] = col.pop("name")
                    else:
                        errors.append(f"The 'relevant_columns[{idx}]' of {table_name} missing 'column_name' field")
                if "relevance_reason" not in col:
                    col["relevance_reason"] = ""
        
        if "rejected_candidates" in result:
            if not isinstance(result["rejected_candidates"], list):
                errors.append("'rejected_candidates' must be a list.")
        
        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _get_cache_path(self, question_id: str, cache_dir: str) -> str:
        safe_question_id = str(question_id).replace('/', '_').replace('\\', '_')
        return os.path.join(cache_dir, f"schema_linking_cache_{safe_question_id}.json")
    
    def _retry_with_backoff(self, func: Callable, max_retries: int = 2, delay: float = 1.0, *args, **kwargs) -> Any:
        last_exception = None
        current_delay = delay
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(current_delay)
                    current_delay *= 2
        raise last_exception
    
    def _load_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        with self.io_lock:
            if not os.path.exists(cache_path):
                return None
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Schema Linking] Error loading cache: {e}")
                return None
    
    def _save_cache(self, cache_path: str, cache_data: Dict[str, Any]) -> None:
        import tempfile
        with self.io_lock:
            try:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                
                existing_data = None
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except Exception:
                        pass
                
                if existing_data:
                    final_data = existing_data
                    for key in cache_data:
                        if key in final_data and isinstance(final_data[key], dict) and isinstance(cache_data[key], dict):
                            final_data[key].update(cache_data[key])
                        else:
                            final_data[key] = cache_data[key]
                else:
                    final_data = cache_data
                
                temp_dir = cache_dir or '.'
                fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix='.json.tmp')
                
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        json.dump(final_data, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    os.replace(temp_path, cache_path)
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise e
            except Exception as e:
                print(f"[Schema Linking] Error saving cache: {e}")

    # =========================================================================
    # Step 0: Data Loading Helper
    # =========================================================================

    def _generate_tables_dict_from_db(self, sqlite_path: str, database_root_dir: str = "data/dev_databases") -> Dict[str, Dict[str, Any]]:
        sqlite_path_obj = Path(sqlite_path)
        db_id = sqlite_path_obj.parent.name
        schema_info = get_table_schema_info(db_id, database_root_dir)
        
        tables_dict = {}
        for table_name in sorted(schema_info.keys()):
            table_info_dict = generate_table_info(db_id, database_root_dir, table_name, schema_info, return_type="dict")
            if not table_info_dict:
                continue
            full_name = f"{db_id}.{table_name}"
            tables_dict[full_name] = table_info_dict
        return tables_dict

    # =========================================================================
    # Step 1: Logical Planning (Pure Abstraction, No Schema)
    # =========================================================================

    def _generate_logical_plan(self, question: str, question_id: str = "") -> str:
        """Generates an abstract logical plan based ONLY on the query."""
        # Check cache first
        cache_path = None
        if self.cache_dir and question_id:
            cache_path = self._get_cache_path(question_id, self.cache_dir)
            cache_data = self._load_cache(cache_path)
            if cache_data and "logical_plan" in cache_data:
                return cache_data["logical_plan"]

        prompt = (
            "*** TASK CONTEXT ***\n"
            "You are a Lead Data Architect. Your task is to break down the User Question into abstract logical steps needed to answer it.\n"
            "**IMPORTANT**: Do NOT reference specific table or column names yet. Focus purely on the logic (e.g., filter, join, count, aggregate).\n\n"
            
            "*** USER QUESTION ***\n"
            f"{question}\n\n"
            
            "*** OUTPUT FORMAT ***\n"
            "```json\n"
            "{\n"
            '  "logical_steps": [\n'
            '    "1. Identify [Entity]...",\n'
            '    "2. Filter where [Condition]...",\n'
            '    "3. Link [Entity A] to [Entity B]...",\n'
            '    "4. Calculate [Aggregation]..."\n'
            '  ]\n'
            "}\n"
            "```\n"
        )

        def _single_plan():
            client = GPTChat(self.model, temperature=PLAN_TEMP)
            client.init_messages()
            resp = client.get_model_response_txt(prompt, tag="plan_gen_sample")
            return self._extract_and_parse_json(resp)

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=PLAN_SAMPLES) as executor:
            futures = [executor.submit(_single_plan) for _ in range(PLAN_SAMPLES)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"[Schema Linking] Planning sample failed: {e}")
        
        # Check for empty plan if FAIL_ON_ERROR is set (although _generate_logical_plan handles empty results by returning string)
        if FAIL_ON_ERROR and not results:
             raise ValueError("Plan generation failed")

        # Merge Plans
        plans_text = ""
        for i, res in enumerate(results):
            steps = res.get("logical_steps", [])
            if isinstance(steps, list): steps = "\n".join(steps)
            plans_text += f"Draft Plan {i+1}:\n{steps}\n\n"
            
        merge_prompt = (
            "*** TASK CONTEXT ***\n"
            f"We have collected {len(results)} draft logical plans. Synthesize them into a single, comprehensive Master Logical Plan.\n"
            "Ensure the steps cover all conditions, filters, joins, and aggregations required.\n\n"
            
            f"*** USER QUESTION ***\n{question}\n\n"
            f"*** DRAFT PLANS ***\n{plans_text}"
            
            "Output just the steps as a numbered list."
        )
        
        summary_client = GPTChat(self.model, temperature=STD_TEMP)
        summary_client.init_messages()
        master_plan = summary_client.get_model_response_txt(merge_prompt, tag="plan_merge")

        if cache_path:
            self._save_cache(cache_path, {"logical_plan": master_plan})
        return master_plan

    # =========================================================================
    # Step 1.5: Negative Pruning + Positive Selection (Schema + Plan) - OPTIONAL
    # =========================================================================

    def _prune_schema_negative(self, question: str, logical_plan: str, tables_dict: Dict[str, Any], evidence: str = "", question_id: str = "") -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Uses the Logical Plan to simultaneously:
          - Prune (negative): identify 100% irrelevant columns to remove.
          - Select (positive): identify relevant columns to keep.
        Both tasks run with PRUNE_SAMPLES samples; pruning uses intersection (all agree),
        selection uses union (any sample votes for it).
        Returns: (removed_columns_list, selected_columns_list, filtered_tables_dict)
        """

        def filter_table(ori_table_dict, remove_list, select_list):
            """Apply remove/select lists to produce the filtered tables dict."""
            remove_set = {item.upper() for item in remove_list}
            select_set = {item.upper() for item in select_list}
            force_save = {'id', 'code', 'name', 'type'}

            ori_column_num = sum(len(t['columns']) for t in ori_table_dict.values())
            kept_num = ori_column_num - len(remove_list)

            filtered = {}
            for t_name, t_info in ori_table_dict.items():
                new_cols = []
                short_name = t_info['table_name']
                for col in t_info['columns']:
                    col_name_lower = col['name'].lower()
                    key       = f"{t_name}.{col['name']}".upper()
                    key_short = f"{short_name}.{col['name']}".upper()
                    desc      = col.get('description', '').lower()

                    # Always keep discriminator columns when schema is small enough
                    is_force_save = any(kw in col_name_lower or kw in desc for kw in force_save)
                    if is_force_save and kept_num < 200:
                        new_cols.append(col)
                        continue

                    if key not in remove_set and key_short not in remove_set:
                        new_cols.append(col)
                    elif key in select_set or key_short in select_set:
                        # Explicitly selected overrides removal
                        new_cols.append(col)

                if new_cols:
                    new_t_info = t_info.copy()
                    new_t_info['columns'] = new_cols
                    filtered[t_name] = new_t_info
            return filtered

        # --- Cache check ---
        cache_path = None
        if self.cache_dir and question_id:
            cache_path = self._get_cache_path(question_id, self.cache_dir)
            cached = self._load_cache(cache_path)
            if cached and "pruning" in cached:
                removed_cols  = cached.get("pruning",   {}).get("removed_columns",  [])
                selected_cols = cached.get("selection", {}).get("selected_columns", [])
                if "filtered_tables" not in cached:
                    filtered_dict = filter_table(tables_dict, removed_cols, selected_cols)
                    self._save_cache(cache_path, {"filtered_tables": filtered_dict})
                else:
                    filtered_dict = cached["filtered_tables"]
                return removed_cols, selected_cols, filtered_dict

        # --- Build schema string (with descriptions) ---
        schema_str = ""
        for full_name, info in tables_dict.items():
            short_name = info['table_name']
            col_parts = []
            for c in info['columns']:
                desc = c.get('description', '')
                col_str = c['name'] + (f": {desc}" if desc else "")
                col_parts.append(col_str)
            schema_str += f"Table: {short_name}\nColumns: {', '.join(col_parts)}\n\n"

        # Build a short_name -> full_name mapping for result normalisation
        short_to_full: Dict[str, str] = {}
        for full_name, info in tables_dict.items():
            short_to_full[info['table_name']] = full_name

        # ------------------------------------------------------------------ #
        # Task A – Negative Pruning prompt
        # ------------------------------------------------------------------ #
        pruning_output_format = (
            "```json\n"
            "{\n"
            '  "obviously_irrelevant_tables": ["table_unused_1"],\n'
            '  "obviously_irrelevant_columns": [\n'
            '    {"table": "t1", "columns": ["col_unused_1", "col_unused_2"]}\n'
            '  ]\n'
            "}\n"
            "```\n"
        )
        pruning_prompt = (
            "*** TASK CONTEXT ***\n"
            "You are a Lead Data Architect. You have a Logical Plan to answer a query.\n"
            "Your task: **Negative Pruning**. Identify database tables or columns that are "
            "**100% IRRELEVANT** to the plan.\n\n"

            "*** USER QUESTION ***\n"
            f"{question}\n\n"

            f"*** MASTER LOGICAL PLAN ***\n{logical_plan}\n\n"

            "*** FULL DATABASE SCHEMA ***\n"
            f"{schema_str}\n"

            f"*** EVIDENCE ***\n{evidence}\n\n"

            f"{CRITICAL_RULES}"

            "*** STRICT GUIDELINES ***\n"
            "1. **High Recall (Safety)**: If you are not sure about the relevance of a column, "
            "**KEEP IT**. Only remove it if you are 100% sure it is noise "
            "(e.g., removing 'hair_color' when asking about 'grades').\n"
            "   - If the column name is related to the query (even 1% chance), keep it. "
            "Check the description and sample values before deciding to remove.\n"
            "2. **Definition of Relevance**: Includes both Lexical Matching and Semantic Relatedness "
            "over column name and description.\n"
            "   - **Lexical**: If a word from the query appears in the name, it MUST be retained.\n"
            "   - **Semantic**: Keep columns conceptually related to the topic.\n"
            "   - **CRITICAL**: Do NOT remove discriminator columns such as `xxx_id`, `xxx_name`, "
            "`xxx_code`, or `xxx_type` if the table itself is kept.\n"
            "3. **Output Removal List**:\n"
            "   - **Tables**: If a whole table is irrelevant, list it in `obviously_irrelevant_tables`.\n"
            "   - **Columns**: If specific columns are noise, list them in `obviously_irrelevant_columns`.\n\n"

            "*** OUTPUT FORMAT ***\n"
            f"{pruning_output_format}"
        )

        # ------------------------------------------------------------------ #
        # Task B – Positive Selection prompt
        # ------------------------------------------------------------------ #
        selection_output_format = (
            "```json\n"
            "{\n"
            '  "relevant_tables": ["table_useful_1"],\n'
            '  "relevant_columns": [\n'
            '    {"table": "t1", "columns": ["col_useful_1", "col_pk_id"]}\n'
            '  ]\n'
            "}\n"
            "```\n"
        )
        selection_prompt = (
            "*** TASK CONTEXT ***\n"
            "You are a Lead Data Architect. You have a Logical Plan to answer a query.\n"
            "Your task: **Positive Selection**. Identify database tables or columns that are "
            "**RELEVANT** or **NECESSARY** to the plan.\n\n"

            "*** USER QUESTION ***\n"
            f"{question}\n\n"

            f"*** MASTER LOGICAL PLAN ***\n{logical_plan}\n\n"

            "*** FULL DATABASE SCHEMA ***\n"
            f"{schema_str}\n"

            f"*** EVIDENCE ***\n{evidence}\n\n"

            f"{CRITICAL_RULES}"

            "*** STRICT GUIDELINES ***\n"
            "1. **High Recall (Safety)**: Select ALL columns that might be useful for joining, "
            "filtering, grouping, or returning results. If unsure, **PICK IT**.\n"
            "2. **Definition of Relevance**: Includes both Lexical Matching and Semantic Relatedness.\n"
            "   - **Lexical**: If a word from the query appears in the table or column name, it MUST be selected.\n"
            "   - **Semantic**: Identify columns conceptually related to the topic.\n"
            "   - **Discriminators**: ALWAYS select primary keys and common identifiers "
            "(`xxx_id`, `xxx_code`, `xxx_name`) for relevant tables.\n"
            "3. **Output Selection List**:\n"
            "   - **Tables**: If a whole table is relevant, list it in `relevant_tables`.\n"
            "   - **Columns**: List specific useful columns in `relevant_columns`. "
            "If a table is already in `relevant_tables`, its columns can be omitted.\n\n"

            "*** OUTPUT FORMAT ***\n"
            f"{selection_output_format}"
        )

        # ------------------------------------------------------------------ #
        # Helper: single LLM call with format validation + retry
        # ------------------------------------------------------------------ #
        def _single_llm_call(prompt_text: str, mode: str):
            client = GPTChat(self.model, temperature=PLAN_TEMP)
            client.init_messages()
            json_example = pruning_output_format if mode == "pruning" else selection_output_format

            resp   = client.get_model_response_txt(prompt_text, tag=f"{mode}_sample")
            parsed = self._extract_and_parse_json(resp)

            def _is_valid(data):
                if not isinstance(data, dict):
                    return False
                if mode == "pruning":
                    cols = data.get("obviously_irrelevant_columns", [])
                else:
                    cols = data.get("relevant_columns", [])
                if not isinstance(cols, list):
                    return False
                for item in cols:
                    if not isinstance(item, dict):
                        return False
                    if "table" not in item or "columns" not in item:
                        return False
                    if not isinstance(item["columns"], list):
                        return False
                return True

            for attempt in range(2):
                if _is_valid(parsed):
                    return parsed
                feedback = (
                    f"Error: output format is invalid (attempt {attempt + 1}).\n"
                    f"Please fix and output ONLY valid JSON following:\n{json_example}"
                )
                resp   = client.get_model_response_txt(feedback, tag=f"{mode}_retry_{attempt}")
                parsed = self._extract_and_parse_json(resp)
            return parsed if isinstance(parsed, dict) else {}

        # ------------------------------------------------------------------ #
        # Run both tasks with PRUNE_SAMPLES parallel samples each
        # ------------------------------------------------------------------ #
        pruning_results   = []
        selection_results = []

        def _run_prune():   return _single_llm_call(pruning_prompt,   "pruning")
        def _run_select():  return _single_llm_call(selection_prompt, "selection")

        with concurrent.futures.ThreadPoolExecutor(max_workers=PRUNE_SAMPLES * 2) as executor:
            prune_futures  = [executor.submit(_run_prune)  for _ in range(PRUNE_SAMPLES)]
            select_futures = [executor.submit(_run_select) for _ in range(PRUNE_SAMPLES)]
            for f in concurrent.futures.as_completed(prune_futures):
                try:    pruning_results.append(f.result())
                except Exception as e: print(f"[Schema Linking] Pruning sample failed: {e}")
            for f in concurrent.futures.as_completed(select_futures):
                try:    selection_results.append(f.result())
                except Exception as e: print(f"[Schema Linking] Selection sample failed: {e}")

        if FAIL_ON_ERROR and not pruning_results:
            raise ValueError("No pruning results")

        # ------------------------------------------------------------------ #
        # Aggregate pruning results: INTERSECTION (all samples must agree)
        # ------------------------------------------------------------------ #
        import collections as _collections
        removal_votes: Dict[str, int] = _collections.defaultdict(int)
        sample_count_prune = len(pruning_results)

        for res in pruning_results:
            if not isinstance(res, dict): continue
            sample_keys: set = set()
            # Whole-table removal
            for short_table in res.get("obviously_irrelevant_tables", []):
                if not isinstance(short_table, str): continue
                full_name = short_to_full.get(short_table)
                if full_name and full_name in tables_dict:
                    for c in tables_dict[full_name]['columns']:
                        sample_keys.add(f"{full_name}.{c['name']}")
            # Column-level removal
            for item in res.get("obviously_irrelevant_columns", []):
                if not isinstance(item, dict): continue
                short_table = item.get("table", "")
                full_name   = short_to_full.get(short_table, short_table)
                cols = item.get("columns", [])
                if isinstance(cols, str): cols = [cols]
                for c in cols:
                    if isinstance(c, str):
                        sample_keys.add(f"{full_name}.{c}")
            for key in sample_keys:
                removal_votes[key] += 1

        # Only remove if ALL samples agreed (intersection)
        final_removal_list = [k for k, v in removal_votes.items() if v >= sample_count_prune]

        # ------------------------------------------------------------------ #
        # Aggregate selection results: UNION (any sample votes for it)
        # ------------------------------------------------------------------ #
        selection_votes: Dict[str, int] = _collections.defaultdict(int)

        for res in selection_results:
            if not isinstance(res, dict): continue
            sample_keys = set()
            # Whole-table selection
            for short_table in res.get("relevant_tables", []):
                if not isinstance(short_table, str): continue
                full_name = short_to_full.get(short_table)
                if full_name and full_name in tables_dict:
                    for c in tables_dict[full_name]['columns']:
                        sample_keys.add(f"{full_name}.{c['name']}")
            # Column-level selection
            for item in res.get("relevant_columns", []):
                if not isinstance(item, dict): continue
                short_table = item.get("table", "")
                full_name   = short_to_full.get(short_table, short_table)
                cols = item.get("columns", [])
                if isinstance(cols, str): cols = [cols]
                for c in cols:
                    if isinstance(c, str):
                        sample_keys.add(f"{full_name}.{c}")
            for key in sample_keys:
                selection_votes[key] += 1

        # Union: keep if at least 1 sample selected it
        final_selection_list = [k for k, v in selection_votes.items() if v >= 1]

        # ------------------------------------------------------------------ #
        # Build filtered dict
        # ------------------------------------------------------------------ #
        filtered_tables_dict = filter_table(tables_dict, final_removal_list, final_selection_list)

        # Save to cache
        if cache_path:
            self._save_cache(cache_path, {
                "pruning":   {"removed_columns": final_removal_list,  "removed_columns_count": len(final_removal_list)},
                "selection": {"selected_columns": final_selection_list, "selected_columns_count": len(final_selection_list)},
                "filtered_tables": filtered_tables_dict,
            })

        return final_removal_list, final_selection_list, filtered_tables_dict

    def plan_and_prune_initial(self, question: str, tables_dict: Dict[str, Any], question_id: str, evidence: str = "", enable_pruning: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Orchestrator for Plan and Prune steps."""
        # Step 1: Generate Plan (caching handled inside _generate_logical_plan)
        master_plan = self._generate_logical_plan(question, question_id)

        # Step 2: Prune + Select based on Plan (Optional)
        filtered_tables_dict = tables_dict
        if enable_pruning:
            _, _, filtered_tables_dict = self._prune_schema_negative(
                question, master_plan, tables_dict, evidence, question_id
            )

        return master_plan, filtered_tables_dict

    # =========================================================================
    # Step 2: Semantic Linking (One-Time Initialization)
    # =========================================================================
    
    def summarize_tables(self, 
                         tables_dict: Dict[str, Any], 
                         question: str, 
                         question_id: str, 
                         evidence: str = "", 
                         logical_plan: str = "",
                         chat_session: Optional[GPTChat] = None) -> Dict[str, Any]:
        """
        Initial Semantic Linking: Maps user query to database structure and defines the specific role of each table.
        This runs ONLY ONCE to establish the base understanding.
        """
        if not tables_dict:
            return {}
        
        cache_key = "table_summarization_init"
        cache_path = None
        if self.cache_dir:
            cache_path = self._get_cache_path(question_id, self.cache_dir)
            cache_data = self._load_cache(cache_path)
            if cache_data and cache_key in cache_data:
                return cache_data[cache_key]

        if chat_session is None:
            chat_session = GPTChat(self.model, temperature=self.temperature)

        table_details_str = ""
        for full_name, info in tables_dict.items():
            # [FIXED]: Added Type and Description here for better Semantic Mapping
            col_list = []
            for c in info['columns']:
                desc = c.get('description', '')
                col_str = f"{c['name']} ({c.get('type', '?')})"
                if desc: col_str += f": {desc}"
                col_list.append(col_str)
            col_info = "\n    ".join(col_list)
            table_details_str += f"- Table: {full_name}\n  Columns:\n    {col_info}\n\n"

        hint_section = f"*** HINT/EVIDENCE ***\n{evidence}\n\n" if evidence else ""
        plan_section = f"*** LOGICAL PLAN ***\n{logical_plan}\n\n" if logical_plan else ""
        
        prompt = (
            "*** TASK CONTEXT ***\n"
            "You are a Senior Data Architect. You have full visibility of the database schema and a user question.\n"
            "Your goal is to perform **Semantic Linking**: Analyze the database structure and how it grounds the user's intent.\n\n"
            
            f"{CRITICAL_RULES}"

            "*** USER QUESTION ***\n"
            f"{question}\n\n"
            
            f"{plan_section}"
            f"{hint_section}"
            
            "*** DATABASE SCHEMA ***\n"
            f"{table_details_str}"
            
            "*** YOUR TASKS ***\n"
            "1. **Query-Specific Content Analysis**: First understand the database structure (what domain it covers, how tables relate), "
            "then analyze the query against the available columns. Identify which tables/columns are likely targets, filters, or join keys.\n"
            "2. **Table Functional Analysis**: For EVERY potentially relevant table, describe its specific function regarding this query.\n"
            "   - Is it a **Target Table**? (Contains the answer columns)\n"
            "   - Is it a **Bridge Table**? (Doesn't have semantic data but is needed to join Table A and Table B via Foreign Keys)\n"
            "   - Is it a **Filtering Table**? (Contains columns for WHERE clauses)\n"
            "   - **CRITICAL**: A table may have multiple roles. If a table is needed as a BRIDGE, you MUST explicitly state that it connects Entity X and Entity Y, even if it looks empty of content.\n\n"
            
            "*** OUTPUT FORMAT ***\n"
            "```json\n"
            "{\n"
            '  "query_specific_content_analysis": "Database structure overview + detailed mapping of query terms to DB tables/columns/logic...",\n'
            '  "table_functions": {\n'
            '    "table_name_1": "Acts as a bridge table connecting Students and Classes via student_id and class_id.",\n'
            "    \"table_name_2\": \"Contains the 'score' column needed for calculation and 'exam_date' for filtering.\"\n"
            '  }\n'
            "}\n"
            "```\n"
            "Perform the semantic linking analysis:"
        )
        
        def _summarize_attempt():
            chat_session.init_messages()
            response = chat_session.get_model_response_txt(prompt, tag="summarize_init")
            return self._extract_and_parse_json(response)
        
        try:
            result = self._retry_with_backoff(_summarize_attempt, max_retries=2, delay=1.0)
            
            if cache_path:
                save_payload = {cache_key: result}
                self._save_cache(cache_path, save_payload)
            return result
        except Exception as e:
            print(f"[Schema Linking] Summarization failed: {e}")
            if FAIL_ON_ERROR:
                raise ValueError("Summarization failed")
            return {}

    # =========================================================================
    # Step 3: Exploration (One-Time)
    # =========================================================================

    def get_exploration_prompt(self, table_info: Dict[str, Any], question: str, summarization_result: Optional[Dict[str, Any]] = None, evidence: str = "") -> str:
        columns_str = "\n".join([f"- {c['name']} ({c['type']}): {c['description']}" for c in table_info['columns']])
        
        t_name = table_info.get('table_name', '')
        specific_function = ""
        if summarization_result and "table_functions" in summarization_result:
            funcs = summarization_result["table_functions"]
            if t_name in funcs:
                specific_function = funcs[t_name]
            else:
                for k, v in funcs.items():
                    if t_name in k or k in t_name:
                        specific_function = v
                        break
        
        function_context = ""
        if specific_function:
            function_context = f"*** ANTICIPATED ROLE ***\nThis table was identified as: {specific_function}\nUse this to guide your exploration.\n\n"

        hint_section = f"*** HINT ***\n{evidence}\n\n" if evidence else ""
        
        prompt = (
            "*** TASK CONTEXT ***\n"
            "You are an agent exploring a database table to verify its relevance to a user question.\n"
            "You must not explore randomly. You must verify if this table fits its anticipated role.\n\n"
            
            f"{CORE_RULES}"

            f"*** TARGET TABLE: {t_name} ***\n"
            f"Columns:\n{columns_str}\n\n"

            "*** USER QUESTION ***\n"
            f"{question}\n\n"
            
            f"{function_context}"
            f"{hint_section}"
            
            "*** YOUR MISSION ***\n"
            "Generate 3-8 SQLite queries to investigate. **Focus on understanding the table's semantics and utility.**\n\n"
            
            "**Motivation for Exploration**:\n"
            "1. **Semantic Alignment**: Check distinct values to understand what the column *means* versus what the query *needs*. (e.g., If column is 'type', does it contain the specific categories? If 'status', does it contain values like 'Active' or code '1'?)\n"
            "2. **Granularity & Scope**: Verify the table's grain (e.g., is it one row per Order or per Item?). This determines if it supports the required aggregations.\n"
            "3. **Bridge/Connectivity**: If this looks like a linking table, verify the Foreign Keys are populated (not all NULL) to ensure it can actually serve as a bridge.\n"
            "4. **Data Quality**: Are critical columns (targets for filters or answers) usable, or are they mostly NULL?\n\n"
            
            "*** OUTPUT FORMAT ***\n"
            "Provide SQL queries in a single `sql` block with comments explaining the *motivation*.\n"
            "```sql\n"
            "-- Motivation: Checking distinct values in 'status' to see if it aligns with the query's filter requirement\n"
            "SELECT DISTINCT status FROM table_name LIMIT 10;\n"
            "```\n"
            "Generate your exploration queries:"
        )
        return prompt

    def get_exploration_feedback_prompt(self, new_results: List[Dict]) -> str:
        if not new_results:
            return "*** FEEDBACK ***\nNo valid queries executed. Try simpler queries or `[COMPLETE]`."
        
        feedback_text = "*** NEW RESULTS ***\n"
        for i, res in enumerate(new_results, 1):
            feedback_text += f"SQL: {res['sql']}\nResult: {str(res['res'])[:300]}\n\n"
        
        feedback_text += (
            "Based on these results:\n"
            "1. Investigate further if ambiguity remains (e.g., verify more values, check nulls).\n"
            "2. Or output `[COMPLETE]` if you have sufficient evidence to judge relevance.\n"
        )
        return feedback_text
    
    def get_linking_prompt(self, table_name: str, table_info: Dict, exploration_results: List[Dict], question: str, evidence: str = "") -> str:
        expl_text = ""
        for i, res in enumerate(exploration_results, 1):
            expl_text += f"Query {i}: {res['sql']}\nResult: {str(res['res'])[:200]}\n\n"
        
        columns_detail = ""
        for col in table_info['columns']:
            desc = col.get('description', '')
            col_str = f"  - {col['name']} ({col.get('type', 'TEXT')})"
            if desc:
                col_str += f": {desc}"
            columns_detail += col_str + "\n"
            
        hint_section = f"*** HINT ***\n{evidence}\n\n" if evidence else ""
        
        prompt = (
            "*** TASK ***\n"
            f"Based on the exploration history and results above, determine if table `{table_name}` is RELEVANT to the User Question.\n\n"
            
            f"{CRITICAL_RULES}"

            "*** EXPLORATION EVIDENCE ***\n"
            f"{expl_text if expl_text else 'No exploration queries executed.'}\n"
            
            "*** USER QUESTION ***\n"
            f"{question}\n\n"
            
            f"{hint_section}"
            
            "*** DECISION GUIDELINES ***\n"
            "- **Direct Match**: Contains the specific answer data.\n"
            "- **Bridge Table**: Contains IDs needed to join other relevant tables (CRITICAL: Keep even if no other useful data).\n"
            "- **Filter Source**: Contains columns needed to restrict the result.\n"
            "- **Calculation Support**: Contains numerical columns needed for aggregation (e.g., `score` for AVG, `price` for SUM).\n\n"
            
            "*** OUTPUT GUIDELINES ***\n"
            "- `relevance_reason`: Explain the LOGICAL role (e.g., 'Provides the Join Key for X and Y', 'Contains the target column Z').\n"
            "- `observations`: Summarize FACTUAL findings from exploration (e.g., 'Column A contains integer codes 1-5', 'Table is empty').\n"
            "- `table_summary`: A concise summary of what this table represents in the context of the query.\n\n"
            
            "*** OUTPUT FORMAT ***\n"
            "```json\n"
            "{\n"
            '  "relevant": true/false,\n'
            '  "relevant_columns": [\n'
            '    {\n'
            '      "column_name": "name",\n'
            '      "relevance_reason": "...",\n'
            '      "observations": "..."\n'
            '    }\n'
            '  ],\n'
            '  "table_summary": "..."\n'
            "}\n"
            "```\n"
            "Provide your analysis:"
        )
        return prompt

    def explore_and_link_table(self, 
                               table_full_name: str, 
                               table_info: Dict[str, Any], 
                               sqlite_path: str, 
                               question: str, 
                               question_id: str, 
                               summarization_result: Optional[Dict[str, Any]] = None, 
                               evidence: str = "") -> Dict[str, Any]:
        """Executes exploration and linking for a SINGLE table."""
        
        # [FIXED] Reading from cache: Check inside the 'table_exploration' dict
        cache_path = None
        if self.cache_dir:
            cache_path = self._get_cache_path(question_id, self.cache_dir)
            cache_data = self._load_cache(cache_path)
            if cache_data and "table_exploration" in cache_data:
                if table_full_name in cache_data["table_exploration"]:
                    return cache_data["table_exploration"][table_full_name]
        
        # Thread-safe client instantiation (local to thread)
        chat_session = GPTChat(self.model, temperature=self.temperature)
        
        sql_env = SqlEnv()
        sql_env.start_db_sqlite(sqlite_path)
        
        result_payload = {
            "table_full_name": table_full_name,
            "relevant": False,
            "relevant_columns": [],
            "table_summary": "",
            "exploration_log": []
        }

        try:
            chat_session.init_messages()
            explore_prompt = self.get_exploration_prompt(table_info, question, summarization_result, evidence)
            exploration_results = []
            max_rounds = 2
            
            round_results_prev = [] 
            for round_idx in range(max_rounds):
                def _explore_attempt():
                    if round_idx == 0:
                        response = chat_session.get_model_response_txt(explore_prompt, tag=f"exp_{table_info['table_name']}_r0")
                    else:
                        response = chat_session.get_model_response_txt(self.get_exploration_feedback_prompt(round_results_prev), tag=f"exp_{table_info['table_name']}_r{round_idx}")
                    code_blocks = extract_all_blocks(response, 'sql')
                    if code_blocks == []:
                        return response
                    return code_blocks
                try:
                    response = self._retry_with_backoff(_explore_attempt, max_retries=2, delay=1.0)
                except Exception:
                    break

                if isinstance(response, str):
                    if "[COMPLETE]" in response:
                        break
                    else:
                        response = []
                
                sqls = response if isinstance(response, list) else [response]
                if not sqls:
                    break
                
                round_results_prev = []
                for sql in sqls:
                    if not sql or not sql.strip(): continue
                    clean_sql = re.sub(r'^```sql\s*|```\s*$', '', sql.strip(), flags=re.IGNORECASE|re.MULTILINE)
                    clean_sql = "\n".join([l for l in clean_sql.split('\n') if not l.strip().startswith('--')])
                    try:
                        res = sql_env.execute_sql_api(clean_sql, sqlite_path, max_len=1000)
                        if isinstance(res, str) and "##ERROR##" not in res:
                            round_results_prev.append({"sql": clean_sql, "res": res})
                    except Exception:
                        continue
                
                if not round_results_prev:
                    break
                exploration_results.extend(round_results_prev)
            
            result_payload["exploration_log"] = exploration_results

            # Linking Step
            try:
                if exploration_results:
                    linking_followup = (
                        f"Based on our exploration above, determine if table `{table_full_name}` is RELEVANT to the User Question.\n"
                        "Consider: Direct Match, Bridge Table (join keys), Filter Source, Calculation Support.\n\n"
                        "Output JSON:\n"
                        "```json\n"
                        "{\n"
                        '  "relevant": true/false,\n'
                        '  "relevant_columns": [\n'
                        '    {"column_name": "...", "relevance_reason": "...", "observations": "..."}\n'
                        '  ],\n'
                        '  "table_summary": "..."\n'
                        "}\n"
                        "```\n"
                    )
                else:
                    linking_followup = self.get_linking_prompt(table_full_name, table_info, exploration_results, question, evidence)

                def _link_attempt():
                    resp = chat_session.get_model_response_txt(linking_followup, tag=f"link_{table_info['table_name']}")
                    return self._extract_and_parse_json(resp)
                    
                parsed_link = self._retry_with_backoff(_link_attempt, max_retries=2, delay=1.0)
                result_payload["relevant"] = parsed_link.get("relevant", False)
                result_payload["relevant_columns"] = parsed_link.get("relevant_columns", [])
                result_payload["table_summary"] = parsed_link.get("table_summary", "")
                if not result_payload["relevant"]:
                    result_payload["relevant_columns"] = []
            except Exception as e:
                print(f"[Schema Linking] Linking failed for {table_full_name}: {e}")
                if FAIL_ON_ERROR:
                    raise ValueError("Linking failed")
                result_payload["relevant"] = False

        except Exception as e:
            print(f"[Schema Linking] Error processing {table_full_name}: {e}")
            if FAIL_ON_ERROR:
                raise ValueError("Error processing table")
            result_payload["error"] = str(e)
        finally:
            sql_env.close_db()
        
        # [FIXED] Saving to cache: Nested structure under 'table_exploration'
        if cache_path:
            save_payload = {
                "table_exploration": {
                    table_full_name: result_payload
                }
            }
            self._save_cache(cache_path, save_payload)
        
        return result_payload

    # =========================================================================
    # Step 4: Global Refinement (Iterative & Incremental)
    # =========================================================================

    def refine_schema_globally(self, 
                               tables_dict: Dict[str, Any],
                               linking_results: Dict[str, Any], 
                               question: str, 
                               sqlite_path: str,
                               question_id: str,
                               iteration_round: int = 0,
                               summarization_result: Optional[Dict[str, Any]] = None,
                               evidence: str = "",
                               confirmed_columns: Set[str] = None,
                               chat_session: Optional[GPTChat] = None) -> Dict[str, Any]:
        """
        Refine schema iteratively.
        Round 0: Initial consolidation from exploration (full prompt, new session).
        Round 1..N: Gap analysis — reuses r0 chat_session if available for incremental context.
        """
        cache_key = f"global_refinement_r{iteration_round}"
        cache_path = None
        if self.cache_dir:
            cache_path = self._get_cache_path(question_id, self.cache_dir)
            cache_data = self._load_cache(cache_path)
            if cache_data and cache_key in cache_data:
                return cache_data[cache_key]
        
        use_incremental = (iteration_round > 0 and chat_session is not None and len(chat_session.messages) > 0)
        
        if chat_session is None:
            chat_session = GPTChat(self.model, temperature=self.temperature)

        output_format = (
            "*** OUTPUT FORMAT ***\n"
            "```json\n"
            "{\n"
            '  "refined_schema": { "table_name": { "relevant_columns": [{"column_name": "...", "relevance_reason": "Functional reason (e.g., Needed for Filtering)"}] } },\n'
            '  "rejected_candidates": ["table.column", "table.column"],\n'
            '  "exploration_queries": ["SELECT 1 FROM t1 JOIN t2 ON t1.id=t2.id LIMIT 1"],\n'
            '  "status": "EXPLORING" or "[CONFIRM]"\n'
            "}\n"
            "```\n"
        )

        if use_incremental:
            confirmed_list_str = ""
            if confirmed_columns:
                confirmed_list_str = "\n".join(f"- {item}" for item in sorted(confirmed_columns))
            else:
                confirmed_list_str = "(None)"
            
            candidates_str = ""
            for full_name, info in tables_dict.items():
                short_name = info['table_name']
                cands = []
                for c in info['columns']:
                    key_full = f"{full_name}.{c['name']}"
                    key_short = f"{short_name}.{c['name']}"
                    if not (confirmed_columns and (key_full in confirmed_columns or key_short in confirmed_columns)):
                        desc = c.get('description', '')
                        col_str = f"{c['name']} ({c.get('type', '?')})"
                        if desc: col_str += f": {desc}"
                        cands.append(col_str)
                if cands:
                    candidates_str += f"Table: {short_name}\n  Candidates: {'; '.join(cands)}\n"

            prompt = (
                "Now perform **GAP ANALYSIS** on the schema you just produced.\n"
                "Do the confirmed columns form a connected graph? Are any join keys or IDs missing?\n\n"
                f"*** CURRENTLY CONFIRMED COLUMNS ***\n{confirmed_list_str}\n\n"
                f"*** REMAINING CANDIDATE COLUMNS ***\n{candidates_str}\n\n"
                "Recover any missing Foreign Keys, IDs, or Bridge Tables from the candidates.\n"
                "If multiple columns serve the same purpose, KEEP ALL.\n\n"
                f"{output_format}"
            )
        else:
            schema_status_str = ""
            
            if iteration_round == 0:
                all_tables = sorted(tables_dict.keys())
                for t_name in all_tables:
                    status = "[NOT EXPLORED]"
                    details = ""
                    if t_name in linking_results:
                        res = linking_results[t_name]
                        if res.get("relevant", False):
                            status = "[MARKED RELEVANT]"
                            details = f"  Summary: {res.get('table_summary')}\n"
                            for c in res.get("relevant_columns", []):
                                details += f"  - {c['column_name']}: {c.get('relevance_reason')} (Obs: {c.get('observations')})\n"
                        else:
                            status = "[MARKED IRRELEVANT]"
                            details = f"  Reason: {res.get('table_summary')}\n"
                    
                    col_list = []
                    for c in tables_dict[t_name]['columns']:
                        desc = c.get('description', '')
                        col_str = f"{c['name']} ({c.get('type', '?')})"
                        if desc: col_str += f": {desc}"
                        col_list.append(col_str)
                    col_sig = ", ".join(col_list)
                    schema_status_str += f"Table: {t_name} {status}\n  Columns: {col_sig}\n{details}\n"
            else:
                schema_status_str += "*** CURRENTLY CONFIRMED COLUMNS ***\n"
                if confirmed_columns:
                    for item in sorted(list(confirmed_columns)):
                        schema_status_str += f"- {item}\n"
                else:
                    schema_status_str += "(None)\n"
                
                schema_status_str += "\n*** REMAINING CANDIDATE COLUMNS (Analyze for Gaps) ***\n"
                for full_name, info in tables_dict.items():
                    short_name = info['table_name']
                    candidates = []
                    for c in info['columns']:
                        key_full = f"{full_name}.{c['name']}"
                        key_short = f"{short_name}.{c['name']}"
                        if not (confirmed_columns and (key_full in confirmed_columns or key_short in confirmed_columns)):
                            desc = c.get('description', '')
                            col_str = f"{c['name']} ({c.get('type', '?')})"
                            if desc: col_str += f": {desc}"
                            candidates.append(col_str)
                    if candidates:
                        schema_status_str += f"Table: {short_name}\n  Candidates: {'; '.join(candidates)}\n"

            summary_context = ""
            if summarization_result:
                if summarization_result.get("query_specific_content_analysis"):
                    summary_context += f"Initial Analysis: {summarization_result['query_specific_content_analysis']}\n"
                if summarization_result.get("table_functions"):
                    summary_context += "Anticipated Roles:\n"
                    for t, r in summarization_result["table_functions"].items():
                        summary_context += f"- {t}: {r}\n"

            hint_section = f"*** HINT ***\n{evidence}\n\n" if evidence else ""
            
            if iteration_round == 0:
                task_intro = (
                    "You are the Lead Data Architect. We are synthesizing initial exploration findings.\n"
                    "Review the [MARKED RELEVANT] and [MARKED IRRELEVANT] tables. Fix blind spots."
                )
            else:
                task_intro = (
                    "You are the Lead Data Architect. We are performing **GAP ANALYSIS**.\n"
                    "Look at the `Currently Confirmed Columns`. Do they form a complete graph? Are keys missing? Can we use other columns to construct another solution path?\n"
                    "Search the `Remaining Candidate Columns` to recover missing IDs, Foreign Keys, or Bridge Tables.\n"
                    "Note: If a column is NOT listed in the CONFIRMED section, it is available in the CANDIDATES list."
                )

            prompt = (
                "*** TASK CONTEXT ***\n"
                f"{task_intro}\n\n"
                
                f"{CRITICAL_RULES}"
                
                "*** USER QUESTION ***\n"
                f"{question}\n\n"
                
                f"{hint_section}"
                f"*** SEMANTIC ANALYSIS ***\n{summary_context}\n\n"
                
                "*** SCHEMA STATUS ***\n"
                f"{schema_status_str}\n"
                
                "*** YOUR MISSION ***\n"
                "Determine the final list of columns required to write the SQL query. "
                "You must ensure the selected columns form a connected graph (tables can be joined) and cover all functional requirements of the query.\n\n"
                
                "*** SELECTION CRITERIA (FUNCTIONALITY) ***\n"
                "Keep a column if it serves one of the following purposes:\n"
                "1. **Identification**: Unique identifiers (IDs, Codes) needed to count or distinguish entities (Primary keys).\n"
                "2. **Linking**: Columns needed to join two tables together (Foreign Keys).\n"
                "3. **Filtering**: Columns involved in conditions (e.g., status='Active', date > 2023).\n"
                "4. **Aggregation**: Numerical columns for calculations (Sum, Avg, Max, Min).\n"
                "5. **Grouping & Sorting**: Columns used for 'GROUP BY' or 'ORDER BY'.\n"
                "6. **Direct Result**: Columns explicitly requested in the output.\n\n"
                "**Note on Multi-Path**: If multiple columns might serve the same purpose, KEEP ALL OF THEM. Alternative columns might help to construct another solution paths.\n"
                "**Note on Type of Entity**: DO NOT guess the type of an unspecified entity even you have some prior knowledge, e.g., if the query contains location entity like 'Riverside', then ALL columns related to location (e.g., County, District, etc.) should be kept. Another example is 'Fresno County Office of Education' which is actually a full name of a district.\n\n"
                
                "*** INTERACTIVE PROCESS ***\n"
                f"You can perform up to {MAX_REFINE_ROUNDS} rounds of verification.\n"
                "- To EXPLORE: Output `exploration_queries` in JSON to test joins or content.\n"
                "- To FINISH: Output `[CONFIRM]` in the JSON (or just output the final refined_schema without queries).\n\n"
                
                f"{output_format}"
                "Begin refinement:"
            )

        final_result_data = {}
        temp_env = None
        if sqlite_path and os.path.exists(sqlite_path):
            temp_env = SqlEnv()
            temp_env.start_db_sqlite(sqlite_path)

        try:
            if not use_incremental:
                chat_session.init_messages()
            
            round_idx = 0
            max_refine_rounds = 2
            current_prompt = prompt
            format_retry_count = 0
            max_format_retries = 2  # 每个round最多重试2次格式检查
            
            while round_idx < max_refine_rounds:
                def _refine_attempt():
                    resp = chat_session.get_model_response_txt(current_prompt, tag=f"global_refine_r{iteration_round}_s{round_idx}_f{format_retry_count}")
                    return self._extract_and_parse_json(resp)
                
                try:
                    result = self._retry_with_backoff(_refine_attempt, max_retries=2, delay=1.0)
                except Exception as e:
                    print(f"[Schema Linking] Refinement sub-round {round_idx} failed: {e}")
                    if FAIL_ON_ERROR:
                        raise ValueError("Refinement sub-round failed")
                    break
                
                # 格式检查
                is_valid, error_msg = self._validate_refinement_output_format(result)
                if not is_valid:
                    format_retry_count += 1
                    if format_retry_count > max_format_retries:
                        # print(f"[Schema Linking] Format validation failed after {max_format_retries} retries in round {round_idx}.")
                        if FAIL_ON_ERROR:
                            raise ValueError("Format validation failed")
                        final_result_data = result
                        break
                    
                    print(f"[Schema Linking] Format validation failed in round {round_idx} (retry {format_retry_count}/{max_format_retries}): {error_msg}")
                    # 构造格式反馈prompt
                    format_feedback = (
                        "*** FORMAT ERROR DETECTED ***\n"
                        f"Your output format is not correct. Error message: {error_msg}\n\n"
                        f'{output_format}'  
                        "Please provide the JSON output in the correct format."
                    )
                    current_prompt = format_feedback
                    continue  # 重新尝试，不执行SQL
                
                # 格式检查通过，重置格式重试计数
                format_retry_count = 0
                final_result_data = result
                
                status = result.get("status", "")
                queries = result.get("exploration_queries", [])
                
                if "[CONFIRM]" in status or not queries or not temp_env:
                    break
                
                extra_evidence = ""
                for sql in queries:
                    clean_sql = re.sub(r'^```sql\s*|```\s*$', '', sql.strip(), flags=re.IGNORECASE|re.MULTILINE)
                    try:
                        res = temp_env.execute_sql_api(clean_sql, sqlite_path, max_len=500)
                        extra_evidence += f"Query: {clean_sql}\nResult: {str(res)}\n\n"
                    except Exception as e:
                        extra_evidence += f"Query: {clean_sql}\nError: {str(e)}\n\n"
                
                current_prompt = (
                    "*** NEW EVIDENCE (Cross-Table Verification) ***\n"
                    f"{extra_evidence}\n\n"
                    "Based on this, please provide the updated JSON. If satisfied, mark status as `[CONFIRM]`."
                )
                round_idx += 1

        except Exception as e:
            print(f"[Schema Linking] Refinement loop error: {e}")
            if FAIL_ON_ERROR:
                raise ValueError("Refinement loop failed")
            if iteration_round == 0:
                fallback_schema = {}
                for t, data in linking_results.items():
                    if data.get("relevant"):
                        fallback_schema[t] = {"relevant_columns": data.get("relevant_columns", [])}
                final_result_data = {"refined_schema": fallback_schema}
            else:
                final_result_data = {"refined_schema": {}}
        
        finally:
            if temp_env:
                temp_env.close_db()
        
        # --- POST-PROCESSING: AUTO-PRESERVE POTENTIAL FOREIGN KEYS ---
        # Logic: If a column is selected in one table, check ALL other tables. 
        # If another table has a column with the SAME NAME (case-insensitive), keep it.
        try:
            current_schema = final_result_data.get("refined_schema", {})
            # Use a set of lower-cased names for matching to handle casing inconsistencies (e.g., "ID" vs "id")
            selected_col_names_lower = set()
            
            # Build a mapping from both short_name and full_name to full_name
            name_to_full = {}
            for full_name, info in tables_dict.items():
                short_name = info['table_name']
                name_to_full[short_name] = full_name
                name_to_full[full_name] = full_name
            
            # Normalize current_schema keys to full names
            normalized_schema = {}
            for key, value in current_schema.items():
                normalized_key = name_to_full.get(key, key)
                normalized_schema[normalized_key] = value
            
            # 1. Collect all currently selected column names (normalized to lower case)
            for t_data in normalized_schema.values():
                for col in t_data.get("relevant_columns", []):
                    if col.get("column_name"):
                        selected_col_names_lower.add(col["column_name"].lower())
            
            # 2. Scan all tables for matches
            for t_name, t_info in tables_dict.items():
                # Get set of already selected columns for this specific table (lower case)
                existing_table_cols_lower = set()
                if t_name in normalized_schema:
                    existing_table_cols_lower = {
                        c["column_name"].lower() 
                        for c in normalized_schema[t_name].get("relevant_columns", []) 
                        if c.get("column_name")
                    }
                
                # Check each column in the table schema
                for col_def in t_info["columns"]:
                    c_name = col_def["name"]
                    c_name_lower = c_name.lower()
                    
                    # Match against selected names (case-insensitive) AND check if not already present
                    if c_name_lower in selected_col_names_lower and c_name_lower not in existing_table_cols_lower:
                        # Ensure table structure exists
                        if t_name not in normalized_schema:
                            normalized_schema[t_name] = {"relevant_columns": []}
                        
                        # Add the column (using ORIGINAL casing from schema)
                        normalized_schema[t_name]["relevant_columns"].append({
                            "column_name": c_name,
                            "relevance_reason": "Auto-preserved: Shares name with a selected column (potential Foreign Key).",
                            "observations": "Automatically added for graph connectivity."
                        })
            
            # Update the result data with normalized schema
            final_result_data["refined_schema"] = normalized_schema


        except Exception as e:
            print(f"[Schema Linking] Auto-preserve FK error: {e}")
            if FAIL_ON_ERROR:
                raise ValueError("Auto-preserve FK error")

        if cache_path:
            save_payload = {cache_key: final_result_data}
            self._save_cache(cache_path, save_payload)
        
        final_result_data.pop("exploration_queries", None)
        final_result_data.pop("status", None)
        return final_result_data

    # =========================================================================
    # MAIN ORCHESTRATOR
    # =========================================================================

    def run_schema_linking(self, 
                           question: str, 
                           question_id: str,
                           evidence: str = "", 
                           sqlite_path: str = "",
                           database_root_dir: str = "data/dev_databases",
                           enable_pruning: bool = False) -> Dict[str, Any]:
        if not sqlite_path:
            raise ValueError("sqlite_path must be provided")
        
        # 1. Load Full Schema
        tables_dict = self._generate_tables_dict_from_db(sqlite_path, database_root_dir)

        # 2. Plan & Prune (High Recall)
        filtered_tables_dict = tables_dict
        if enable_pruning:
            master_plan, filtered_tables_dict = self.plan_and_prune_initial(question, tables_dict, question_id, evidence, enable_pruning=True)
            # print(f"[Schema Linking] Plan & Prune: kept {len(filtered_tables_dict)}/{len(tables_dict)} tables.")
        else:
            # Generate plan only
            master_plan, _ = self.plan_and_prune_initial(question, tables_dict, question_id, evidence, enable_pruning=False)
        
        if DEBUG:
            return None

        # 3. Semantic Linking (One-time)
        summary_res = self.summarize_tables(filtered_tables_dict, question, question_id, evidence, logical_plan=master_plan)
        
        # State Tracking
        confirmed_columns_set = set() # Stores "table.column"
        final_refined_schema = {}
        initial_table_judgments = {}
        refinement_chat_session = GPTChat(self.model, temperature=self.temperature)
        
        # 4. Iterative Refinement Loop
        for round_idx in range(MAX_REFINE_ROUNDS):
            # print(f"[Schema Linking] Starting Round {round_idx}")
            
            linking_results = {}
            
            # Round 0: Exploration Step
            if round_idx == 0:
                target_tables = list(filtered_tables_dict.keys())
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_table = {
                        executor.submit(
                            self.explore_and_link_table, 
                            t_name, filtered_tables_dict[t_name], sqlite_path, 
                            question, question_id, summary_res, evidence
                        ): t_name for t_name in target_tables
                    }
                    for future in concurrent.futures.as_completed(future_to_table):
                        try:
                            res = future.result()
                            linking_results[res["table_full_name"]] = res
                        except Exception as e:
                            print(f"[Schema Linking] Worker exception: {e}")
                            if FAIL_ON_ERROR:
                                raise ValueError("Table exploration failed")
                
                # Save initial judgments for final output
                initial_table_judgments = linking_results

            refined_res = self.refine_schema_globally(
                filtered_tables_dict, 
                linking_results, 
                question, sqlite_path, question_id, 
                iteration_round=round_idx,
                summarization_result=summary_res,
                evidence=evidence,
                confirmed_columns=confirmed_columns_set if round_idx > 0 else None,
                chat_session=refinement_chat_session
            )
            
            # Merge & Update State
            current_refined = refined_res.get("refined_schema", {})
            added_count = 0
            
            # Build a mapping from short_name to full_name for normalization
            short_to_full = {}
            for full_name, info in filtered_tables_dict.items():
                short_name = info['table_name']
                short_to_full[short_name] = full_name
                short_to_full[full_name] = full_name  # Also map full_name to itself
            
            for t_name, t_data in current_refined.items():
                # Normalize table name: always use full name
                normalized_t_name = short_to_full.get(t_name, t_name)
                
                if normalized_t_name not in final_refined_schema:
                    final_refined_schema[normalized_t_name] = {"relevant_columns": []}
                
                existing_cols = {c['column_name'] for c in final_refined_schema[normalized_t_name]['relevant_columns']}
                for new_col in t_data.get("relevant_columns", []):
                    c_name = new_col.get('column_name')
                    if c_name and c_name not in existing_cols:
                        final_refined_schema[normalized_t_name]['relevant_columns'].append(new_col)
                        
                        # Add to confirmed set for context (use both full and short for matching)
                        confirmed_columns_set.add(f"{normalized_t_name}.{c_name}")
                        short_name = normalized_t_name.split('.')[-1]
                        confirmed_columns_set.add(f"{short_name}.{c_name}")
                        added_count += 1

            
            print(f"[Schema Linking - Question: {question_id}] Round {round_idx} added {added_count} columns.")
            if round_idx > 0 and added_count == 0:
                print(f"[Schema Linking - Question: {question_id}] Convergence reached after {round_idx + 1} rounds.")
                break

        return {
            "refined_schema": final_refined_schema,
            "table_judgments": initial_table_judgments
        }

def process_single_case(case_data: Dict[str, Any], 
                       output_file: str, 
                       file_lock: threading.Lock, 
                       model: str, 
                       cache_dir: str):
    try:
        question_id = case_data['question_id']
        linker = SchemaLinking(model=model, max_workers=DEFAULT_MAX_WORKERS, cache_dir=cache_dir)
        q = case_data['question']
        evidence = case_data.get('evidence', '')
        db_id = case_data['db_id']
        sqlite_path = f"data/dev_databases/{db_id}/{db_id}.sqlite"
        
        if not os.path.exists(sqlite_path):
            raise ValueError(f"SQLite path not found: {sqlite_path}") 
            
        result = linker.run_schema_linking(
            q, question_id, evidence=evidence, sqlite_path=sqlite_path,
            database_root_dir="data/dev_databases",
            enable_pruning=False
        )

        if DEBUG:
            return
        
        final_output = {
            "question_id": question_id,
            "question": q,
            "db_id": db_id,
            "evidence": evidence,
            "result": result
        }

        with file_lock:
            current_results = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        current_results = json.load(f)
                except Exception:
                    pass
            current_results[str(question_id)] = final_output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)
                
    except Exception as e:
        print(f"Error in process_single_case (ID {case_data.get('question_id')}): {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Schema Linking")
    parser.add_argument("--model", type=str, default="deepseek-v3.2", help="Model to use")
    parser.add_argument("--max_concurrent_cases", type=int, default=4, help="Max parallel cases")
    parser.add_argument("--data_file_path", type=str, default='./data/dev.json', help="Path to data")
    parser.add_argument("--subset_path", type=str, default=None, help="Path to subset file")
    parser.add_argument("--output_dir", type=str, default="schema_linking", help="Output dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model = args.model
    MAX_CONCURRENT_CASES = args.max_concurrent_cases

    cache_dir = f"{args.output_dir}/cache/{model}"
    output_file = f"{args.output_dir}/results_{model}.json"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    print("[Main] Loading data files...")
    with open(args.data_file_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    if args.subset_path is not None:
        with open(args.subset_path, 'r', encoding='utf-8') as f:
            subset_ids = [c['question_id'] for c in json.load(f)]
        dev_data = [c for c in dev_data if c['question_id'] in subset_ids]
    
    # dev_data = dev_data[:20]

    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except Exception:
            pass
    
    cases_to_process = [c for c in dev_data if str(c['question_id']) not in existing_results]
    print(f"[Main] {len(cases_to_process)} cases remaining to process")

    results_file_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CASES) as executor:
        futures = []
        for case in cases_to_process:
            future = executor.submit(process_single_case, case, output_file, results_file_lock, model, cache_dir)
            futures.append(future)
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Cases"):
            pass
            
    print(f"\n[Main] All done! Results saved to {output_file}")
