"""Frozen DIN-SQL and C3 schema-linking adapters for the shared-backend comparison.

    adapter(schema, llm, db, question, evidence) -> selection(schema, columns, **extra)

Model and database access are injected by source_adapters.py. The DIN-SQL
schema-linking few-shot exemplars below preserve the released public prompt.
"""
from __future__ import annotations

import re

from baselines.common import schema_columns, SafeDB


# The released DIN-SQL schema-linking prompt (few-shot chain-of-thought).
DIN_SQL_SCHEMA_LINKING_PROMPT = '''Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Find the buildings which have rooms with capacity more than 50."
A: Let's think step by step. In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
"the buildings which have rooms" so we need column = [classroom.building]
"rooms with capacity" so we need column = [classroom.capacity]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
Schema_links: [classroom.building,classroom.capacity,50]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "How many heads of the departments are older than 56 ?"
A: Let's think step by step. In the question "How many heads of the departments are older than 56 ?", we are asked:
"How many heads of the departments" so we need column = [head.*]
"older" so we need column = [head.age]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [56]. So the Schema_links are:
Schema_links: [head.*,head.age,56]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"
A: Let's think step by step. In the question "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?", we are asked:
"distinct creation years of the departments" so we need column = [department.Creation]
"departments managed by" so we need column = [management.department_ID]
"born in" so we need column = [head.born_state]
Based on the columns and tables, we need these Foreign_keys = [department.Department_ID = management.department_ID,management.head_ID = head.head_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = ['Alabama']. So the Schema_links are:
Schema_links: [department.Creation,department.Department_ID = management.department_ID,head.head_ID = management.head_ID,head.born_state,'Alabama']

Table Addresses, columns = [*,address_id,line_1,line_2,city,zip_postcode,state_province_county,country]
Table Candidate_Assessments, columns = [*,candidate_id,qualification,assessment_date,asessment_outcome_code]
Table Candidates, columns = [*,candidate_id,candidate_details]
Table Courses, columns = [*,course_id,course_name,course_description,other_details]
Table People, columns = [*,person_id,first_name,middle_name,last_name,cell_mobile_number,email_address,login_name,password]
Table People_Addresses, columns = [*,person_address_id,person_id,address_id,date_from,date_to]
Table Student_Course_Attendance, columns = [*,student_id,course_id,date_of_attendance]
Table Student_Course_Registrations, columns = [*,student_id,course_id,registration_date]
Table Students, columns = [*,student_id,student_details]
Foreign_keys = [Students.student_id = People.person_id,People_Addresses.address_id = Addresses.address_id,People_Addresses.person_id = People.person_id,Student_Course_Registrations.course_id = Courses.course_id,Student_Course_Registrations.student_id = Students.student_id,Student_Course_Attendance.student_id = Student_Course_Registrations.student_id,Student_Course_Attendance.course_id = Student_Course_Registrations.course_id,Candidates.candidate_id = People.person_id,Candidate_Assessments.candidate_id = Candidates.candidate_id]
Q: "List the id of students who never attends courses?"
A: Let's think step by step. In the question "List the id of students who never attends courses?", we are asked:
"id of students" so we need column = [Students.student_id]
"never attends courses" so we need column = [Student_Course_Attendance.student_id]
Based on the columns and tables, we need these Foreign_keys = [Students.student_id = Student_Course_Attendance.student_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [Students.student_id = Student_Course_Attendance.student_id]

'''


def _tables_block(schema):
    lines = []
    for table in schema.table_names():
        cols = ",".join(["*"] + [c.name for c in schema.tables[table]])
        lines.append(f"Table {table}, columns = [{cols}]")
    fks = ",".join(f"{f.src_table}.{f.src_column} = {f.ref_table}.{f.ref_column}"
                   for f in schema.declared_fks)
    lines.append(f"Foreign_keys = [{fks}]")
    return "\n".join(lines)


def _valid_columns(schema):
    valid = {}
    for table, cols in schema.tables.items():
        for c in cols:
            valid[f"{table}.{c.name}".lower()] = f"{table}.{c.name}"
    return valid


def din_sql(schema, llm, db, question, evidence):
    """DIN-SQL schema linking: released few-shot CoT that emits Schema_links."""
    prompt = (DIN_SQL_SCHEMA_LINKING_PROMPT + _tables_block(schema) +
              (f'\nEvidence: {evidence}' if evidence else '') +
              f'\nQ: "{question}"\nA: Let\'s think step by step.')
    raw = llm.complete("", prompt, stage="din_schema_linking")
    valid = _valid_columns(schema)
    # Parse the final "Schema_links: [...]" list, per the released post-processing.
    match = re.findall(r"Schema_links:\s*\[(.*?)\]", raw, re.DOTALL)
    body = match[-1] if match else raw
    tokens = re.findall(r"[A-Za-z_][\w]*\.[A-Za-z_*][\w*]*", body)
    valid_tables = {t.lower(): t for t in schema.table_names()}
    columns = set()
    for token in tokens:
        low = token.lower()
        if low in valid:
            columns.add(valid[low])
        elif low.endswith(".*") and low[:-2] in valid_tables:
            # A "table.*" mention selects every column of that table.
            table = valid_tables[low[:-2]]
            columns |= {f"{table}.{c.name}" for c in schema.tables[table]}
    if not columns:
        # Degenerate parse: fall back to every table the model named, else full.
        named = [t for t in schema.table_names() if re.search(rf"\b{re.escape(t)}\b", raw)]
        columns = set(schema_columns(schema, [f"{t}.{c.name}" for t in named
                                              for c in schema.tables[t]])) if named \
            else set(schema_columns(schema))
    return selection(schema, columns, details={"schema_links_raw": raw})


C3_TABLE_SYSTEM = """You are an expert database engineer performing schema linking.
Given a database schema and a question, rank ALL tables by their relevance to
answering the question and return the relevant ones. Output only a JSON object
{"tables": ["t1", "t2", ...]} ordered from most to least relevant. Include every
table needed for joins and filtering; exclude clearly irrelevant tables."""

C3_COLUMN_SYSTEM = """You are an expert database engineer performing schema linking.
For each candidate table, list the columns needed to answer the question,
including columns used in SELECT, WHERE, GROUP BY, ORDER BY, HAVING and JOIN
conditions. Output only a JSON object mapping each table to its needed column
list, e.g. {"t1": ["c1", "c2"], "t2": ["c3"]}. Use only real column names."""


def _c3_schema_text(schema, tables=None):
    names = tables if tables is not None else schema.table_names()
    lines = []
    for table in names:
        cols = ", ".join(c.name for c in schema.tables[table])
        lines.append(f"# {table} ({cols})")
    fks = "; ".join(f"{f.src_table}.{f.src_column}={f.ref_table}.{f.ref_column}"
                    for f in schema.declared_fks)
    return "\n".join(lines) + (f"\nForeign keys: {fks}" if fks else "")


def c3_sql(schema, llm, db, question, evidence):
    """C3-SQL clear-prompting schema linking: table recall then column recall."""
    from gws.parsing import extract_json as response_object

    ask = (f"Schema:\n{_c3_schema_text(schema)}\n\n"
           f"Question: {question}" + (f"\nEvidence: {evidence}" if evidence else ""))
    table_raw = llm.complete(C3_TABLE_SYSTEM, ask, stage="c3_table_recall")
    try:
        chosen = response_object(table_raw, ("tables",)).get("tables", [])
    except Exception:
        chosen = []
    valid_tables = {t.lower(): t for t in schema.table_names()}
    tables = [valid_tables[str(t).lower()] for t in chosen
              if str(t).lower() in valid_tables]
    if not tables:
        tables = list(schema.table_names())

    col_ask = (f"Schema:\n{_c3_schema_text(schema, tables)}\n\n"
               f"Question: {question}" + (f"\nEvidence: {evidence}" if evidence else "") +
               f"\nCandidate tables: {tables}")
    col_raw = llm.complete(C3_COLUMN_SYSTEM, col_ask, stage="c3_column_recall")
    try:
        mapping = response_object(col_raw, ()) or {}
    except Exception:
        mapping = {}
    valid = _valid_columns(schema)
    columns = set()
    for table, cols in mapping.items() if isinstance(mapping, dict) else []:
        if not isinstance(cols, list):
            continue
        for col in cols:
            key = f"{table}.{col}".lower()
            if key in valid:
                columns.add(valid[key])
    if not columns:
        # Fall back to all columns of the recalled tables.
        columns = set(schema_columns(schema, [f"{t}.{c.name}" for t in tables
                                              for c in schema.tables[t]]))
    return selection(schema, columns,
                     details={"table_recall_raw": table_raw, "column_recall_raw": col_raw,
                              "recalled_tables": tables})


def selection(schema, columns, **extra):
    """Mirror of methods.selection: attach owning tables to a column set."""
    columns = set(columns)
    lookup = {f"{t}.{c.name}": t for t, cs in schema.tables.items() for c in cs}
    return {"columns": sorted(columns),
            "tables": sorted({lookup[c] for c in columns if c in lookup}),
            **extra}


NEW_ADAPTERS = {"din_sql": din_sql, "c3_sql": c3_sql}
