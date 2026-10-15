"""Parsing helpers shared across the model calls.

Structured responses are read as the last complete JSON object containing the
required keys; a whole-response JSON repair is attempted on failure (no merging
of multiple objects, no model retry). SQL responses are stripped of an optional
code fence and a trailing semicolon.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:sql|json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Return a single cleaned SQL statement from a model response."""
    if not text:
        return ""
    m = _FENCE.search(text)
    sql = m.group(1) if m else text
    sql = sql.strip()
    # keep only up to the first statement terminator run; strip trailing ';'
    sql = sql.strip().rstrip(";").strip()
    return sql


def _iter_json_objects(text: str):
    """Yield top-level {...} spans from text, outermost first by end position."""
    stack = []
    start = None
    spans = []
    in_str = False
    esc = False
    quote = ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
        elif ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    spans.append((start, i + 1))
    for start, end in spans:
        yield text[start:end]


def extract_json(text: str, required_keys: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the last complete JSON object containing all required keys.

    Falls back to a whole-response ``json.loads`` repair; returns None if no
    valid object with the required keys is found.
    """
    if not text:
        return None
    candidates = list(_iter_json_objects(text))
    for span in reversed(candidates):            # last complete object first
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in required_keys):
            return obj
    # whole-response repair
    try:
        from json_repair import loads
        obj = loads(text.strip())
        if isinstance(obj, dict) and all(k in obj for k in required_keys):
            return obj
    except (ValueError, TypeError):
        pass
    return None
