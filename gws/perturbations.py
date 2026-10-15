"""DB-shared visible metadata perturbations; underlying SQLite bytes stay fixed."""
from dataclasses import asdict
import copy
import hashlib
import json
import random
import re
from .schema import ForeignKey
from .metrics import edge_keys


def type_family(dtype):
    upper = dtype.upper()
    if any(x in upper for x in ("INT", "REAL", "NUM", "DEC", "FLOA", "DOUB")):
        return "numeric"
    if any(x in upper for x in ("CHAR", "TEXT", "CLOB", "STRING")):
        return "text"
    return upper


def visible_schema(base, condition="clean", seed=42, excluded=()):
    if condition not in {"clean", "D100", "N100"}:
        raise ValueError("Unknown perturbation")
    schema = copy.deepcopy(base)
    injected, removed = [], []
    if condition == "D100":
        removed = edge_keys([asdict(f) for f in base.declared_fks])
        schema.declared_fks = []
    elif condition == "N100":
        forbidden = edge_keys([asdict(f) for f in base.declared_fks]) | {tuple(e) for e in excluded}
        pairs = {}
        for table, cs in base.tables.items():
            for col in cs:
                if not (col.pk or re.search(r"id$|key$|code$", col.name, re.I)):
                    continue
                for target, other_cs in base.tables.items():
                    if table == target:
                        continue
                    for other in other_cs:
                        if not (other.pk or re.search(r"id$|key$|code$", other.name, re.I)):
                            continue
                        if type_family(col.type) != type_family(other.type):
                            continue
                        fk = ForeignKey(table, col.name, target, other.name)
                        key = next(iter(edge_keys([asdict(fk)])))
                        if key not in forbidden:
                            pairs.setdefault(key, fk)
        candidates = [pairs[k] for k in sorted(pairs)]
        offset = int(hashlib.sha256(json.dumps(base.name + ":noise").encode()).hexdigest()[:8], 16)
        random.Random(seed + offset).shuffle(candidates)
        chosen = candidates[:len(base.declared_fks)]
        schema.declared_fks.extend(chosen)
        injected = edge_keys([asdict(f) for f in chosen])
    schema.metadata = {"schema_mode": "uncertain"}
    schema.raw_ddl = {t: schema._trusted_table_ddl(t, schema.table_names())
                      for t in schema.table_names()}
    audit = {"condition": condition, "seed": seed, "db_id": base.name,
             "original_fk_count": len(base.declared_fks), "removed_edges": sorted(removed),
             "injected_edges": sorted(injected),
             "visible_edges": sorted(edge_keys([asdict(f) for f in schema.declared_fks]))}
    return schema, audit
