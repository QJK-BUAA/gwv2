"""Shared dataclasses for the Ground--Explore--Commit pipeline.

These mirror the structured objects the paper passes between its six logical
model calls and the deterministic machinery that runs between them (candidate
graph, paths, probes, beliefs, coverage contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Ground
# --------------------------------------------------------------------------- #
@dataclass
class Requirement:
    """One semantic need parsed from the question (Appendix: requirement record).

    ``kind`` enumerates output / filter / aggregate / group / order / having /
    denominator / branch; ``columns`` are the physical columns the requirement
    maps to; ``alternatives`` are competing columns; ``reason`` is the model's
    justification. An empty requirement list is an error upstream.
    """
    id: str
    kind: str
    need: str
    columns: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class RelationCandidate:
    """A candidate join edge between two physical columns.

    ``source`` labels provenance ("structural", "visible_fk", "model") and
    ``score`` is the source-based path-ranking prior (0.65 / 0.75). The same
    unordered column pair keeps the larger score; different column-level
    predicates between the same table pair are preserved separately.
    """
    src_table: str
    src_column: str
    dst_table: str
    dst_column: str
    score: float
    sources: tuple[str, ...] = ()

    @property
    def edge_id(self) -> str:
        a = f"{self.src_table}.{self.src_column}"
        b = f"{self.dst_table}.{self.dst_column}"
        return "|".join(sorted([a, b]))

    def normalized(self) -> "RelationCandidate":
        """Canonical orientation: endpoints sorted lexicographically."""
        a = (self.src_table, self.src_column)
        b = (self.dst_table, self.dst_column)
        if (a[0].lower(), a[1].lower()) <= (b[0].lower(), b[1].lower()):
            return self
        return RelationCandidate(
            src_table=b[0], src_column=b[1], dst_table=a[0], dst_column=a[1],
            score=self.score, sources=self.sources,
        )


@dataclass
class Path:
    """A bounded simple path over the candidate graph (<= MAX_PATH_EDGES edges)."""
    tables: list[str]
    edges: list[RelationCandidate]
    cost: float

    @property
    def path_id(self) -> str:
        return " -> ".join(self.tables)


# --------------------------------------------------------------------------- #
# Explore
# --------------------------------------------------------------------------- #
@dataclass
class Interpretation:
    """One of 2--4 mutually exclusive readings of a local decision."""
    key: str
    label: str
    prior: float


@dataclass
class Probe:
    """A read-only observation action assigned to exactly one decision.

    ``kind`` is one of equals / nonnull / duplicate / overlap. ``likelihoods``
    maps interpretation key -> p(hit | z): the probability the probe returns a
    HIT given each interpretation (the MISS likelihood is the complement). The
    four templates are compiled to a single ``SELECT EXISTS`` each.
    """
    id: str
    decision_id: str
    kind: str
    table: str = ""
    column: str = ""
    literal: Any = None
    table2: str = ""
    column2: str = ""
    likelihoods: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Decision:
    """A local grounding decision with its interpretations and attached probes."""
    id: str
    description: str
    interpretations: list[Interpretation]
    probes: list[Probe] = field(default_factory=list)


@dataclass
class ProbeOutcome:
    """Result of executing one probe: HIT / MISS / unavailable."""
    probe_id: str
    kind: str
    status: str          # "hit" | "miss" | "error" | "timeout"
    available: bool      # True iff a Boolean row came back (hit/miss)
    sql: str = ""

    @property
    def is_hit(self) -> bool:
        return self.available and self.status == "hit"


# --------------------------------------------------------------------------- #
# Commit
# --------------------------------------------------------------------------- #
@dataclass
class CoverageItem:
    requirement_id: str
    state: str           # "covered" | "replaced" | "not_needed"
    columns: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class CommitContract:
    """The Commit output S* + coverage contract C (Sec. Commit)."""
    columns: list[str] = field(default_factory=list)          # physical "table.col"
    joins: list[RelationCandidate] = field(default_factory=list)
    coverage: list[CoverageItem] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    reason: str = ""

    def selected_tables(self) -> list[str]:
        tables: list[str] = []
        for col in self.columns:
            if "." in col:
                t = col.split(".", 1)[0]
                if t not in tables:
                    tables.append(t)
        return tables
