"""Atomic inference accounting; evaluation is outside the inference budget."""
from dataclasses import dataclass, field
import threading


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_calls: int = 6
    max_reads: int = 5
    calls: int = 0
    reads: int = 0
    events: list = field(default_factory=list)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def charge_call(self, stage, final=False):
        with self._lock:
            # One slot remains available for the single final submission.
            ceiling = self.max_calls if final else self.max_calls - 1
            if self.calls >= ceiling:
                raise BudgetExceeded("Model-call budget exhausted")
            self.calls += 1
            self.events.append({"kind": "model", "stage": stage, "index": self.calls})

    def charge_read(self, stage):
        with self._lock:
            if self.reads >= self.max_reads:
                raise BudgetExceeded("Database-read budget exhausted")
            self.reads += 1
            self.events.append({"kind": "read", "stage": stage, "index": self.reads})

    def snapshot(self):
        return {"calls": self.calls, "reads": self.reads,
                "max_calls": self.max_calls, "max_reads": self.max_reads}
