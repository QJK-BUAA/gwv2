"""Exact-request response cache for reproducible LLM experiments."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestEnvelope:
    model: str
    messages: list[dict]
    temperature: float
    seed: int | None
    extra_body: dict = field(default_factory=dict)
    max_tokens: int = 4096
    backend_hash: str = ""

    @property
    def request_hash(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    request_hash: str
    text: str
    response_id: str
    usage: dict[str, Any]
    model: str = ""
    finish_reason: str = ""


class ResponseCache:
    """Append-only JSONL cache keyed by an exact canonical request hash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._items = self._load()

    def _load(self) -> dict[str, CachedResponse]:
        if not self.path.exists():
            return {}
        items: dict[str, CachedResponse] = {}
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    item = CachedResponse(
                        request_hash=str(payload["request_hash"]),
                        text=str(payload["text"]),
                        response_id=str(payload.get("response_id", "")),
                        usage=dict(payload.get("usage") or {}),
                        model=str(payload.get("model", "")),
                        finish_reason=str(payload.get("finish_reason", "")),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid response cache entry at "
                        f"{self.path}:{line_number}"
                    ) from exc
                items[item.request_hash] = item
        return items

    def get(
        self,
        request: RequestEnvelope,
    ) -> CachedResponse | None:
        with self._lock:
            return self._items.get(request.request_hash)

    def put(
        self,
        request: RequestEnvelope,
        text: str,
        response_id: str = "",
        usage: dict[str, Any] | None = None,
        model: str = "",
        finish_reason: str = "",
    ) -> CachedResponse:
        item = CachedResponse(
            request_hash=request.request_hash,
            text=text,
            response_id=response_id,
            usage=dict(usage or {}),
            model=model,
            finish_reason=finish_reason,
        )
        payload = json.dumps(
            asdict(item),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")
            self._items[item.request_hash] = item
        return item
