"""Typed, in-memory view of a statement cache entry's metadata.

Sibling to :class:`cash.backends.CacheMetadata` (the decorator-side view).
Both are dataclass *views* over the same opaque dict channel the backends
round-trip — see ADR-013 and the ``MetadataDict`` note in
``cash/backends/_base.py``.

Lives in its own leaf module (imports nothing from the ``statement``
package) so that both :mod:`processor` and its sibling helpers
(:mod:`freshness`, :mod:`file_deps`, :mod:`restore`) can import it at
runtime without re-introducing the ``processor → freshness → processor``
cycle.

Wire contract (mirrors :class:`CacheMetadata`):
    * ``to_dict()`` omits ``None`` fields, preserving the historical
      "only-set-keys" dict shape so backend presence-checks keep working.
    * ``from_dict()`` is lenient: unknown keys (legacy aliases like
      ``cell_code``, backend-private keys) are ignored, missing keys
      default to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

__all__ = ["StatementCacheMetadata"]


@dataclass(frozen=True)
class StatementCacheMetadata:
    """Metadata stored alongside a cached statement result."""

    key: str | None = None
    timestamp: float | None = None
    inputs: list[str] | None = None
    outputs: list[str] | None = None
    execution_time: float | None = None
    source_hash: str | None = None
    code: str | None = None
    # path -> {"mtime": float, "size": int}
    file_dependencies: dict[str, dict[str, float]] | None = None
    force_persist: bool | None = None
    output_lineages: dict[str, str] | None = None
    storage: list[str] | None = None
    source: str | None = None
    skipped_reason: str | None = None
    metadata_only: bool | None = None
    ttl: int | None = None
    cost_model_size_bytes: int | None = None
    cost_model_restore_seconds: float | None = None
    cost_model_type_name: str | None = None
    cost_model_family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: value
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatementCacheMetadata:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
