"""The decorator's typed view of a cache entry's metadata."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

__all__ = ["CacheMetadata"]


@dataclass(frozen=True)
class CacheMetadata:
    """Typed, in-memory view of a decorator cache entry's metadata.

    This is the *edge* representation: producers (the ``@cash.cache``
    decorator) build one and call :meth:`to_dict` before handing it to a
    backend; consumers call :meth:`from_dict` on what a backend returns.
    Backends themselves never see this class — they round-trip a plain
    ``dict`` opaquely (the metadata channel is polymorphic; the notebook
    layer pushes a differently-shaped ``StatementCacheMetadata`` through
    the same channel).

    Every field is optional and defaults to ``None`` — "absent" and
    "unset" are deliberately collapsed (a frozen dataclass field always
    has a value). :meth:`to_dict` omits ``None`` fields so the on-the-wire
    dict matches the historical "only-set-keys" shape, and :meth:`from_dict`
    ignores unknown keys and defaults missing ones, so it tolerates caches
    written by older or newer versions.
    """

    key: str | None = None
    created_at: float | None = None
    last_access: float | None = None
    access_count: int | None = None
    size: int | None = None
    storage: list[str] | None = None
    ttl: int | None = None
    # The ttl came from the decorator (``ttl=``), not from a tier's
    # ``default_ttl``: a lowered tier default shortens only the latter, and
    # ``cash inspect`` / ``cash clear --expired`` need to tell them apart.
    ttl_declared: bool | None = None
    execution_time: float | None = None
    # The function's OWN time, excluding everything cash did around it.
    # ``execution_time`` is measured from the top of the wrapper and so
    # includes cache-key hashing and lookup -- fine for reporting what a call
    # cost, useless for asking whether caching PAID, because the overhead
    # being judged is inside the number it would be judged against.
    body_seconds: float | None = None
    # The wall time a hit stands in for: the body's time divided by the
    # threads that were running cached calls alongside it: sixteen 0.5 s calls
    # on eight threads take 1 s, not 8 s.
    saves_seconds: float | None = None
    outputs: list[str] | None = None
    lineage_hash: str | None = None
    source: str | None = None  # Backend source identifier (e.g. 'RAM', 'disk')
    # Decorator-stamped identity / lineage fields.
    func_name: str | None = None
    args_hash: str | None = None
    state_hash: str | None = None
    timestamp: float | None = None
    auto_file_deps: dict[str, dict[str, float]] | None = None
    iterator_storage: str | None = None
    n_chunks: int | None = None
    # Deserialization instruction; round-tripped so get() can rebuild the value.
    serializer_cls: type | None = None
    # Notebook-annotation flags consumed by TieredBackend / lineage.
    #: Written by ``@cash.cache``: this entry was asked for by a decorator, so
    #: the compute floor, the cost model and the rate ceiling do not gate it
    #: (``TieredBackend.set``). Says nothing about how the value is stored.
    decorator_entry: bool | None = None

    #: The stored value IS what the next call hands back, so the RAM tier must
    #: really copy it and refuses one it cannot (``InMemoryBackend.set``). Set
    #: by ``@cash.cache`` EXCEPT for a ``frozen=True`` function, which has
    #: already promised the result is not modified -- handing the same object
    #: back is what that promises. Split out of ``decorator_entry``: the two
    #: rode on one flag, so ``frozen=True`` silently lost disk persistence
    #: when the rate ceiling started reading it as "not a decorated entry".
    copy_required: bool | None = None

    #: Where the global RNG stood before and after the call that computed this
    #: entry, so a hit can leave it where the body did (``RngWatch.replay_parts``).
    rng_replay: dict[str, Any] | None = None

    force_persist: bool | None = None
    metadata_only: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of set fields, omitting ``None`` (the wire format)."""
        return {f.name: value for f in fields(self) if (value := getattr(self, f.name)) is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheMetadata:
        """Build from a backend dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
