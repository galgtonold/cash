"""Redis-based cache backend for Cash."""

from __future__ import annotations

import contextlib
import logging
import math
import pickle
from typing import Any

from cash.exceptions import CacheBackendError, DependencyNotFoundError

from . import cache_dir as _cache_dir
from ._base import CacheBackend, MetadataDict
from ._writes import PendingWrites
from .serialization import RESTORE_ERRORS, PickleSerializer, Serializer, restore_value

try:
    import redis  # noqa: F401 - an availability probe
    from redis.exceptions import RedisError

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    RedisError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

__all__ = ["RedisBackend", "HAS_REDIS"]

#: What ends the two keys of an entry. `RedisBackend.clear` removes keys by
#: these, so it takes nothing cash did not write.
_META_SUFFIX = ":meta"
_DATA_SUFFIX = ":data"


class RedisBackend(CacheBackend):
    """Entries in a Redis server, shared by every machine that reaches it.

    Needs the ``redis`` package (``pip install redis``). Entries are pickled:
    anyone who can write to the server can run code in your process. As a
    tier, it does not take values over 10 MiB.

    Args:
        host: Server host name.
        port: Server port.
        db: Redis database number.
        password: Server password, if any.
        prefix: Prepended to every key cash writes. An entry's keys are
            ``<prefix>f<N>:<key>:meta`` and ``...:data``, N being the cache
            format, so cash versions with different formats share a server
            without reading each other's entries.
        socket_keepalive: Keep idle connections open.
        health_check_interval: Seconds between connection health checks.
        retry_on_timeout: Retry a command that timed out.
        max_retries: Retries per command, with exponential backoff.
        **kwargs (Any): Passed to ``redis.Redis``, for example ``ssl=True``.

    Raises:
        DependencyNotFoundError: ``redis`` is not installed.
    """

    source_label: str = "REDIS"
    cost_kind: str = "redis"
    # Tier-promotion hint: Redis is in-memory on the server and the
    # protocol doesn't love multi-MB transfers, so the tiered pipeline
    # skips Redis for objects past this cap. Bare-backend writes are
    # not gated — users can opt-in to bigger values explicitly.
    max_size_bytes: int | None = 10 * 1024 * 1024

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "cash:",
        socket_keepalive: bool = True,
        health_check_interval: int = 30,
        retry_on_timeout: bool = True,
        max_retries: int = 3,
        **kwargs,
    ):
        try:
            import redis
            from redis.backoff import ExponentialBackoff
            from redis.retry import Retry
        except ImportError as exc:
            raise DependencyNotFoundError("The Redis backend needs the redis package: pip install redis") from exc

        retry_strategy = Retry(ExponentialBackoff(), max_retries)

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
            retry=retry_strategy,
            **kwargs,
        )
        self.prefix = prefix
        # Per-backend async writes: serialization happens on the calling
        # thread, the actual Redis pipeline runs in this executor.
        self._writes = PendingWrites()

    def _entry_prefix(self) -> str:
        """The prefix of this cache format's entries (`cache_dir.format_tag`).

        The format is part of every key, so a server shared by machines on
        different cash versions holds each format's entries side by side: an
        entry written in another format is never read, and never removed by
        a build that cannot judge it.
        """
        return f"{self.prefix}{_cache_dir.format_tag()}:"

    def _get_keys(self, key: str) -> tuple[str, str]:
        base = f"{self._entry_prefix()}{key}"
        return f"{base}{_META_SUFFIX}", f"{base}{_DATA_SUFFIX}"

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        # Wait for any pending write for this key.
        self._writes.wait(key)
        meta_key, data_key = self._get_keys(key)

        try:
            # Use pipeline for atomicity/performance
            pipe = self.client.pipeline()
            pipe.get(meta_key)
            pipe.get(data_key)
            meta_bytes, data_bytes = pipe.execute()
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis get failed for key {key!r}: {exc}") from exc

        if meta_bytes and data_bytes:
            try:
                metadata = pickle.loads(meta_bytes)
                value = restore_value(metadata, data_bytes)
                metadata.setdefault("source", self.source_label)
                return metadata, value
            except RESTORE_ERRORS as e:
                logger.debug("Redis get() deserialization error: %s", e)
                return None, None
        return None, None

    def get_metadata(self, key: str) -> MetadataDict | None:
        """Fetch the metadata key only, not the value.

        Not the base ``get()``: that brings the whole cached object over the
        wire to answer with a few hundred bytes of metadata.

        A metadata key with no data key still reports, matching the file
        backend: an entry whose value was too large to persist keeps its
        execution time and lineages for badges and upstream simulation.
        """
        self._writes.wait(key)
        meta_key, _data_key = self._get_keys(key)

        try:
            meta_bytes = self.client.get(meta_key)
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis get_metadata failed for key {key!r}: {exc}") from exc

        if not meta_bytes:
            return None

        try:
            metadata = pickle.loads(meta_bytes)
        except RESTORE_ERRORS as e:
            logger.debug("Redis get_metadata() deserialization error: %s", e)
            return None

        metadata.setdefault("source", self.source_label)
        return metadata

    def set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> None:
        """Serialize on the calling thread, run the pipeline in background."""
        meta_key, data_key = self._get_keys(key)

        metadata = self._init_metadata(metadata, key)

        if serializer is None:
            serializer = PickleSerializer()

        # IMPORTANT: serialize on the calling thread.
        serialized_value = serializer.serialize(value)

        # Store size and storage identifier
        metadata["size"] = len(serialized_value)
        if "storage" not in metadata:
            metadata["storage"] = [self.source_label]

        # Serialize metadata (includes ttl inside the pickled blob)
        meta_bytes = pickle.dumps(metadata)
        ttl = metadata.get("ttl")

        self._writes.submit_sized(
            key,
            len(serialized_value),
            True,
            self._do_set_sync,
            meta_key,
            data_key,
            meta_bytes,
            serialized_value,
            ttl,
            key,
        )

    def _do_set_sync(
        self, meta_key: str, data_key: str, meta_bytes: bytes, serialized_value: bytes, ttl: float | None, key: str
    ) -> None:
        """The actual Redis pipeline — runs in the PendingWrites worker."""
        try:
            pipe = self.client.pipeline()
            pipe.set(meta_key, meta_bytes)
            pipe.set(data_key, serialized_value)

            if ttl:
                # Redis expires in whole seconds; round up so a fractional
                # ttl never drops the entry before cash would.
                pipe.expire(meta_key, math.ceil(ttl))
                pipe.expire(data_key, math.ceil(ttl))

            pipe.execute()
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis set failed for key {key!r}: {exc}") from exc

    def delete(self, key: str) -> None:
        # Drain any pending write so the delete actually deletes.
        self._writes.drain(key)
        meta_key, data_key = self._get_keys(key)
        try:
            self.client.delete(meta_key, data_key)
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis delete failed for key {key!r}: {exc}") from exc

    def clear(self) -> None:
        """Delete this cache's entries, of every cache format, and nothing else.

        The prefix is a string: ``team`` also starts ``team-reports:...``,
        and an empty one starts every key of the database. So only keys named
        as cash names an entry's two keys go (`cache_dir.is_store_entry_name`).
        """
        self._writes.wait_all()
        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=f"{_glob_escape(self.prefix)}*", count=100)
                keys = [k for k in keys if self._wrote(k)]
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis clear failed: {exc}") from exc

    def _wrote(self, redis_key: Any) -> bool:
        """Is *redis_key* one of an entry's keys, in any cache format?"""
        name = redis_key.decode("utf-8", "replace") if isinstance(redis_key, bytes) else str(redis_key)
        return name.startswith(self.prefix) and _cache_dir.is_store_entry_name(
            name[len(self.prefix) :], ":", (_META_SUFFIX, _DATA_SUFFIX), untagged=bool(self.prefix)
        )

    def list_entries(self) -> list[dict]:
        self._writes.wait_all()
        entries = []
        # Scan for meta keys
        match_pattern = f"{_glob_escape(self._entry_prefix())}*{_META_SUFFIX}"
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=match_pattern, count=100)
            if keys:
                # One round trip per SCAN page (up to 100 keys).
                pipe = self.client.pipeline()
                for k in keys:
                    pipe.get(k)
                results = pipe.execute()

                for res in results:
                    if res:
                        try:
                            entries.append(pickle.loads(res))
                        except RESTORE_ERRORS as e:
                            logger.debug("Failed to deserialize Redis cache entry: %s", e)
            if cursor == 0:
                break
        return entries

    def shutdown(self) -> None:
        self._writes.shutdown(wait=True)
        self.client.close()

    def lock(self, key: str) -> contextlib.AbstractContextManager:
        """A lock on *key* shared by every process using this server: Redis's
        own lock (``SET NX PX``), as a context manager. Its name is in a
        ``lock:`` namespace under the prefix, apart from the entries' keys."""
        return self.client.lock(f"{self.prefix}lock:{key}", timeout=60, blocking_timeout=10)


def _glob_escape(text: str) -> str:
    """*text* as a literal in a Redis ``MATCH`` pattern."""
    return "".join(f"\\{c}" if c in "*?[]\\" else c for c in text)
