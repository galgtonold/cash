"""Redis-based cache backend for Cash."""

from __future__ import annotations

import contextlib
import logging
import pickle
from typing import Any

from cash.exceptions import CacheBackendError, DependencyNotFoundError

from ._base import CacheBackend, CacheMetadata
from .serialization import PickleSerializer, Serializer

try:
    import redis  # noqa: F401
    from redis.exceptions import RedisError
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    RedisError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

__all__ = ["RedisBackend", "HAS_REDIS"]

class RedisBackend(CacheBackend):
    """
    Redis-based cache backend.
    Requires 'redis' package: pip install redis
    """
    source_label: str = "REDIS"

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: str | None = None, prefix: str = 'cash:',
                 socket_keepalive: bool = True,
                 health_check_interval: int = 30,
                 retry_on_timeout: bool = True,
                 max_retries: int = 3,
                 **kwargs):
        try:
            import redis
            from redis.backoff import ExponentialBackoff
            from redis.exceptions import ConnectionError, TimeoutError  # noqa: F401
            from redis.retry import Retry
        except ImportError as exc:
            raise DependencyNotFoundError("RedisBackend requires 'redis' package. Install it with 'pip install redis'.") from exc

        retry_strategy = Retry(ExponentialBackoff(), max_retries)

        self.client = redis.Redis(
            host=host, port=port, db=db, password=password,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
            retry=retry_strategy,
            **kwargs
        )
        self.prefix = prefix

    def _get_keys(self, key: str) -> tuple[str, str]:
        return f"{self.prefix}{key}:meta", f"{self.prefix}{key}:data"

    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
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

                # Deserialize data
                serializer_cls = metadata.get('serializer_cls', PickleSerializer)
                serializer = serializer_cls()
                value = serializer.deserialize(data_bytes)

                return metadata, value
            except (pickle.UnpicklingError, KeyError, TypeError, ValueError) as e:
                logger.debug("Redis get() deserialization error: %s", e)
                return None, None
        return None, None

    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Serializer | None = None) -> None:
        meta_key, data_key = self._get_keys(key)

        metadata = self._init_metadata(metadata, key)

        if serializer is None:
            serializer = PickleSerializer()

        # Serialize value
        serialized_value = serializer.serialize(value)

        # Store size and storage identifier
        metadata['size'] = len(serialized_value)
        if 'storage' not in metadata:
            metadata['storage'] = ['Redis']

        # Serialize metadata (includes ttl inside the pickled blob)
        meta_bytes = pickle.dumps(metadata)

        # TTL handling — also set Redis-level EXPIRE so keys auto-evict.
        # Reading from metadata (not meta_bytes) is safe: pickle.dumps is
        # non-mutating.
        ttl = metadata.get('ttl')

        try:
            pipe = self.client.pipeline()
            pipe.set(meta_key, meta_bytes)
            pipe.set(data_key, serialized_value)

            if ttl:
                pipe.expire(meta_key, ttl)
                pipe.expire(data_key, ttl)

            pipe.execute()
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis set failed for key {key!r}: {exc}") from exc

    def delete(self, key: str) -> None:
        meta_key, data_key = self._get_keys(key)
        try:
            self.client.delete(meta_key, data_key)
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis delete failed for key {key!r}: {exc}") from exc

    def clear(self) -> None:
        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor=cursor, match=f"{self.prefix}*", count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except (RedisError, OSError) as exc:
            raise CacheBackendError(f"Redis clear failed: {exc}") from exc

    def list_entries(self) -> list[dict]:
        entries = []
        # Scan for meta keys
        match_pattern = f"{self.prefix}*:meta"
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=match_pattern, count=100)
            if keys:
                # Get all metadata
                # MGET might be too big if many keys, but let's try batching if needed.
                # For now, just iterate or pipeline batches.
                # Let's pipeline in batches of 100
                pipe = self.client.pipeline()
                for k in keys:
                    pipe.get(k)
                results = pipe.execute()

                for res in results:
                    if res:
                        try:
                            entries.append(pickle.loads(res))
                        except (pickle.UnpicklingError, KeyError, TypeError) as e:
                            logger.debug("Failed to deserialize Redis cache entry: %s", e)
            if cursor == 0:
                break
        return entries

    def shutdown(self) -> None:
        self.client.close()

    def lock(self, key: str) -> contextlib.AbstractContextManager:
        """
        Acquire a distributed lock for the key.
        Uses Redis standard locking (SET NX px).
        """
        # Lock name should be distinct from data key to avoid collision?
        # Actually, redis.lock creates a key, usually `lock:name`.
        # We'll use prefix + "lock:" + key
        lock_name = f"{self.prefix}lock:{key}"
        # Return the lock object which is a context manager
        return self.client.lock(lock_name, timeout=60, blocking_timeout=10)
