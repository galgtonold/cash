"""S3-based cache backend for Cash."""

from __future__ import annotations

import logging
import pickle
from typing import Any

from cash.exceptions import CacheBackendError, DependencyNotFoundError

from ._base import CacheBackend, MetadataDict, PendingWrites
from .serialization import PickleSerializer, Serializer

try:
    import boto3  # noqa: F401
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

logger = logging.getLogger(__name__)

__all__ = ["S3Backend", "HAS_BOTO3"]

class S3Backend(CacheBackend):
    """
    S3-based cache backend.
    Requires 'boto3' package: pip install boto3
    """
    source_label: str = "S3"

    def __init__(self, bucket: str, prefix: str = 'cash/',
                 max_pool_connections: int = 10,
                 retries: int = 3,
                 **kwargs):
        try:
            import boto3
            import botocore
            from botocore.config import Config
        except ImportError as exc:
            raise DependencyNotFoundError("S3Backend requires 'boto3' package. Install it with 'pip install boto3'.") from exc

        config = Config(
            max_pool_connections=max_pool_connections,
            retries={"max_attempts": retries, "mode": 'standard'}
        )

        self.s3 = boto3.client('s3', config=config, **kwargs)
        self.bucket = bucket
        self.prefix = prefix
        self.botocore_exceptions = botocore.exceptions
        # Per-backend async writes: serialize on the calling thread, run
        # the S3 PUTs in this executor so a slow upload doesn't block the
        # cell that produced the value.
        self._writes = PendingWrites()

    def _get_keys(self, key: str) -> tuple[str, str]:
        # S3 keys (paths)
        return f"{self.prefix}{key}.meta", f"{self.prefix}{key}.data"

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        # Wait for any pending write for this key.
        self._writes.wait(key)
        meta_key, data_key = self._get_keys(key)

        try:
            # Get metadata
            meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_key)
            meta_bytes = meta_obj['Body'].read()
            metadata = pickle.loads(meta_bytes)

            # Get data
            data_obj = self.s3.get_object(Bucket=self.bucket, Key=data_key)
            data_bytes = data_obj['Body'].read()

            # Deserialize
            serializer_cls = metadata.get('serializer_cls', PickleSerializer)
            serializer = serializer_cls()
            value = serializer.deserialize(data_bytes)

            metadata.setdefault('source', self.source_label)
            return metadata, value
        except self.botocore_exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                return None, None
            raise CacheBackendError(f"S3 get() failed for key {key!r}: {e}") from e
        except (pickle.UnpicklingError, KeyError, TypeError, OSError) as e:
            logger.debug("S3 get() deserialization error for key: %s", e)
            return None, None

    def get_metadata(self, key: str) -> MetadataDict | None:
        """GET the metadata object only.

        The base implementation performs a full ``get()`` and discards the
        value, which on S3 costs an extra request AND pulls the entire cached
        object across the network -- measured at 4,194,457 bytes downloaded
        for a 4MB entry, to return roughly 150 bytes of metadata. Both halves
        of that are billed.

        A metadata object with no data object still reports, matching the file
        backend: an entry whose value was too large to persist keeps its
        execution time and lineages for badges and upstream simulation.
        """
        self._writes.wait(key)
        meta_key, _data_key = self._get_keys(key)

        try:
            meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_key)
            metadata = pickle.loads(meta_obj['Body'].read())
        except self.botocore_exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                return None
            raise CacheBackendError(
                f"S3 get_metadata() failed for key {key!r}: {e}"
            ) from e
        except (pickle.UnpicklingError, KeyError, TypeError, OSError) as e:
            logger.debug("S3 get_metadata() deserialization error for key: %s", e)
            return None

        metadata.setdefault('source', self.source_label)
        return metadata

    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None) -> None:
        """Serialize on the calling thread, run the S3 PUTs in background."""
        meta_key, data_key = self._get_keys(key)

        metadata = self._init_metadata(metadata, key)

        if serializer is None:
            serializer = PickleSerializer()

        # IMPORTANT: serialize on the calling thread.
        serialized_value = serializer.serialize(value)

        metadata['size'] = len(serialized_value)
        if 'storage' not in metadata:
            metadata['storage'] = [self.source_label]
        meta_bytes = pickle.dumps(metadata)

        self._writes.submit(
            key, self._do_set_sync,
            meta_key, data_key, meta_bytes, serialized_value,
        )

    def _do_set_sync(self, meta_key: str, data_key: str,
                     meta_bytes: bytes, serialized_value: bytes) -> None:
        """The actual S3 PUTs — runs in the PendingWrites worker thread."""
        try:
            self.s3.put_object(Bucket=self.bucket, Key=data_key, Body=serialized_value)
            # Upload metadata (only after data succeeds)
            self.s3.put_object(Bucket=self.bucket, Key=meta_key, Body=meta_bytes)
        except self.botocore_exceptions.ClientError as e:
            # If data wrote but meta failed, the orphan is harmless
            # (we need meta to find it). Best-effort cleanup of the data
            # blob so we don't leak storage.
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=data_key)
            except self.botocore_exceptions.ClientError:
                logger.warning("Failed to clean up partial S3 write for key %s", data_key)
            raise CacheBackendError(f"S3 Write Error: {e}") from e

    #: Error codes that mean "this endpoint has no batch delete", as opposed
    #: to "the delete failed". Only these fall back to one request per object;
    #: anything else is a real failure and is raised.
    _NO_BATCH_DELETE = frozenset({'NotImplemented', 'MethodNotAllowed', '501', '405'})

    def delete(self, key: str) -> None:
        # Drain any pending write so the delete actually deletes.
        self._writes.drain(key)
        meta_key, data_key = self._get_keys(key)
        try:
            # One request for both objects. S3 has had DeleteObjects for as
            # long as it has had DeleteObject, and an entry is always two
            # objects, so this halves the request count of every eviction.
            self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={'Objects': [{'Key': meta_key}, {'Key': data_key}]},
            )
        except self.botocore_exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code not in self._NO_BATCH_DELETE:
                raise CacheBackendError(
                    f"S3 delete failed for key '{key}': {e}"
                ) from e
            # An S3-compatible endpoint without the batch API. Fall back
            # rather than fail: the point was to save a request, not to
            # require one particular implementation.
            logger.debug("Batch delete unavailable (%s); deleting individually",
                         error_code)
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=meta_key)
                self.s3.delete_object(Bucket=self.bucket, Key=data_key)
            except self.botocore_exceptions.ClientError as inner:
                raise CacheBackendError(
                    f"S3 delete failed for key '{key}': {inner}"
                ) from inner

    def clear(self) -> None:
        self._writes.wait_all()
        # List and delete all objects with prefix
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

            delete_keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        delete_keys.append({'Key': obj['Key']})

                        # Batch delete (max 1000)
                        if len(delete_keys) >= 1000:
                            self.s3.delete_objects(Bucket=self.bucket, Delete={'Objects': delete_keys})
                            delete_keys = []

            if delete_keys:
                self.s3.delete_objects(Bucket=self.bucket, Delete={'Objects': delete_keys})
        except self.botocore_exceptions.ClientError as e:
            raise CacheBackendError(
                f"S3 clear failed for prefix '{self.prefix}': {e}"
            ) from e

    def list_entries(self) -> list[dict]:
        self._writes.wait_all()
        entries = []
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.meta'):
                            try:
                                meta_obj = self.s3.get_object(Bucket=self.bucket, Key=key)
                                meta_bytes = meta_obj['Body'].read()
                                entries.append(pickle.loads(meta_bytes))
                            except (pickle.UnpicklingError, self.botocore_exceptions.ClientError, KeyError) as e:
                                logger.debug("Failed to deserialize S3 metadata for key %s: %s", key, e)
        except self.botocore_exceptions.ClientError as e:
            raise CacheBackendError(
                f"S3 list_entries failed for prefix '{self.prefix}': {e}"
            ) from e
        return entries

    def shutdown(self) -> None:
        self._writes.shutdown(wait=True)
