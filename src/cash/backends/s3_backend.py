"""S3-based cache backend for Cash."""

from __future__ import annotations

import logging
import pickle
from typing import Any

from cash.exceptions import CacheBackendError, DependencyNotFoundError

from ._base import CacheBackend, MetadataDict, PendingWrites
from .entry_format import (
    ENTRY_SUFFIX,
    CorruptEntry,
    metadata_span,
    pack_entry,
    unpack_entry,
)
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

    #: Bytes fetched from the front of an object when only its metadata is
    #: wanted. Metadata runs a few hundred bytes, so one ranged GET covers it
    #: with room to spare; anything larger gets a second request rather than a
    #: wrong answer.
    METADATA_PREFETCH_BYTES = 8192

    def _get_key(self, key: str) -> str:
        """One object per entry.

        It was two, a ``.meta`` and a ``.data``, which cost two requests for
        every read, write and delete -- and made reading metadata download the
        whole value. S3 has ranged GETs and the entry format has a
        length-prefixed header; together they make a metadata read one small
        request.

        Objects written by an older build use the ``.meta``/``.data`` suffixes
        and are simply invisible here: nothing reads them, so no migration
        runs against someone's bucket. They keep occupying storage until
        ``clear()`` (which sweeps the whole prefix) or a lifecycle rule
        removes them.
        """
        return f"{self.prefix}{key}{ENTRY_SUFFIX}"

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        # Wait for any pending write for this key.
        self._writes.wait(key)
        obj_key = self._get_key(key)

        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=obj_key)
            metadata, payload = unpack_entry(obj['Body'].read(), with_payload=True)
        except self.botocore_exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                return None, None
            raise CacheBackendError(f"S3 get() failed for key {key!r}: {e}") from e
        except (CorruptEntry, pickle.UnpicklingError, KeyError, TypeError, OSError) as e:
            logger.debug("S3 get() deserialization error for key %s: %s", key, e)
            return None, None

        # A metadata-only entry carries no value to restore; get_metadata
        # still reports it, for badges and upstream simulation.
        if metadata.get('metadata_only'):
            return None, None

        try:
            serializer_cls = metadata.get('serializer_cls', PickleSerializer)
            value = serializer_cls().deserialize(payload)
        except (pickle.UnpicklingError, AttributeError, ImportError,
                EOFError, TypeError, ValueError) as e:
            logger.debug("S3 get() could not restore the value for %s: %s", key, e)
            return None, None

        metadata.setdefault('source', self.source_label)
        return metadata, value

    def get_metadata(self, key: str) -> MetadataDict | None:
        """One ranged GET of the front of the object.

        The base implementation performs a full ``get()`` and discards the
        value -- two requests and the whole cached object over the network,
        measured at 4,194,457 bytes for a 4MB entry to return about 150 bytes
        of answer. Both halves of that are billed.
        """
        self._writes.wait(key)
        obj_key = self._get_key(key)

        try:
            head = self._ranged_get(obj_key, self.METADATA_PREFETCH_BYTES)
            try:
                metadata, _ = unpack_entry(head, with_payload=False)
            except CorruptEntry:
                # Widen only if the header says the metadata really is longer
                # than the prefetch. Anything else is a corrupt object, and
                # asking again would not help.
                span = metadata_span(head)
                if span <= len(head):
                    raise
                head = self._ranged_get(obj_key, span)
                metadata, _ = unpack_entry(head, with_payload=False)
        except self.botocore_exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchKey'):
                return None
            raise CacheBackendError(
                f"S3 get_metadata() failed for key {key!r}: {e}"
            ) from e
        except (CorruptEntry, pickle.UnpicklingError, KeyError, TypeError, OSError) as e:
            logger.debug("S3 get_metadata() error for key %s: %s", key, e)
            return None

        metadata.setdefault('source', self.source_label)
        return metadata

    def _ranged_get(self, obj_key: str, nbytes: int) -> bytes:
        """The first *nbytes* of an object. S3 clamps a range past the end."""
        obj = self.s3.get_object(Bucket=self.bucket, Key=obj_key,
                                 Range=f"bytes=0-{nbytes - 1}")
        return obj['Body'].read()

    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None) -> None:
        """Serialize on the calling thread, run the S3 PUT in background."""
        obj_key = self._get_key(key)

        metadata = self._init_metadata(metadata, key)

        if serializer is None:
            serializer = PickleSerializer()

        # IMPORTANT: serialize on the calling thread.
        serialized_value = serializer.serialize(value)

        metadata['size'] = len(serialized_value)
        if 'storage' not in metadata:
            metadata['storage'] = [self.source_label]

        self._writes.submit(
            key, self._do_set_sync,
            obj_key, pack_entry(dict(metadata), serialized_value),
        )

    def _do_set_sync(self, obj_key: str, blob: bytes) -> None:
        """The actual S3 PUT -- runs in the PendingWrites worker thread.

        One object, so one request, and no ordering to reason about. The
        two-object version had to PUT the data first and the metadata second
        so a reader could never find metadata pointing at a payload that was
        not there yet, and had to delete the orphan when the second PUT
        failed.
        """
        try:
            self.s3.put_object(Bucket=self.bucket, Key=obj_key, Body=blob)
        except self.botocore_exceptions.ClientError as e:
            raise CacheBackendError(f"S3 Write Error: {e}") from e

    def set_metadata_only(self, key: str, metadata: dict) -> None:
        """Store metadata with no value, for an entry too large to persist.

        Will NOT overwrite an entry that carries a real payload.
        """
        self._writes.wait(key)
        obj_key = self._get_key(key)

        existing = self.get_metadata(key)
        if existing is not None and not existing.get('metadata_only'):
            return

        metadata = dict(metadata)
        metadata['key'] = key
        metadata['metadata_only'] = True
        metadata.setdefault('size', 0)

        try:
            self.s3.put_object(Bucket=self.bucket, Key=obj_key,
                               Body=pack_entry(metadata, b""))
        except self.botocore_exceptions.ClientError as e:
            logger.debug("Failed to write metadata-only entry for key %r: %s", key, e)

    def delete(self, key: str) -> None:
        # Drain any pending write so the delete actually deletes.
        self._writes.drain(key)
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=self._get_key(key))
        except self.botocore_exceptions.ClientError as e:
            raise CacheBackendError(
                f"S3 delete failed for key '{key}': {e}"
            ) from e

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
                        if not key.endswith(ENTRY_SUFFIX):
                            continue        # a stray, or a pre-v2 .meta/.data
                        try:
                            # Ranged: listing a cache must not download it. The
                            # two-object version fetched whole .meta objects,
                            # which was already bounded -- this keeps that
                            # property now that metadata shares an object with
                            # the value.
                            head = self._ranged_get(key, self.METADATA_PREFETCH_BYTES)
                            metadata, _ = unpack_entry(head, with_payload=False)
                            entries.append(metadata)
                        except (CorruptEntry, pickle.UnpicklingError,
                                self.botocore_exceptions.ClientError, KeyError) as e:
                            logger.debug("Skipping unreadable S3 entry %s: %s", key, e)
        except self.botocore_exceptions.ClientError as e:
            raise CacheBackendError(
                f"S3 list_entries failed for prefix '{self.prefix}': {e}"
            ) from e
        return entries

    def shutdown(self) -> None:
        self._writes.shutdown(wait=True)
