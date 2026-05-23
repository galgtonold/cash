"""S3-based cache backend for Cash."""

from __future__ import annotations

import logging
import pickle
from typing import Any

from cash.exceptions import CacheBackendError, DependencyNotFoundError

from ._base import CacheBackend, CacheMetadata
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

    def _get_keys(self, key: str) -> tuple[str, str]:
        # S3 keys (paths)
        return f"{self.prefix}{key}.meta", f"{self.prefix}{key}.data"

    def get(self, key: str) -> tuple[CacheMetadata | None, Any | None]:
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

    def set(self, key: str, value: Any, metadata: CacheMetadata | None = None, serializer: Serializer | None = None) -> None:
        meta_key, data_key = self._get_keys(key)

        metadata = self._init_metadata(metadata, key)

        if serializer is None:
            serializer = PickleSerializer()

        serialized_value = serializer.serialize(value)

        metadata['size'] = len(serialized_value)
        if 'storage' not in metadata:
            metadata['storage'] = [self.source_label]
        meta_bytes = pickle.dumps(metadata)

        # Upload data
        try:
            self.s3.put_object(Bucket=self.bucket, Key=data_key, Body=serialized_value)

            # Upload metadata (only after data succeeds)
            self.s3.put_object(Bucket=self.bucket, Key=meta_key, Body=meta_bytes)
        except self.botocore_exceptions.ClientError as e:
            # Clean up if partial write?
            # If data wrote but meta failed, it's orphaned but harmless as we need meta to find it.
            # If we want to be clean, we could delete data_key.
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=data_key)
            except self.botocore_exceptions.ClientError:
                logger.warning("Failed to clean up partial S3 write for key %s", data_key)
            raise CacheBackendError(f"S3 Write Error: {e}") from e

    def delete(self, key: str) -> None:
        meta_key, data_key = self._get_keys(key)
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=meta_key)
            self.s3.delete_object(Bucket=self.bucket, Key=data_key)
        except self.botocore_exceptions.ClientError as e:
            raise CacheBackendError(
                f"S3 delete failed for key '{key}': {e}"
            ) from e

    def clear(self) -> None:
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
        pass
