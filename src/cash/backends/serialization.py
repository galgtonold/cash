"""Serialization strategies for cache value persistence.

Provides `Serializer` (abstract base), `PickleSerializer`,
`JSONSerializer`, and `CloudPickleSerializer`.
"""

from __future__ import annotations

import io
import pickle
from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Serializer", "PickleSerializer", "CloudPickleSerializer", "ParquetSerializer"]

class Serializer(ABC):
    """Abstract base class for serializers."""

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""

class PickleSerializer(Serializer):
    """Serializer using Python's built-in pickle module."""

    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class CloudPickleSerializer(Serializer):
    """Serializer using cloudpickle for lambda functions, closures, and dynamic classes.

    Requires: pip install cloudpickle
    Falls back to standard pickle if cloudpickle is not installed.
    """

    def serialize(self, data: Any) -> bytes:
        try:
            import cloudpickle
            return cloudpickle.dumps(data)
        except ImportError:
            return pickle.dumps(data)

    def deserialize(self, data: bytes) -> Any:
        # cloudpickle-serialized data can be deserialized with standard pickle
        return pickle.loads(data)

class ParquetSerializer(Serializer):
    """Serializer for pandas DataFrames using Parquet."""

    def serialize(self, data: Any) -> bytes:
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=True)
        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("ParquetSerializer requires pandas: pip install pandas") from exc
        buffer = io.BytesIO(data)
        return pd.read_parquet(buffer)

def get_serializer(data: Any) -> Serializer:
    """Factory to get the appropriate serializer for the data."""
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            try:
                import pyarrow  # noqa: F401
                return ParquetSerializer()
            except ImportError:
                try:
                    import fastparquet  # noqa: F401
                    return ParquetSerializer()
                except ImportError:
                    pass
            # Fall back to pickle for DataFrames without parquet support
            return PickleSerializer()
    except ImportError:
        pass

    return PickleSerializer()
