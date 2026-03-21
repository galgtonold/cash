"""Tests for serialization functionality."""
import pytest


def _has_parquet_support():
    """Check if pyarrow or fastparquet is available."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def test_get_serializer_dataframe(sample_dataframe):
    from cash.backends.serialization import get_serializer, ParquetSerializer, PickleSerializer
    
    serializer = get_serializer(sample_dataframe)
    
    # Should return ParquetSerializer if pyarrow/fastparquet available
    # Otherwise PickleSerializer (fallback)
    if _has_parquet_support():
        assert isinstance(serializer, ParquetSerializer)
    else:
        assert isinstance(serializer, PickleSerializer)


@pytest.mark.skipif(not _has_parquet_support(), reason="pyarrow or fastparquet required")
def test_parquet_serialization(sample_dataframe):
    import pandas as pd
    from cash.backends.serialization import ParquetSerializer
    
    df = sample_dataframe.copy()
    df['strings'] = ['x', 'y', 'z']
    
    serializer = ParquetSerializer()
    data = serializer.serialize(df)
    
    assert isinstance(data, bytes)
    
    df_restored = serializer.deserialize(data)
    pd.testing.assert_frame_equal(df, df_restored)


def test_pickle_fallback():
    from cash.backends.serialization import PickleSerializer
    
    serializer = PickleSerializer()
    data = {'a': 1, 'b': 2, 'nested': [1, 2, 3]}
    
    serialized = serializer.serialize(data)
    assert isinstance(serialized, bytes)
    
    restored = serializer.deserialize(serialized)
    assert data == restored


# --- Corruption and edge-case tests ---

def test_pickle_deserialize_corrupted_bytes():
    """PickleSerializer.deserialize() raises on corrupted / truncated bytes."""
    from cash.backends.serialization import PickleSerializer
    import pickle

    serializer = PickleSerializer()
    with pytest.raises((pickle.UnpicklingError, EOFError, Exception)):
        serializer.deserialize(b"this is definitely not pickle data \x00\xff")


def test_pickle_deserialize_truncated():
    """PickleSerializer.deserialize() raises on truncated pickle stream."""
    from cash.backends.serialization import PickleSerializer

    serializer = PickleSerializer()
    good_bytes = serializer.serialize({"key": "value"})
    with pytest.raises(Exception):
        serializer.deserialize(good_bytes[:4])  # strip most of the payload


def test_get_serializer_non_dataframe_returns_pickle():
    """get_serializer returns PickleSerializer for non-DataFrame types."""
    from cash.backends.serialization import get_serializer, PickleSerializer

    for obj in [None, 42, "hello", [1, 2, 3], {"a": 1}]:
        serializer = get_serializer(obj)
        assert isinstance(serializer, PickleSerializer), (
            f"Expected PickleSerializer for {type(obj).__name__}, got {type(serializer).__name__}"
        )


def test_get_serializer_numpy_array_returns_pickle():
    """get_serializer returns PickleSerializer for numpy arrays (not DataFrames)."""
    pytest.importorskip("numpy")
    import numpy as np
    from cash.backends.serialization import get_serializer, PickleSerializer

    arr = np.array([1, 2, 3])
    serializer = get_serializer(arr)
    assert isinstance(serializer, PickleSerializer)


@pytest.mark.skipif(not _has_parquet_support(), reason="pyarrow or fastparquet required")
def test_parquet_deserialize_corrupted_bytes():
    """ParquetSerializer.deserialize() raises on corrupted bytes."""
    from cash.backends.serialization import ParquetSerializer

    serializer = ParquetSerializer()
    with pytest.raises(Exception):
        serializer.deserialize(b"not parquet data at all \x42\x00\xff")


def test_pickle_round_trip_complex_object():
    """PickleSerializer round-trips a complex nested object exactly."""
    from cash.backends.serialization import PickleSerializer

    serializer = PickleSerializer()
    obj = {"nested": [1, (2, 3), {"inner": True}], "unicode": "héllo wörld"}
    assert serializer.deserialize(serializer.serialize(obj)) == obj
