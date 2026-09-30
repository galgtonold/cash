"""Tests for serialization functionality."""

import pytest


def test_pickle_round_trip():
    from cash.backends.serialization import PickleSerializer

    serializer = PickleSerializer()
    data = {"a": 1, "b": 2, "nested": [1, 2, 3]}

    serialized = serializer.serialize(data)
    assert isinstance(serialized, bytes)

    restored = serializer.deserialize(serialized)
    assert data == restored


# --- Corruption and edge-case tests ---


def test_pickle_deserialize_corrupted_bytes():
    """PickleSerializer.deserialize() raises on corrupted / truncated bytes."""
    import pickle

    from cash.backends.serialization import PickleSerializer

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


def test_pickle_round_trip_complex_object():
    """PickleSerializer round-trips a complex nested object exactly."""
    from cash.backends.serialization import PickleSerializer

    serializer = PickleSerializer()
    obj = {"nested": [1, (2, 3), {"inner": True}], "unicode": "héllo wörld"}
    assert serializer.deserialize(serializer.serialize(obj)) == obj


def test_a_dataframe_is_pickled_with_protocol_5(sample_dataframe):
    """Protocol 5 writes a frame's column buffers in one piece; protocol 4
    copied them and was ~10x slower on a million-row frame."""
    import pickletools

    import pandas as pd

    from cash.backends.serialization import PickleSerializer

    serializer = PickleSerializer()
    raw = serializer.serialize(sample_dataframe)
    assert next(pickletools.genops(raw))[1] == 5
    pd.testing.assert_frame_equal(serializer.deserialize(raw), sample_dataframe)
