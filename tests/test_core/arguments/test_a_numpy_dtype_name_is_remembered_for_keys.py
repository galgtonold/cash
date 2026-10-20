"""An array's key names its dtype as ``str(dtype)`` does, remembered for
numpy's own scalar dtypes.

numpy writes a dtype's name in Python, about half of what a small array's
key took; p25 keys a list of 24,000 small arrays twice per call. The name
remembered is the one ``str`` gives, so every key stays the same.
"""

import pytest

np = pytest.importorskip("numpy")

from cash import content_hashers

DTYPES = [
    np.dtype("uint32"),
    np.dtype("<u4"),
    np.dtype("I"),
    np.dtype("int64"),
    np.dtype("q"),
    np.dtype("l"),
    np.dtype("float64"),
    np.dtype("d"),
    np.dtype("bool"),
    np.dtype("complex64"),
    np.dtype(">u4"),
    np.dtype("<U5"),
    np.dtype("S3"),
    np.dtype("datetime64[ns]"),
    np.dtype("m8[s]"),
    np.dtype([("a", "<i4"), ("b", "<f8")]),
    np.dtype("int64", metadata={"unit": "m"}),
    np.dtype("float16"),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_the_name_is_the_one_str_gives(dtype, monkeypatch):
    monkeypatch.setattr(content_hashers, "_DTYPE_TEXTS", {})
    for other in DTYPES:  # whatever was met before
        content_hashers._dtype_text(other)
    assert content_hashers._dtype_text(dtype) == str(dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_an_array_keys_as_before(dtype, monkeypatch):
    value = np.zeros(4, dtype=dtype)
    with monkeypatch.context() as m:
        m.setattr(content_hashers, "_dtype_text", str)
        expected = content_hashers.hash_numpy(value)
    for other in DTYPES:
        content_hashers._dtype_text(other)
    assert content_hashers.hash_numpy(value) == expected


def test_only_numpy_s_own_scalar_dtypes_are_remembered(monkeypatch):
    monkeypatch.setattr(content_hashers, "_DTYPE_TEXTS", {})
    for dtype in DTYPES:
        content_hashers._dtype_text(dtype)
    kept = set(content_hashers._DTYPE_TEXTS)
    assert np.dtype("uint32") in kept
    for dtype in (np.dtype(">u4"), np.dtype("<U5"), np.dtype([("a", "<i4"), ("b", "<f8")])):
        assert dtype not in kept
    assert all(d.isbuiltin == 1 and not d.metadata for d in kept)
