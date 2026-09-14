"""A frame is sized from its column arrays, to ``memory_usage(deep=True)``'s number.

Cash sized every stored frame twice -- the RAM tier's cap and the restore-cost
estimate -- through ``memory_usage``, which spends nearly all its time building
a result Series: 18% of a loop over a thousand small files (round 23).
"""
from __future__ import annotations

import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import _sizing  # noqa: E402
from cash._sizing import pandas_nbytes  # noqa: E402
from cash.backends.memory_backend import InMemoryBackend  # noqa: E402
from cash.notebook.object_hashing import estimate_object_size  # noqa: E402

pytestmark = [pytest.mark.core]

N = 3000


def _cases():
    rng = np.random.default_rng(0)
    yield "numeric", pd.DataFrame({"a": np.arange(N), "b": rng.random(N)})
    yield "python strings", pd.DataFrame({"s": pd.array([f"x{i}" for i in range(N)], dtype="string[python]")})
    yield "object strings", pd.DataFrame({"s": np.array([f"row-{i:06d}" for i in range(N)], dtype=object)})
    yield "category", pd.DataFrame({"c": pd.Categorical(["a", "b", "c"] * (N // 3))})
    yield "datetime tz", pd.DataFrame({"t": pd.date_range("2026-01-01", periods=N, freq="min", tz="UTC")})
    yield "object index", pd.DataFrame({"a": np.arange(N)}, index=[f"k{i}" for i in range(N)])
    yield "multiindex", pd.DataFrame({"a": np.arange(N)},
                                     index=pd.MultiIndex.from_arrays([np.arange(N) % 7, np.arange(N)]))
    yield "empty", pd.DataFrame({"a": pd.Series([], dtype="int64")})
    yield "nullable int", pd.DataFrame({"a": pd.array(range(N), dtype="Int64")})
    yield "series", pd.Series(np.array([f"v{i}" for i in range(N)], dtype=object))
    try:
        yield "arrow strings", pd.DataFrame({"s": pd.array([f"x{i}" for i in range(N)], dtype="string[pyarrow]")})
    except ImportError:
        pass


@pytest.fixture(params=["default strings", "object strings (pandas < 3)"])
def string_inference(request):
    """pandas 3 stores strings in Arrow; pandas 2 -- CI's Python 3.10 jobs --
    keeps them as Python objects, in columns and in categories alike. Both,
    wherever the running pandas can be asked for the old behaviour."""
    if request.param.startswith("object"):
        try:
            with pd.option_context("future.infer_string", False):
                yield
        except (KeyError, pd.errors.OptionError):
            pytest.skip("this pandas has no future.infer_string")
    else:
        yield


def test_the_size_is_memory_usage_deep(string_inference):
    for name, obj in _cases():
        usage = obj.memory_usage(deep=True)
        exact = int(usage.sum()) if isinstance(obj, pd.DataFrame) else int(usage)
        assert pandas_nbytes(obj) == pytest.approx(exact, rel=0.03), name


def test_neither_sizing_builds_a_memory_usage_series(monkeypatch):
    frame = pd.DataFrame({"a": np.arange(N), "s": [f"x{i}" for i in range(N)]})
    expected = pandas_nbytes(frame)

    def refuse(*_a, **_k):
        raise AssertionError("memory_usage was called")
    monkeypatch.setattr(pd.DataFrame, "memory_usage", refuse)
    monkeypatch.setattr(pd.Series, "memory_usage", refuse)
    assert estimate_object_size(frame) == expected
    b = InMemoryBackend()
    b.set("k", {"variables": {"df": frame}})
    assert b._store["k"][0]["size"] >= expected


def test_python_objects_are_sampled_not_walked(monkeypatch):
    """deep=True walks every value; a sample answers the order of magnitude.

    The column repeats every 5 rows, the shape a fixed stride aliases on."""
    looked = []
    real = sys.getsizeof
    monkeypatch.setattr(_sizing.sys, "getsizeof", lambda o: looked.append(1) or real(o))
    col = np.array([f"row-{i}" * (1 + i % 5) for i in range(200_000)], dtype=object)
    frame = pd.DataFrame({"s": pd.Series(col, dtype=object)})
    assert frame["s"].dtype == object
    size = pandas_nbytes(frame)
    assert len(looked) <= _sizing._OBJECT_SAMPLE
    exact = int(frame.memory_usage(deep=True).sum())
    assert 0.85 * exact < size < 1.15 * exact


def test_anything_else_is_not_its_business():
    assert pandas_nbytes([1, 2, 3]) is None
    assert pandas_nbytes(np.arange(3)) is None
