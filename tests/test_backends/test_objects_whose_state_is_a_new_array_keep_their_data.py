"""A RAM hit keeps each object's own data when the object pickles its state
as an array or series it builds on the spot.

``__getstate__`` returning ``np.asarray(self.values)`` hands the pickler a
new array that is freed once written; the next object's array can get the
same address. The RAM copy maps arrays and frames to their copies by
address, so every object came back with the first one's data. A disk hit
was right.
"""

from __future__ import annotations

import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]


def _builders():
    np = pytest.importorskip("numpy")
    builders = {"ndarray": lambda values: np.asarray(values, dtype=float)}
    try:
        import pandas as pd

        builders["pandas"] = lambda values: pd.Series(values, dtype=float)
    except ImportError:  # pragma: no cover - pandas is a test dependency
        pass
    try:
        import polars as pl

        builders["polars"] = lambda values: pl.Series("v", values, dtype=pl.Float64)
    except ImportError:  # pragma: no cover - polars is a test dependency
        pass
    return builders


class _Signal:
    """Values kept as a list, pickled as one array built for the pickle."""

    build = None

    def __init__(self, values):
        self.values = list(values)

    def __getstate__(self):
        return type(self).build(self.values)

    def __setstate__(self, state):
        self.values = [float(v) for v in state]


@pytest.mark.parametrize("kind", ["ndarray", "pandas", "polars"])
def test_each_object_comes_back_with_its_own_data(kind, monkeypatch):
    builders = _builders()
    if kind not in builders:
        pytest.skip(f"{kind} is not installed")
    monkeypatch.setattr(_Signal, "build", staticmethod(builders[kind]))
    value = [_Signal([float(i), 2.0 * i]) for i in range(50)]
    expected = [s.values for s in value]

    b = InMemoryBackend()
    b.set("k", value, {"copy_required": True})
    for _ in range(2):
        assert [s.values for s in b.get("k")[1]] == expected
