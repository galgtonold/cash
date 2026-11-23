"""Positive controls for ``STORE-IDENTITY-HELD``
(``test_a_result_holding_a_live_objects_id_is_not_stored.py``): a result
whose numbers or digits are no live object's ``id()``, and the big results
the check reads at C speed, are still stored and served, with no warning.
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")


def _category_frame(n):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"c": pd.Categorical(["a", "b"] * 500), "n": n})


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda n: n * 1_000_000_008, id="a big multiple of 8"),
        pytest.param(lambda n: [f"order-{140_000_000_000_000 + 16 * n}", 2**40], id="digits in text"),
        pytest.param(lambda n: np.arange(200_000, dtype=np.int64) * 8 + n, id="an int64 array"),
        pytest.param(lambda n: [f"s{i}-{n}" for i in range(100_000)], id="100k small strings"),
        pytest.param(lambda n: _category_frame(n), id="a category column"),
        pytest.param(lambda n: {"rows": [{"id": i, "v": i * 0.5} for i in range(2_000)], "n": n}, id="records"),
    ],
)
def test_an_ordinary_result_is_still_stored_and_served(disk_cash, make):
    """Positive control: numbers and digits that are no live object's id."""
    runs = []

    @disk_cash.cache
    def f(n):
        runs.append(n)
        return make(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f(3)
        f(3)
    assert "STORE-IDENTITY-HELD" not in _codes(caught)
    assert runs == [3], "an ordinary result was not served from the cache"
