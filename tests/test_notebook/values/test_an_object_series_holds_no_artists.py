"""An object column is no array of axes (`holds_artists`).

The check looked for ``.flat`` on anything with an object dtype and a size, so
a pandas Series of mixed values -- the row ``a`` of ``for _, a in
df.iterrows()`` -- raised AttributeError once the loop ended and its names
were checked for shared objects. The error came out as the loop's own, and
the upstream repair ran the loop again, adding its amounts twice.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash.notebook.pyplot_draws import holds_artists


@pytest.mark.parametrize(
    "value",
    [
        pd.Series(["2026-06", "opex", -50.0]),
        pd.Index(["a", "b"], dtype=object),
        pd.DataFrame({"a": ["x", "y"]}),
        pd.Series([1.0, 2.0]),
    ],
    ids=["object-series", "object-index", "object-frame", "float-series"],
)
def test_a_pandas_object_is_not_an_artist_holder(value):
    assert holds_artists(value) is False


def test_an_object_array_of_axes_still_holds_artists():
    fig, axes = plt.subplots(1, 2)
    try:
        assert holds_artists(axes)
        assert holds_artists(list(axes))
        assert holds_artists(axes[0])
        assert not holds_artists(np.array(["a", "b"], dtype=object))
    finally:
        plt.close(fig)
