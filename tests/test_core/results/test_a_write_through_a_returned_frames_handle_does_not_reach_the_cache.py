"""Writing into a returned frame through ``.values`` / ``.array`` never
changes the next call's result (the RAM tier hands out deep copies)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


@pytest.mark.parametrize("edited", ["miss", "hit"])
def test_a_handle_write_does_not_reach_later_hits(cash_instance, edited):
    @cash_instance.cache
    def load_scores():
        return pd.DataFrame(
            {
                "score": pd.array([1, 2, 3], dtype="Int64"),
                "grade": pd.Categorical(["a", "b", "a"]),
                "w": [1.0, 2.0, 3.0],
            }
        )

    first = load_scores()
    df = first if edited == "miss" else load_scores()
    df["score"].values[0] = 100
    df["grade"].values[0] = "b"
    df["w"].array[0] = 50.0
    again = load_scores()
    assert (int(again["score"][0]), again["grade"][0], float(again["w"][0])) == (1, "a", 1.0)
    assert load_scores.cache_info()["hits"] >= 1
