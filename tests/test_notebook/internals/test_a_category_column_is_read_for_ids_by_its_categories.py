"""A pandas category column is read for ids by its categories: every value
is one of them, the very object. Made objects one by one, a column of a
million rows of three categories took 40 ms of every check."""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders

pd = pytest.importorskip("pandas")


class _Thing:
    pass


@pytest.fixture
def no_values_made(monkeypatch):
    """Reading a category column's values as objects fails the test."""

    def refuse(self, *args, **kwargs):
        raise AssertionError("the category column's values were made objects")

    monkeypatch.setattr(pd.Categorical, "to_numpy", refuse)
    monkeypatch.setattr(pd.Categorical, "__array__", refuse)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda thing: pd.DataFrame({"c": pd.Categorical([repr(thing), "x"] * 50)}), id="frame of text"),
        pytest.param(lambda thing: pd.Series(pd.Categorical([id(thing), 5] * 50)), id="series of ids"),
        pytest.param(lambda thing: pd.Categorical(["x", f"o{id(thing)}"]), id="array"),
    ],
)
def test_an_id_among_the_categories_is_found(no_values_made, make):
    thing = _Thing()
    reason = identity_holders.identity_held_reason("v", make(thing), {"thing": thing})
    assert reason is not None and reason.kind == "an id()"
