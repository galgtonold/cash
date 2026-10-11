"""Looking for a Figure or Axes inside a value does not visit every plain item.

Every statement that names a variable asks whether it holds a matplotlib
Figure or Axes, because drawing on one must not be cached. For a parsed log of
two million (action, datetime) pairs that question was a Python call per item:
19 s for `print(logs[-1])`, again for every later statement that named `logs`.
Plain data holds no object, and a value cannot be an instance of a class whose
module was never imported, so neither needs the walk.
"""

from __future__ import annotations

import datetime

import pytest

from cash.analysis import cacheability_decision as decision

pytestmark = [pytest.mark.core]


def _log(n: int = 300) -> list:
    stamp = datetime.datetime(2019, 10, 15, 18, 8, 2)
    return [(f"@User{i}", [(f"Action_{j}", stamp) for j in range(4)]) for i in range(n)]


@pytest.fixture
def visited(monkeypatch) -> list:
    """Every value `_coupled_kind` is asked about."""
    asked: list = []
    real = decision._coupled_kind

    def counting(value):
        asked.append(value)
        return real(value)

    monkeypatch.setattr(decision, "_coupled_kind", counting)
    return asked


def test_a_parsed_log_is_answered_without_visiting_its_items(visited):
    pytest.importorskip("matplotlib.figure")
    assert decision.identity_coupled_reason("logs", _log()) is None
    assert decision.receiver_is_identity_coupled(_log()) is False
    assert len(visited) == 2, "only each value itself was looked at"


def test_json_like_records_are_answered_without_visiting_their_items(visited):
    pytest.importorskip("matplotlib.figure")
    records = [{"id": i, "tags": ["a", "b"], "when": datetime.date(2020, 1, 1)} for i in range(300)]
    assert decision.identity_coupled_reason("records", records) is None
    assert len(visited) == 1


def test_nothing_is_visited_while_no_class_of_the_kind_is_imported(visited, monkeypatch):
    monkeypatch.setattr(decision, "_COUPLED_MODULES", ("a_module_that_is_not_imported.figure",))
    holder = [object(), {"k": object()}, [object()]]
    assert decision.identity_coupled_reason("holder", holder) is None
    assert len(visited) == 1


def test_an_axes_among_plain_bulk_is_still_found():
    figure_mod = pytest.importorskip("matplotlib.figure")
    ax = figure_mod.Figure().add_subplot()
    assert decision.identity_coupled_reason("v", [_log(), {"ax": ax}]) is not None
    assert decision.identity_coupled_reason("v", [_log(), [(1, [ax])]]) is not None
    assert decision.receiver_is_identity_coupled([[1, 2, 3], ({"n": 1}, [ax])]) is True


def test_a_container_with_an_object_beside_plain_bulk_still_gets_the_walk(visited):
    pytest.importorskip("matplotlib.figure")

    class Thing:
        pass

    assert decision.identity_coupled_reason("v", [_log(10), Thing()]) is None
    assert any(isinstance(v, Thing) for v in visited), "an object in the value is looked at"


def test_a_list_of_numpy_scalars_is_asked_once_per_type(visited):
    """``plt.hist(deltas)`` over 1.7 million numpy timedeltas took 3.6 s
    longer than plain, asking each item."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("matplotlib.figure")
    deltas = [np.timedelta64(i, "s") for i in range(300)] + [np.float64(i) for i in range(300)]
    assert decision.receiver_is_identity_coupled(deltas) is False
    assert len(visited) <= 4


def test_an_axes_after_many_scalars_or_in_an_object_array_is_still_found():
    np = pytest.importorskip("numpy")
    figure_mod = pytest.importorskip("matplotlib.figure")
    ax = figure_mod.Figure().add_subplot()
    assert decision.receiver_is_identity_coupled([np.float64(i) for i in range(300)] + [ax]) is True
    holder = np.empty(1, dtype=object)
    holder[0] = ax
    assert decision.receiver_is_identity_coupled([np.zeros(3), np.zeros(2), holder]) is True
