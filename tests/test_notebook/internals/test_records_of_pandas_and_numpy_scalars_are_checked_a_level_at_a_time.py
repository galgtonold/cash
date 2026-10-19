"""Records holding a ``pd.Timestamp``, ``NaT`` or a numpy number are read a
level at a time, as records of plain values are.

``pd.Timestamp`` subclasses ``datetime`` and ``np.float64`` ``float``, so no
check that asks a type by its exact type took them for values: a loop that
parsed a log into rows ``{'user': ..., 'action_time': pd.Timestamp(...)}``
had each row walked one container at a time, by every check of every
statement, and ran 1.5x plain (JuNE expert_2's two biggest cells). These
tests pin that the per-container walks are not taken for such records, and
that what the walks look for is still found inside them.
"""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def _records(n: int) -> list:
    when = pd.Timestamp("2021-05-03 01:43:11")
    return [
        {"user": f"@User{i}", "session": i, "time": when if i % 7 else pd.NaT, "score": np.float64(i)} for i in range(n)
    ]


def _counting_children(monkeypatch) -> list:
    asked: list = []
    real = shared_objects.children_of

    def counting(value):
        asked.append(1)
        return real(value)

    monkeypatch.setattr(shared_objects, "children_of", counting)
    return asked


def test_the_library_scalars_are_exact_value_types():
    exact = shared_objects.exact_value_types()
    for value in (pd.Timestamp(0), pd.Timedelta(1), pd.NaT, pd.NA, np.float64(1), np.int64(1), np.bool_(True)):
        assert type(value) in exact
        assert shared_objects.is_value(value, shared_objects.VALUE_TYPES + shared_objects.library_value_types())
    # A structured array's row is a view of its memory: still asked one by one.
    assert np.void not in exact


def test_the_holder_check_does_not_walk_each_record(monkeypatch):
    namespace = {"rows": _records(5_000)}
    asked = _counting_children(monkeypatch)

    assert shared_objects.shared_names({"rows": namespace["rows"]}, (namespace,)) == set()
    assert len(asked) < 10


def test_the_holder_check_still_finds_a_record_another_name_holds(monkeypatch):
    namespace = {"rows": _records(5_000)}
    namespace["row"] = namespace["rows"][-1]
    asked = _counting_children(monkeypatch)

    holders, shared = shared_objects.share_group(["rows"], {"rows": namespace["rows"]}, namespace)

    assert (sorted(holders), shared) == (["row"], set())
    assert len(asked) < 50


def test_the_view_check_does_not_walk_records(monkeypatch):
    from cash.notebook.statement import derivation_edges

    walks = []
    real = derivation_edges._aliases_in

    def counting(value, *args):
        walks.append(1)
        return real(value, *args)

    monkeypatch.setattr(derivation_edges, "_aliases_in", counting)
    rows = _records(5_000)
    assert not derivation_edges.is_uncacheable_alias(rows, {"rows": rows})
    assert walks == []

    base = np.arange(10)
    rows[-1]["window"] = base[:3]
    assert derivation_edges.is_uncacheable_alias(rows, {"rows": rows, "base": base})


def test_a_view_inside_records_is_still_found(monkeypatch):
    base = np.arange(10.0)
    rows = _records(5_000)
    asked = _counting_children(monkeypatch)
    assert shared_objects.view_base_names([rows], {"rows": rows, "base": base}) == set()
    assert len(asked) == 0

    rows[-1]["window"] = base[2:5]
    assert shared_objects.view_base_names([rows], {"rows": rows, "base": base}) == {"base"}


def test_memory_borrowed_inside_records_is_still_found():
    from cash.notebook.borrowed_memory import borrowed_memory_names

    arr = np.arange(100.0)
    rows = _records(5_000)
    assert borrowed_memory_names([rows], {"arr": arr}) == set()

    rows[-1]["series"] = pd.Series(arr, copy=False)
    assert borrowed_memory_names([rows], {"arr": arr}) == {"arr"}


def test_the_figure_check_does_not_visit_each_record(monkeypatch):
    figure_mod = pytest.importorskip("matplotlib.figure")
    from cash.analysis import cacheability_decision as decision

    asked: list = []
    real = decision._coupled_kind

    def counting(value):
        asked.append(value)
        return real(value)

    monkeypatch.setattr(decision, "_coupled_kind", counting)
    rows = _records(5_000)
    assert decision.identity_coupled_reason("rows", rows) is None
    assert len(asked) == 1

    rows[-1]["ax"] = figure_mod.Figure().add_subplot()
    assert decision.identity_coupled_reason("rows", rows) is not None


def test_a_statements_look_answers_for_the_wider_leaves(monkeypatch):
    # The checks ask with the library scalars among the leaves; the facts a
    # statement's one look found over the plain leaves still answer them.
    from cash import _plain_data

    rows = [{"id": i, "tags": ["a"]} for i in range(1_000)]
    walks: list = []
    real = _plain_data.tree_levels

    def counting(*args, **kwargs):
        walks.append(1)
        return real(*args, **kwargs)

    with _plain_data.one_look({"rows": rows}):
        _plain_data.tree_facts(rows)
        monkeypatch.setattr(_plain_data, "tree_levels", counting)
        assert _plain_data.is_tree(rows, shared_objects.tree_leaves())
        assert _plain_data.held_only_by_parents(rows, shared_objects.tree_leaves())
    assert walks == []
