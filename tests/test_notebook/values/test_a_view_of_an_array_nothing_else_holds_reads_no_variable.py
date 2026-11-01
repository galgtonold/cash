"""A view whose base array nothing but the view holds needs no search of the
notebook for what else holds that array.

``predictions = model.predict(X)`` returns a view of an array made inside
``predict``. After every such statement cash walked each variable of the
notebook for one holding that array; in a notebook of fitted models the
walk ran past the statement's 0.1 s budget, the statement's reads were
linked to what it made (``model`` to ``housing``), and on the next Run All
every fit below them ran again.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import _bases_held_by_views_alone, view_base_names

np = pytest.importorskip("numpy")


def test_a_view_of_an_array_made_inside_a_call_is_held_by_it_alone():
    view = np.arange(10.0).reshape(2, 5)
    assert view.base is not None
    assert _bases_held_by_views_alone([view])
    assert view_base_names([view], {"view": view}) == set()


def test_a_view_of_a_named_array_is_not_held_by_it_alone():
    big = np.arange(10.0)
    head = big[2:5]
    assert not _bases_held_by_views_alone([head])
    assert view_base_names([head], {"big": big, "head": head}) == {"big", "head"}


def test_two_views_of_one_array_hold_it_together():
    first, second = np.split(np.arange(6.0), 2)
    assert _bases_held_by_views_alone([first, second])
    # Asked of one, the other is a holder it does not know.
    assert not _bases_held_by_views_alone([first])


def test_an_array_held_in_a_container_is_still_found():
    data = {"x": np.arange(10.0)}
    view = data["x"][2:5]
    assert view_base_names([view], {"data": data, "view": view}) == {"data", "view"}


def test_a_frame_of_text_is_read_past_its_strings():
    pd = pytest.importorskip("pandas")
    held = np.arange(10.0)
    frame = pd.DataFrame({"t": [f"s{i}" for i in range(5_000)], "o": [None] * 4_999 + [held]})
    view = held[1:3]
    assert "frame" in view_base_names([view], {"frame": frame, "view": view, "held": held})


def test_the_variables_reaching_an_object_are_found_down_from_them():
    from cash.notebook.shared_objects import names_reaching

    class Step:
        def __init__(self, inner=None):
            self.inner = inner

    target = Step()
    # As a pipeline holds its steps: a list of (name, step) pairs, five
    # levels down, past a column of text.
    pipeline = Step([("a", Step(Step([("b", target)])))])
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"t": [f"s{i}" for i in range(5_000)]})
    ns = {"pipeline": pipeline, "frame": frame, "target": target, "other": Step([1, 2])}
    assert names_reaching({id(target)}, ns) == {"pipeline", "target"}
    assert names_reaching({id(target)}, ns, max_objects=2) is None
