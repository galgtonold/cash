"""What a kept plot run's last line echoed is kept as the value, which a
hit binds to ``_`` and ``Out`` as a run does, unless it holds an object of
matplotlib's: that is the live figure a run drew, which a copy is not, so
such a run is not kept. And the names a plot run may bind and still be
kept are the modules it imports, bound again to the module.
"""

from __future__ import annotations

import pickle
import types

import pytest

from cash.notebook.ipython import plot_runs


class Result:
    def __init__(self, tour, msec):
        self.tour = tour
        self.msec = msec


def test_a_plain_value_is_kept_and_rebuilt_equal():
    kept = plot_runs._echo_bytes({"tour": [1 + 2j, 3j], "msec": 4.0})
    assert kept is not None
    assert pickle.loads(kept) == {"tour": [1 + 2j, 3j], "msec": 4.0}


def test_an_object_of_the_users_is_kept():
    kept = plot_runs._echo_bytes(Result([1j], 2.0))
    assert kept is not None and pickle.loads(kept).tour == [1j]


@pytest.mark.parametrize("make", ["figure", "lines", "nested"])
def test_a_value_holding_matplotlib_objects_is_not_kept(make):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        lines = plt.plot([1, 2])
        value = {"figure": fig, "lines": lines, "nested": {"a": [Result(lines, 1.0)]}}[make]
        assert plot_runs._echo_bytes(value) is None
    finally:
        plt.close(fig)


def test_a_value_pickle_refuses_is_not_kept():
    assert plot_runs._echo_bytes(lambda: 1) is None


def test_only_names_bound_to_their_module_count_as_imports():
    ns = {"np_alias": types, "copy": types.ModuleType("not_imported"), "x": 1}
    assert plot_runs._bound_modules({"np_alias", "copy", "x"}, ns) == {"np_alias": "types"}
