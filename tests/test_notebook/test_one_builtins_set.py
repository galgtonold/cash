"""One set of builtin names, and one MRO matcher.

The upstream simulation skipped names from a hand-picked list of 35 (its
comment said it mirrored a module that no longer existed), while the runtime's
cacheability decision and the decorator's analyzer used ``dir(builtins)``. So
``abs``, ``round`` or ``any`` in a statement was a builtin to the runtime and a
missing input to the simulation, which then could not verify an unsaved edit
reading it; and a user's own ``max = ...`` was skipped by the simulation
whatever it held. Two copies of the "first class in the MRO named in a table"
walk had the same shape and drifted the same way.
"""

from __future__ import annotations

import builtins
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from cash.analysis.cacheability_decision import _coupled_kind
from cash.notebook.stateful_carriers import stateful_carrier_kind
from cash.notebook.upstream.virtual_lineage import VirtualLineage
from cash.value_types import BUILTIN_NAMES, mro_kind

SRC = Path(__file__).resolve().parents[2] / "src" / "cash"


def test_every_builtin_and_the_ipython_names_are_in_it():
    # IPython adds names to `builtins` while it runs (`display`), so compare
    # with the names every interpreter has.
    assert set(dir(builtins)) >= {"abs", "round", "any", "max", "print", "ValueError"}
    assert {"abs", "round", "any", "max", "print", "ValueError", "get_ipython", "__builtins__"} <= BUILTIN_NAMES


def test_no_module_keeps_its_own_copy():
    copies = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name != "value_types.py" and re.search(r"frozenset\(\s*dir\(\s*_?builtins\s*\)", path.read_text())
    ]
    assert copies == []


def test_a_name_the_user_bound_is_not_a_builtin_to_the_simulation():
    sim = SimpleNamespace(variable_lineage={"max": "lineage-of-max"})
    unbound = VirtualLineage._unbound_builtin
    assert unbound(sim, "abs")
    assert not unbound(sim, "max"), "the kernel holds a max of the user's"
    assert not unbound(sim, "id", {"id": "bound-above"}), "a statement above binds id"
    assert not unbound(sim, "frame")


def test_the_mro_matcher_names_the_first_listed_class():
    class Base:
        pass

    class Child(Base):
        pass

    bases = {f"{Base.__module__}.{Base.__qualname__}": "a base"}
    prefix = (Base.__module__.split(".")[0],)
    assert mro_kind(Child(), bases, prefix) == "a base"
    assert mro_kind(Child(), bases, ("elsewhere",)) is None
    assert mro_kind(3, bases, prefix) is None


def test_both_callers_still_recognise_their_objects():
    np = pytest.importorskip("numpy")
    assert stateful_carrier_kind(np.random.default_rng(0)) is not None
    assert stateful_carrier_kind(3) is None
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        assert _coupled_kind(fig) == "matplotlib Figure"
        assert _coupled_kind(ax) == "matplotlib Axes"
        assert _coupled_kind([1, 2]) is None
    finally:
        plt.close(fig)
