"""A generator handed to a helper as an argument is drawn from, not changed.

``paths = simulate(n, rng)`` with ``rng = np.random.default_rng(7)`` in the
cell above is the usual Monte Carlo cell. The helper moves ``rng``'s stream,
and cash watched ``rng`` as an argument the call could change in place: the
draw read as an in-place change, ``rng``'s lineage was bumped as a mutated
receiver's, and the statement was never stored, also after a restart that
re-ran the seed. A draw is followed as a carrier's is instead
(``carrier_advances``): ``rng`` gets the lineage of a generator the statement
drew from, the entry stores where the call left it, and a hit puts it there,
so the draws below match a run without cash.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from cash.analysis.namespace_effects import call_arguments
from cash.notebook.cache_key import statement_source_hash
from tests._cell_driver import run_cash_cell

HELPER = "def simulate(n, rng):\n    z = rng.standard_normal((n, 3))\n    return z.sum(axis=1)"
SEED = "rng = np.random.default_rng(7)"
SIM = "paths = simulate(4, rng)"
AFTER = "after = rng.standard_normal(2)"
CELLS = [HELPER, SEED, SIM, AFTER]


def _plain(seed: int = 7):
    rng = np.random.default_rng(seed)
    paths = rng.standard_normal((4, 3)).sum(axis=1)
    return paths, rng.standard_normal(2), rng.bit_generator.state


def _users_function(source: str):
    ns: dict = {}
    exec(compile(source, "<cell>", "exec"), ns)
    return ns


# -- Which arguments a call is watched for -----------------------------------------


@pytest.mark.parametrize(
    "rng",
    [np.random.default_rng(0), np.random.RandomState(0), __import__("random").Random(0)],
    ids=["Generator", "RandomState", "random.Random"],
)
def test_a_generator_handed_to_a_helper_is_not_watched(rng):
    ns = _users_function("def simulate(n, rng, out):\n    out.append(rng)")
    ns.update(rng=rng, buf=[])
    tree = ast.parse("paths = simulate(4, rng, buf)")
    names = call_arguments(tree, ns)
    assert "rng" not in names
    assert "buf" in names, "control: the list handed over beside it is still watched"


def test_a_generator_handed_to_a_bare_call_is_not_watched():
    ns = _users_function("def burn(rng, out):\n    out.append(rng.random())")
    ns.update(rng=np.random.default_rng(0), buf=[])
    names = call_arguments(ast.parse("burn(rng, buf)"), ns)
    assert "rng" not in names
    assert "buf" in names


# -- In a notebook -----------------------------------------------------------------


@pytest.fixture
def stored(cash_magics, cash_instance):
    """``cash_magics`` storing every statement, however cheap, so the helper
    call is stored and a second run can restore it."""
    cash_instance.config.persist_all = True
    cash_magics.shell.user_ns.update(np=np)
    return cash_magics


def _statements(magics):
    return magics.cash_status("dict")["last_cell"]["statements"]


def _run_all(magics):
    statuses = {}
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)
        statuses[cell] = [str(m.get("status")) for m in _statements(magics)]
    return statuses


def test_the_helper_call_is_stored_and_restored_where_the_stream_stood(stored):
    plain_paths, plain_after, plain_state = _plain()
    first = _run_all(stored)
    ns = stored.shell.user_ns
    assert first[SIM] == ["COMPUTED"], first[SIM]
    reasons = [r for m in _statements(stored) for r in m.get("uncacheable_reasons", ())]
    assert not reasons
    np.testing.assert_array_equal(ns["paths"], plain_paths)
    np.testing.assert_array_equal(ns["after"], plain_after)
    assert stored.tracking_state.carrier_advances[statement_source_hash(SIM)] == {"rng"}

    second = _run_all(stored)
    assert second[SIM] == ["RESTORED"], f"the helper call was not served from the cache: {second[SIM]}"
    np.testing.assert_array_equal(ns["paths"], plain_paths)
    np.testing.assert_array_equal(ns["after"], plain_after)
    assert ns["rng"].bit_generator.state == plain_state


def test_a_moved_generator_is_not_served_the_fresh_stream(stored):
    """Run again without the seed: the generator stands past the first draw,
    so the call draws on from there, as without cash."""
    _run_all(stored)
    ns = stored.shell.user_ns
    rng = np.random.default_rng(7)
    rng.standard_normal((4, 3))
    rng.standard_normal(2)
    expected = rng.standard_normal((4, 3)).sum(axis=1)
    run_cash_cell(stored, SIM)  # no notebook around it: nothing above is rebuilt
    assert [str(m.get("status")) for m in _statements(stored)] == ["COMPUTED"]
    np.testing.assert_array_equal(ns["paths"], expected)


def test_a_reseed_is_a_new_stream_and_the_old_seed_comes_back(stored):
    _run_all(stored)
    cells = [HELPER, SEED.replace("7", "8"), SIM, AFTER]
    for cell in cells[1:]:
        run_cash_cell(stored, cell, cells=cells)
        if cell == SIM:
            assert [str(m.get("status")) for m in _statements(stored)] == ["COMPUTED"]
    ns = stored.shell.user_ns
    plain_paths, plain_after, _ = _plain(8)
    np.testing.assert_array_equal(ns["paths"], plain_paths)
    np.testing.assert_array_equal(ns["after"], plain_after)

    statuses = _run_all(stored)
    assert statuses[SIM] == ["RESTORED"]
    plain_paths, plain_after, _ = _plain(7)
    np.testing.assert_array_equal(ns["paths"], plain_paths)
    np.testing.assert_array_equal(ns["after"], plain_after)
