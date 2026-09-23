"""A current-cell cache hit stands in for the upstream run of what it restores.

A variable the cell reads is broken (gone after a restart, or stale after an
upstream edit), but the cell's own ``df['b'] = f(df)`` is a cache hit whose
restore puts ``df`` back: nothing upstream needs to run for it. The upstream
check decides that by probing the cache before the cell runs
(``eliminate_broken_vars_via_current_cell_probe``). The probe must build the
statement's key as the runtime does, must count only an entry the runtime can
restore, and the name it holds in the namespace for the restore must not
outlive the cell when no restore fills it.
"""

from __future__ import annotations

import pytest

from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.upstream import NotebookSimulator
from tests._cell_driver import run_cash_cell
from tests.conftest import MockShell


def _on(magics):
    magics.cash_on("")
    # Every statement is written to disk, however cheap.
    magics.cash_persist("on")
    magics.badges.mode = "off"
    return magics


@pytest.fixture
def kernel(tmp_path):
    """A fresh kernel over a cache directory that outlives it: its magics and
    its Cash."""

    def _start() -> tuple[CashMagics, Cash]:
        # Each kernel has its own shell; the cache directory is what they share.
        cash = Cash(cache_dir=str(tmp_path), register_magic=False)
        return _on(CashMagics(MockShell(), cash)), cash

    return _start


def _run(magics, cell, cells, **kwargs) -> BaseException | None:
    try:
        run_cash_cell(magics, cell, cells=cells, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - what the cell raised is the observation
        return exc
    return None


def test_a_statement_that_draws_is_probed_under_its_real_key(cash_magics, cash_instance, mock_shell):
    """A draw reads the RNG as a hidden input, which the runtime folds into the
    key. A probe without it never found the entry, and restored ``df``'s
    producer from upstream instead."""
    _on(cash_magics)
    draw = "df['b'] = df['a'] * random.random()"
    before = ["import random\nrandom.seed(1)", "df = {'a': 1}", draw]
    edited = ["import random\nrandom.seed(1)", "df = {'a': 2}", draw]
    for cell in before:
        run_cash_cell(cash_magics, cell, cells=before)
    for cell in edited[1:]:
        run_cash_cell(cash_magics, cell, cells=edited)

    # The edit is undone: ``df`` in memory is stale, and the draw's entry from
    # the first run restores it.
    simulator = NotebookSimulator(mock_shell, cash_instance, cash_magics.tracking_state)
    plan = simulator.simulate_upstream(2, before, {"df", "random"})

    assert plan.statements == []
    assert plan.restored_info == [], "df was rebuilt upstream although the cell's own hit restores it"


def test_a_record_without_a_value_is_not_a_hit(kernel):
    """A statement whose value was too large to write leaves a metadata-only
    record on disk. After a restart that record restores nothing, so the
    upstream cell must run: the cell computes as Python would."""
    cells = ["df = {'a': 1}", "df['b'] = df['a'] * 2\ntotal = sum(df.values())"]
    first, cash = kernel()
    for cell in cells:
        run_cash_cell(first, cell, cells=cells)
    key = first.tracking_state.variable_sources["df"]
    metadata = cash.backend.get_metadata(key)
    cash.backend.delete(key)
    cash.backend.set_metadata_only(key, metadata)

    second, _ = kernel()
    assert _run(second, cells[1], cells) is None
    assert second.shell.user_ns["df"] == {"a": 1, "b": 2}
    assert second.shell.user_ns["total"] == 3


def test_a_held_name_no_restore_filled_does_not_outlive_the_cell(kernel):
    """The probe found the entry, but this run's ``ttl=0`` expires it: the
    runtime misses and no restore replaces the name the probe held. The next
    cell must not find it bound to that stand-in."""
    cells = ["df = {'a': 1}", "df['b'] = df['a'] * 2", "z = 1"]
    first, _ = kernel()
    for cell in cells:
        run_cash_cell(first, cell, cells=cells)

    second, _ = kernel()
    _run(second, cells[1], cells, ttl=0)
    assert _run(second, cells[2], cells) is None

    df = second.shell.user_ns.get("df")
    assert df is None or isinstance(df, dict), f"df is still the probe's stand-in: {df!r}"
