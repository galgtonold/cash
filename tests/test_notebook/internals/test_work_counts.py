"""Work counts of the notebook path: the costs that made cells slow, counted.

Each test counts the work behind one cost that once grew -- statements run
and bytes hashed for a trivial cell in a long notebook, a module's table
re-hashed before every cell, a comprehension keying every call before the
many-cheap-calls guard, a re-run copying a list of records an item at a
time -- and asserts it under a generous bound, so growth fails here, without
a timer. The bounds sit well above today's counts (in each comment) and well
below what the slow version did. Counters: ``tests/_work_counts.py``.
"""

from __future__ import annotations

import sys

from cash.notebook.call_key import CallKeys
from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell
from tests._work_counts import deepcopy_steps, hashed_bytes, method_calls, pickled_bytes

KB = 1024


def _last_cell_work(magics, n: int, tag: str):
    cells = ["base = 1"] + [f"{tag}{i} = base + {i}" for i in range(n)]
    for cell in cells[:-1]:
        run_cash_cell(magics, cell, cells=cells)
    with hashed_bytes() as hashed, method_calls(StatementProcessor, "process_statement") as statements:
        run_cash_cell(magics, cells[-1], cells=cells)
    return hashed, statements


def test_a_trivial_cell_does_the_same_small_work_in_a_long_notebook(cash_magics):
    short_hashed, short_statements = _last_cell_work(cash_magics, 10, "short")
    long_hashed, long_statements = _last_cell_work(cash_magics, 60, "long")

    # Its own statement only: nothing above it is re-run.
    assert short_statements.calls == long_statements.calls == 1
    # ~2 KB hashed today: the cell's key and its inputs' lineage.
    assert 0 < short_hashed.bytes <= 16 * KB
    assert long_hashed.bytes <= short_hashed.bytes + 4 * KB, (short_hashed.bytes, long_hashed.bytes)


def test_a_module_table_is_not_hashed_again_before_an_unrelated_cell(cash_magics, tmp_path, monkeypatch):
    (tmp_path / "wc_helpers.py").write_text(
        "import numpy as np\nEMB = np.random.default_rng(0).random((125_000, 8))\n\n\ndef score(i):\n    return float(EMB[i].sum())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "wc_helpers", raising=False)
    table_bytes = 125_000 * 8 * 8
    cells = ["import wc_helpers", "v = wc_helpers.score(3)", "x1 = 1", "x2 = 2"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with hashed_bytes() as reading:
        run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    with hashed_bytes() as unrelated:
        run_cash_cell(cash_magics, cells[3], cells=cells)

    # The statement reading the table keys on its content ...
    assert reading.bytes >= table_bytes
    # ... and a cell that cannot reach it hashes ~1 KB (an 8 MB re-hash per cell before).
    assert unrelated.bytes <= 64 * KB


def test_a_comprehension_keys_a_bounded_number_of_calls(cash_magics):
    keyed = {}
    hashed = {}
    run_cash_cell(cash_magics, "def f(i):\n    return i * 2")
    for n in (2_000, 8_000):
        with method_calls(CallKeys, "key") as keys, hashed_bytes() as h:
            run_cash_cell(cash_magics, f"d{n} = [f(i) for i in range({n})]")
        keyed[n], hashed[n] = keys.calls, h.bytes
        assert cash_magics.shell.user_ns[f"d{n}"][-1] == 2 * (n - 1)

    # The guard keys 5 calls, sees caching costs far more than they compute,
    # times a few plain, then runs the site plain: no key per element (50
    # before it decided on the first calls' evidence).
    assert 0 < keyed[2_000] == keyed[8_000] <= 10, keyed
    # ~36 KB hashed today, the same for 4x the elements.
    assert hashed[8_000] <= 96 * KB and hashed[8_000] <= hashed[2_000] + 8 * KB, hashed


def test_a_rerun_restores_built_records_without_a_step_per_item(cash_magics):
    cells = [
        "N = 100_000",
        "recs = [{'id': i, 'user': 'u', 'amount': i * 0.5} for i in range(N)]",
        "index = {f'k{i}': i for i in range(N)}",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells[1:]:
        with deepcopy_steps() as copies, pickled_bytes() as pickled:
            run_cash_cell(cash_magics, cell, cells=cells)
        (statement,) = cash_magics.cash_status("dict")["last_cell"]["statements"]
        assert str(statement["status"]).endswith("RESTORED"), statement["status"]
        # Copied a container level at a time in C: no deepcopy step at all
        # today, one per item before; and nothing pickled but the metadata.
        assert copies.calls <= 10, (cell, copies.calls)
        assert pickled.bytes <= 4 * KB, (cell, pickled.bytes)


def test_a_fast_cell_draws_its_badge_twice(cash_magics):
    """RUNNING, then the final badge: a progress render between them, for a
    statement that finished at once, was ~2 ms of a trivial cell."""
    from cash.notebook.ipython.badges import BadgePresenter

    cash_magics._cell_executor._badges.mode = "html"
    run_cash_cell(cash_magics, "base = 1")
    with method_calls(BadgePresenter, "render") as renders:
        run_cash_cell(cash_magics, "a = base + 1\nb = a + 1\nc = b + 1")

    # Three before: RUNNING, the first statement's progress, DONE.
    assert renders.calls == 2, renders.calls
