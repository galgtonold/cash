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

import pytest

from cash.notebook.call_key import CallKeys
from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell
from tests._work_counts import deepcopy_steps, hashed_bytes, method_calls, pickled_bytes, ram_copied_bytes
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

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


def test_a_comprehension_keys_a_bounded_number_of_calls(cash_magics, cash_instance):
    # No call is stored, however slow the machine: a call stalled past the
    # 3 ms floor in the first comprehension was stored and served in the
    # second, and a site with hits is judged only after 50 keyed calls.
    cash_instance.config.call_cost_floor_seconds = 10.0
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


def test_a_cell_adding_columns_to_a_table_copies_each_column_once(cash_magics):
    """``df["hour"] = ...`` statement after statement: each stores the table
    again, and the RAM tier copied all of it each time (440 MB a statement
    for a 2.9-million-row table). It copies the new column now, and takes
    the others from the entry before."""
    from cash.backends import frame_sharing

    pytest.importorskip("pandas")
    if not frame_sharing.enabled():
        pytest.skip("needs pandas copy-on-write")
    n = 200_000
    run_cash_cell(
        cash_magics,
        "import time\nimport numpy as np\nimport pandas as pd\n"
        "def slow(s):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return s * 2",
    )
    cell = (
        f"df = pd.DataFrame({{'a': np.arange({n}, dtype=float), 'b': np.ones({n}), 'c': np.zeros({n})}})\n"
        "df['d'] = slow(df['a'])\n"
        "df['e'] = slow(df['b'])\n"
        "df['f'] = slow(df['c'])"
    )
    with ram_copied_bytes() as copied:
        run_cash_cell(cash_magics, cell)
    df = cash_magics.shell.user_ns["df"]
    assert list(df.columns) == ["a", "b", "c", "d", "e", "f"]
    column = n * 8
    # Three stored versions of a growing table: each new column copied once
    # (3 columns) beside the first version's 3 -- ~6 columns today; every
    # version copied whole was 4 + 5 + 6 = 15.
    assert column * 3 <= copied.bytes <= column * 8, copied.bytes / column


_CLEANING_CELL = (
    "df = (time.sleep({s}), pd.DataFrame({{'a': np.arange(200_000.0), 'z': np.zeros(200_000)}}))[1]\n"
    "df = (time.sleep({s}), df[df['a'] > 5])[1]\n"
    "df['b'] = (time.sleep({s}), df['a'] * 2)[1]"
).format(s=ABOVE_PERSISTENCE_FLOOR_S)


def _cleaning_cell_work(cash_magics, monkeypatch, bytes_per_second: float):
    from cash.backends import frame_sharing, memory_backend

    pytest.importorskip("pandas")
    if not frame_sharing.enabled():
        pytest.skip("needs pandas copy-on-write")
    run_cash_cell(cash_magics, "import time\nimport numpy as np\nimport pandas as pd")
    # The speed table stores were measured at: what a copy is priced by.
    monkeypatch.setattr(memory_backend, "_STORE_COPY_SPEED", [bytes_per_second])
    with ram_copied_bytes() as copied:
        run_cash_cell(cash_magics, _CLEANING_CELL)
    rows = cash_magics.cash_status("dict")["last_cell"]["statements"]
    return copied, [row.get("skipped_reason") for row in rows]


def test_an_intermediate_dearer_to_copy_than_to_compute_is_not_copied(cash_magics, monkeypatch):
    """``df = df[...]`` again and again in a cleaning cell: each version was
    copied into the RAM tier, though only a re-run of part of the cell could
    restore it, and on a 2.9-million-row table each copy cost more than the
    filter. A version every name of which the cell rebinds later, and whose
    copy costs more than computing it did, is kept out of RAM now; the
    cell's last version is stored as before."""
    from cash.notebook.statement.store import KEPT_OUT_OF_RAM

    copied, reasons = _cleaning_cell_work(cash_magics, monkeypatch, 1e6)  # 1 MB/s: dear
    assert [bool(r and r.startswith(KEPT_OUT_OF_RAM)) for r in reasons] == [True, True, False], reasons
    df = cash_magics.shell.user_ns["df"]
    # The last version alone is copied: 3 columns of ~200,000 rows (8 bytes),
    # where every version copied was 2 + 2 + 3.
    assert copied.bytes <= 3 * 200_000 * 8 + 4096, copied.bytes
    assert list(df.columns) == ["a", "z", "b"] and len(df) == 200_000 - 6

    # The cell run again is served by its last version.
    run_cash_cell(cash_magics, "df = None")
    run_cash_cell(cash_magics, _CLEANING_CELL)
    rows = cash_magics.cash_status("dict")["last_cell"]["statements"]
    assert str(rows[-1]["status"]).endswith("RESTORED"), rows
    assert cash_magics.shell.user_ns["df"]["b"].iloc[0] == 12.0


def test_an_intermediate_cheap_to_copy_is_still_kept(cash_magics, monkeypatch):
    """Positive control: priced at a fast copy, every version is stored."""
    copied, reasons = _cleaning_cell_work(cash_magics, monkeypatch, 1e12)
    assert reasons == [None, None, None], reasons
    assert copied.bytes >= (2 + 2) * 200_000 * 8
