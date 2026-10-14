"""A costly statement that runs again over a whole array, although only some
of its rows changed since its last run, says which rows and suggests a
cached helper per item or chunk (``CACHE-PER-ITEM-RERUN``).

Silent when every row changed, when nothing did (it ran for another
reason), when the statement is cheap, and after the first time.
"""

from __future__ import annotations

import warnings

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

CODE = "CACHE-PER-ITEM-RERUN"
#: Above the hint's one-second floor, with room for a slow machine's timer.
SLOW = 1.1

SETUP = "import numpy as np, time"
BASE = "imgs = np.arange(3000 * 16, dtype=np.int64).reshape(3000, 16)"
GROWN = "imgs = np.concatenate([np.arange(3000 * 16, dtype=np.int64).reshape(3000, 16), np.ones((400, 16), np.int64)])"
MIDDLE = "imgs = np.arange(3000 * 16, dtype=np.int64).reshape(3000, 16); imgs[1000:1100] = -1"
ALL_NEW = "imgs = np.arange(3000 * 16, dtype=np.int64).reshape(3000, 16) + 1"
STEP = f"clean = [imgs * 2, time.sleep({SLOW})][0]"
CHEAP = "clean = imgs * 2"


def _run(magics, *cells: str) -> list[warnings.WarningMessage]:
    """Run *cells* in turn; the per-item warnings they raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell)
    return [w for w in caught if getattr(w.message, "code", None) == CODE]


def _status(magics, code: str) -> str:
    rows = magics.cash_status("dict")["last_cell"]["statements"]
    return next(r["status"] for r in rows if code.startswith(r.get("code") or "\0"))


def test_a_costly_rerun_over_appended_rows_names_them(cash_magics):
    assert not _run(cash_magics, SETUP, BASE, STEP), "a first run has nothing to compare with"

    found = _run(cash_magics, GROWN, STEP)

    assert _status(cash_magics, STEP) == CacheStatus.COMPUTED
    assert len(found) == 1, found
    text = " ".join(str(found[0].message).split())
    assert f"`{STEP}` re-ran over all 3,400 rows of `imgs`" in text, text
    assert "only 400 rows differ from the last run (rows 3000-3399)" in text, text
    assert "Fix: do the work in a @cash.cache function that takes one chunk of rows" in text


def test_a_block_changed_in_the_middle_is_named(cash_magics):
    _run(cash_magics, SETUP, BASE, STEP)

    found = _run(cash_magics, MIDDLE, STEP)

    assert len(found) == 1, found
    text = " ".join(str(found[0].message).split())
    assert "re-ran over all 3,000 rows of `imgs`" in text
    # The 100 rows sit inside 47-row blocks: the blocks around them.
    assert "rows 987-1127" in text, text


def test_every_row_changed_says_nothing(cash_magics):
    _run(cash_magics, SETUP, BASE, STEP)

    found = _run(cash_magics, ALL_NEW, STEP)

    assert _status(cash_magics, STEP) == CacheStatus.COMPUTED, "control: it ran again"
    assert not found


def test_a_cheap_step_says_nothing(cash_magics):
    _run(cash_magics, SETUP, BASE, CHEAP)

    found = _run(cash_magics, GROWN, CHEAP)

    assert _status(cash_magics, CHEAP) == CacheStatus.COMPUTED, "control: it ran again"
    assert not found


def test_it_is_named_once_per_statement(cash_magics):
    _run(cash_magics, SETUP, BASE, STEP)

    found = _run(cash_magics, GROWN, STEP, MIDDLE, STEP)

    assert _status(cash_magics, STEP) == CacheStatus.COMPUTED, "control: it ran again"
    assert len(found) == 1, found


def test_an_edited_statement_does_not_compare_with_the_old_one(cash_magics):
    _run(cash_magics, SETUP, BASE, STEP)

    found = _run(cash_magics, GROWN, STEP.replace("imgs * 2", "imgs * 3"))

    assert not found


FRAME = (
    "import pandas as pd\nplayed = pd.DataFrame({'season': np.repeat(np.arange(50), 60), 'goals': np.arange(3000) % 7})"
)
FRAME_GROWN = (
    "played = pd.concat([pd.DataFrame({'season': np.repeat(np.arange(50), 60), 'goals': np.arange(3000) % 7}),"
    " pd.DataFrame({'season': [50] * 60, 'goals': np.arange(60) % 5})], ignore_index=True)"
)
ELO = f"hist = [played.assign(p=played['goals'].cumsum()), time.sleep({SLOW})][0]"


def test_a_frame_grown_by_a_season_names_the_new_rows(cash_magics):
    _run(cash_magics, SETUP, FRAME, ELO)

    found = _run(cash_magics, FRAME_GROWN, ELO)

    assert len(found) == 1, found
    text = " ".join(str(found[0].message).split())
    assert "re-ran over all 3,060 rows of `played`" in text, text
    assert "only 60 rows differ from the last run (rows 3000-3059)" in text, text


def _session(magics) -> list:
    """What the session serves: each step's status and value."""
    seen = []
    for cell in (SETUP, BASE, STEP, GROWN, STEP, STEP, MIDDLE, STEP):
        run_cash_cell(magics, cell)
        if cell == STEP:
            seen.append((_status(magics, STEP), magics.shell.user_ns["clean"].tolist()))
    return seen


def test_the_hint_never_changes_what_is_served(cash_magics, cash_instance, monkeypatch):
    """The same session with the hint switched off serves the same statuses
    and values."""
    import cash.notebook.statement.processor as processor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_hint = _session(cash_magics)
    assert any(getattr(w.message, "code", None) == CODE for w in caught), "control: the hint fired"

    cash_instance.backend.clear()  # a fresh cache and session for the second pass
    cash_magics.tracking_state.reset_session_state()
    monkeypatch.setattr(processor, "PER_ITEM_MIN_COST_S", float("inf"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        without = _session(cash_magics)
    assert not any(getattr(w.message, "code", None) == CODE for w in caught), "control: the hint is off"

    assert with_hint == without
    assert [s for s, _ in with_hint] == [
        CacheStatus.COMPUTED,
        CacheStatus.COMPUTED,
        CacheStatus.RESTORED,
        CacheStatus.COMPUTED,
    ]
