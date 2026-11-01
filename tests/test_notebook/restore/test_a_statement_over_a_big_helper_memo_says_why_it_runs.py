"""A statement reading a ``functools.cache`` memo that holds more answers
than cash checks runs every time, as before, and now says why once
(``NOTEBOOK-MEMO-UNREAD``): only the badge did, and the cell was just slow.
A memo of a few hundred answers is checked and served."""

from __future__ import annotations

import warnings

from cash.notebook import helper_memos
from cash.notebook.cache_status import CacheStatus
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

DEFINE = "import functools, time\n@functools.cache\ndef h(n):\n    return n * n"


def _run(sp, code):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        meta = sp.process_statement(code)
    return meta, [str(w.message) for w in caught if "NOTEBOOK-MEMO-UNREAD" in str(w.message)]


def test_a_memo_over_the_limit_warns_once_and_runs_every_time(statement_processor, monkeypatch):
    monkeypatch.setattr(helper_memos, "MAX_KEYED", 50)
    sp = statement_processor
    sp.process_statement(DEFINE)
    sp.process_statement("_fill = [h(i) for i in range(60)]")
    read = f"tot = (sum(h(i) for i in range(60)), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    first, told = _run(sp, read)
    assert len(told) == 1
    assert "the helper memo h, which holds 60 answers" in told[0]
    assert "up to 50 answers" in told[0]
    again, told_again = _run(sp, read)
    assert again.get("status") is not CacheStatus.RESTORED
    assert told_again == []  # once per statement
    assert sp.shell.user_ns["tot"] == sum(i * i for i in range(60))


def test_a_memo_of_three_hundred_answers_is_checked_and_served(statement_processor):
    assert helper_memos.MAX_KEYED >= 300
    sp = statement_processor
    sp.process_statement(DEFINE)
    sp.process_statement("_fill = [h(i) for i in range(300)]")
    read = f"tot = (sum(h(i) for i in range(300)), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    _, told = _run(sp, read)
    assert told == []
    meta, _ = _run(sp, read)
    assert meta.get("status") is CacheStatus.RESTORED


def test_defining_a_memoised_helper_is_checked_without_a_warning(statement_processor):
    # functools' keyword mark, a bare object() in every memo of the process,
    # sent the share check up thousands of holders: each @functools.cache
    # said NOTEBOOK-SHARE-UNCHECKED.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statement_processor.process_statement(DEFINE)
    assert not [w for w in caught if "NOTEBOOK-SHARE-UNCHECKED" in str(w.message)]


def test_a_function_made_by_exec_is_named_in_the_warning(statement_processor):
    sp = statement_processor
    sp.process_statement("ns = {}\nexec('def bump(x): return x + 1', ns)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.process_statement('out = list(map(ns["bump"], range(5)))')
    told = [str(w.message) for w in caught if "NOTEBOOK-CALLABLE-UNREAD" in str(w.message)]
    assert len(told) == 1
    assert "the function 'bump' made by exec, held in 'ns'," in told[0]
    assert sp.shell.user_ns["out"] == [1, 2, 3, 4, 5]
