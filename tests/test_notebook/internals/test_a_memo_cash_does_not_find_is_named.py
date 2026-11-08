"""`helper_memos.unlocated` and `helper_memos.refillable`.

A memo of the user's held in a closure is not where cash looks for memos:
what a run leaves in it is in no record, so it is named, and a plot run
calling it is not kept. One bound in the notebook is found. And whether an
entry over memos it does not hold may be served once a memo is filled again
is told without calling anything.
"""

from __future__ import annotations

import functools

from cash.notebook import helper_memos

CLOSURE = (
    "def make():\n"
    "    @functools.cache\n"
    "    def cities(n):\n"
    "        return tuple(range(n))\n"
    "    def get(n):\n"
    "        return cities(n)\n"
    "    return get\n"
    "get_cities = make()\n"
)


def _ns(code: str) -> dict:
    ns = {"functools": functools, "__name__": "__main__"}
    exec(compile(code, __file__, "exec"), ns)
    return ns


def test_a_memo_held_in_a_closure_is_named():
    ns = _ns(CLOSURE)
    assert helper_memos.unlocated([ns["get_cities"]], ns) == "cities"


def test_a_memo_bound_in_the_notebook_is_found():
    ns = _ns("@functools.cache\ndef cities(n):\n    return tuple(range(n))\ndef get(n):\n    return cities(n)\n")
    assert helper_memos.unlocated([ns["get"]], ns) is None


def test_a_function_reaching_no_memo_names_none():
    ns = _ns("def get(n):\n    return n\n")
    assert helper_memos.unlocated([ns["get"]], ns) is None


PURE = "@functools.cache\ndef base(n):\n    return n * 2\n"


def _read_record(ns: dict, code: str) -> list[dict]:
    before = helper_memos.watched(ns)
    exec(code, ns)
    records = helper_memos.used(before)
    assert isinstance(records, list) and records, records
    return records


def test_a_pure_memo_lacking_what_was_read_is_refillable_and_left_unfilled():
    ns = _ns(PURE)
    ns["base"](1)
    records = _read_record(ns, "x = base(1)")
    ns["base"].cache_clear()
    assert helper_memos.unfilled(records, ns, refill=False) is not None
    assert helper_memos.refillable(records, ns)
    assert ns["base"].cache_info().currsize == 0, "telling whether it may refill filled the memo"


def test_a_memo_that_is_not_pure_is_not_refillable():
    ns = _ns("import random\n@functools.cache\ndef pts(n):\n    random.seed(n)\n    return random.random()\n")
    ns["pts"](1)
    records = _read_record(ns, "x = pts(1)")
    ns["pts"].cache_clear()
    assert helper_memos.unfilled(records, ns, refill=False) is not None
    assert not helper_memos.refillable(records, ns)
