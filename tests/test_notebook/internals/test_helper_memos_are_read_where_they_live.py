"""`helper_memos`: the memos a statement called, found where they live, and
an entry over them served only while they hold every key and answer the
statement read.

Each case below fills a memo, records a statement that only reads it, puts
the memo in a state where it does not hold what the statement read, and asks
whether the entry may be served: it must not be.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from cash.notebook import helper_memos


def _record(ns, code):
    before = helper_memos.watched(ns)
    exec(code, ns)
    return helper_memos.used(before)


DECORATED = (
    "def deco(fn):\n    @functools.wraps(fn)\n    def w(*a):\n        return fn(*a)\n    return w\n"
    "@deco\n@functools.cache\ndef f(n):\n    return n"
)


@pytest.mark.parametrize(
    ("setup", "fill", "read", "then"),
    [
        pytest.param(
            "@functools.cache\ndef f(n):\n    return n",
            "f(1); f(2)",
            "x = f(1) + f(2)",
            "f.cache_clear(); f(3); f(4)",
            id="cleared-and-refilled-with-as-many-other-keys",
        ),
        pytest.param(
            "@functools.lru_cache(maxsize=8)\ndef f(n):\n    return {'v': n}",
            "f(1)",
            "x = f(1)",
            "f.cache_clear(); f(2); f(3)",
            id="bounded-memo-returning-dicts",
        ),
        pytest.param(
            "@functools.lru_cache(typed=True)\ndef f(n):\n    return n",
            "f(1)",
            "x = f(1)",
            "f.cache_clear(); f(1.0); f(2.0)",
            id="typed-keys",
        ),
        pytest.param(
            "@functools.lru_cache(maxsize=2)\ndef f(n):\n    return n",
            "f(1)",
            "x = f(1)",
            "f(2); f(3)",
            id="the-key-evicted",
        ),
        pytest.param(DECORATED, "f(1)", "x = f(1)", "f.__wrapped__.cache_clear()", id="under-another-decorator"),
        pytest.param(
            "class C:\n    @staticmethod\n    @functools.cache\n    def f(n):\n        return n",
            "C.f(1)",
            "x = C.f(1)",
            "C.f.cache_clear()",
            id="a-staticmethod",
        ),
        pytest.param(
            "G = 1\n@functools.cache\ndef f(n):\n    return n + G",
            "f(1)",
            "x = f(1)",
            "f.cache_clear(); G = 5; f(1)",
            id="the-same-key-with-another-answer",
        ),
    ],
)
def test_an_entry_is_not_served_over_a_memo_without_what_it_read(setup, fill, read, then):
    ns: dict = {}
    exec("import functools\n" + setup, ns)
    exec(fill, ns)
    records = _record(ns, read)
    assert isinstance(records, list) and records, records
    assert helper_memos.unfilled(records, ns) is None, "control: as the statement left it"
    exec(then, ns)
    assert helper_memos.unfilled(records, ns), records


def test_a_memo_holding_more_answers_than_are_read_is_never_served_over():
    ns: dict = {}
    exec("import functools\n@functools.cache\ndef f(n):\n    return n", ns)
    exec("[f(i) for i in range(300)]", ns)
    records = _record(ns, "x = [f(i) for i in range(300)]")
    assert records == [{"where": ["ns", "f"], "unread": True}], records
    assert helper_memos.unfilled(records, ns), "not read: never served"


def test_a_run_that_makes_a_memo_compute_is_said_to_fill_it():
    ns: dict = {}
    exec("import functools\n@functools.cache\ndef f(n):\n    return n", ns)
    assert _record(ns, "x = f(1)") == (helper_memos.FILLED, "f")
    assert isinstance(_record(ns, "x = f(1)"), list), "the second run only reads it"


def test_a_memo_in_a_module_of_the_users_is_followed(tmp_path, monkeypatch):
    (tmp_path / "memo_mod_hm.py").write_text(
        "import functools\n\n@functools.cache\ndef table(n):\n    return list(range(n))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("memo_mod_hm")
    try:
        ns: dict = {"mod": mod}
        assert _record(ns, "t = mod.table(3)") == (helper_memos.FILLED, "memo_mod_hm.table")
        records = _record(ns, "t = mod.table(3)")
        assert records and records[0]["where"] == ["module", "memo_mod_hm", "table"], records
        assert helper_memos.unfilled(records, ns) is None
        mod.table.cache_clear()
        assert helper_memos.unfilled(records, ns), "emptied: a hit would leave it so"
    finally:
        sys.modules.pop("memo_mod_hm", None)


def test_keyword_arguments_key_the_same_after_a_refill():
    ns: dict = {}
    exec("import functools\n@functools.cache\ndef f(a, b=1):\n    return a + b", ns)
    ns["f"](1, b=2)
    records = _record(ns, "x = f(1, b=2)")
    assert records and records[0]["items"], records
    ns["f"].cache_clear()
    ns["f"](1, b=2)
    assert helper_memos.unfilled(records, ns) is None


def test_a_key_with_no_stable_digest_does_not_break_the_statement():
    class Odd:
        def __reduce__(self):
            raise RuntimeError("no pickling")

    ns: dict = {"arg": Odd()}
    exec("import functools\n@functools.cache\ndef g(a):\n    return 1", ns)
    exec("g(arg)", ns)
    records = _record(ns, "y = g(arg)")
    assert records, records
    helper_memos.unfilled(records, ns)


def test_a_library_memo_is_not_watched():
    import functools

    ns = {"split": functools.lru_cache()(str.split)}
    assert helper_memos.watched(ns) is None
