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
def test_an_entry_is_not_served_over_a_memo_without_what_it_read(setup, fill, read, then, monkeypatch):
    # Helpers that do something besides return (`_pure` is narrow; these
    # one-liners would pass it, and a pure memo may be served emptied).
    monkeypatch.setattr(helper_memos, "_pure", lambda memo: False)
    ns: dict = {}
    exec("import functools\n" + setup, ns)
    _record(ns, fill)  # the run that fills it, so its answers are seen
    records = _record(ns, read)
    assert isinstance(records, list) and records, records
    assert helper_memos.unfilled(records, ns) is None, "control: as the statement left it"
    exec(then, ns)
    assert helper_memos.unfilled(records, ns), records


def test_a_memo_holding_more_answers_than_are_read_is_never_served_over():
    ns: dict = {}
    exec("import functools\n@functools.cache\ndef f(n):\n    return n", ns)
    exec("[f(i) for i in range(300)]", ns)
    said = _record(ns, "x = [f(i) for i in range(300)]")
    assert said[0] == helper_memos.FILLED and "more answers than cash checks" in said[1], said


def test_a_run_that_makes_a_memo_compute_is_said_to_fill_it(monkeypatch):
    monkeypatch.setattr(helper_memos, "_pure", lambda memo: False)
    ns: dict = {}
    exec("import functools\n@functools.cache\ndef f(n):\n    return n", ns)
    said = _record(ns, "x = f(1)")
    assert said[0] == helper_memos.FILLED and "filled the helper memo f:" in said[1], said
    assert isinstance(_record(ns, "x = f(1)"), list), "the second run only reads it"


def test_a_memo_in_a_module_of_the_users_is_followed(tmp_path, monkeypatch):
    (tmp_path / "memo_mod_hm.py").write_text(
        "import functools\n\n@functools.cache\ndef table(n):\n    print(n)\n    return list(range(n))\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("memo_mod_hm")
    try:
        ns: dict = {"mod": mod}
        said = _record(ns, "t = mod.table(3)")
        assert said[0] == helper_memos.FILLED and "memo_mod_hm.table" in said[1], said
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


def test_the_memos_are_looked_for_again_only_when_something_could_bring_one(monkeypatch, tmp_path):
    # Every cached call in a statement asks for the memos it reaches; looking
    # through every module of the user's at each one made a 50,000-call loop
    # take over 30 s where the suite imports hundreds of them.
    looks = {"ns": 0, "modules": 0}
    real_ns, real_modules = helper_memos._ns_memos, helper_memos._module_memos

    def count(kind, real):
        def counted(*args):
            looks[kind] += 1
            return real(*args)

        return counted

    monkeypatch.setattr(helper_memos, "_ns_memos", count("ns", real_ns))
    monkeypatch.setattr(helper_memos, "_module_memos", count("modules", real_modules))
    ns: dict = {"functools": __import__("functools")}
    helper_memos.new_statement()
    for i in range(100):
        ns["i"] = i  # a loop variable: nothing that can hold a memo
        assert helper_memos.watched(ns) is None
    assert looks == {"ns": 1, "modules": 1}

    # A memo made inside the statement (a loop body, a def in the cell) is
    # seen by the cached calls after it.
    exec("@functools.cache\ndef h(n):\n    return n", ns)
    assert [w[0] for w in helper_memos.watched(ns)] == [["ns", "h"]]
    exec("@functools.cache\ndef h(n):\n    return -n", ns)  # rebound
    assert helper_memos.watched(ns)[0][1] is ns["h"]
    assert looks["ns"] == 3

    # So is one in a module imported inside the statement.
    (tmp_path / "memo_mid_hm.py").write_text(
        "import functools\n\n@functools.cache\ndef t(n):\n    return n\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        importlib.import_module("memo_mid_hm")
        assert ["module", "memo_mid_hm", "t"] in [w[0] for w in helper_memos.watched(ns)]
    finally:
        sys.modules.pop("memo_mid_hm", None)


def test_a_memo_made_and_filled_in_the_same_statement_is_said_to_fill_it():
    ns: dict = {"functools": __import__("functools")}
    helper_memos.new_statement()
    before = helper_memos.watched(ns)
    assert before is None
    exec("h = functools.cache(lambda n: [n])\nx = h(1)", ns)
    said = helper_memos.used(before, ns=ns)
    assert said and said[0] == helper_memos.FILLED and "h" in said[1], said


@pytest.mark.parametrize(
    ("body", "pure", "params", "later"),
    [
        ("return n * n", True, "n", ""),
        ("return sum(range(n)) + len(str(n))", True, "n", ""),
        ("return 1 if n < 2 else n * h(n - 1)", True, "n", ""),
        ("return g(n) * 2", False, "n", ""),  # a later cell can redefine g
        ("return n * k", True, "n, k=2", ""),
        ("out(n)\n    return n", False, "n, out=print", ""),
        ("acc.append(n)\n    return len(acc)", False, "n, acc=[]", ""),
        ("return n", False, "n, *, out=print", ""),
        ("return abs(n)", False, "n", "def abs(v):\n    return 99\n"),
        ("print(n)\n    return n", False, "n", ""),
        ("return SCALE * n", False, "n", ""),
        ("return g_prints(n)", False, "n", ""),
        ("return os.environ.get('X', n)", False, "n", ""),
        ("return random.random()", False, "n", ""),
        ("return next(COUNTER)", False, "n", ""),
        ("return open(n).read()", False, "n", ""),
        ("return n.upper()", False, "n", ""),
    ],
)
def test_a_helper_is_pure_only_when_its_answer_comes_from_its_arguments_alone(body, pure, params, later):
    ns: dict = {}
    exec(
        "import functools, itertools, os, random\nSCALE = 2\nCOUNTER = itertools.count()\n"
        "def g(n):\n    return n + 1\ndef g_prints(n):\n    print(n)\n    return n\n"
        f"@functools.cache\ndef h({params}):\n    {body}\n{later}",
        ns,
    )
    assert helper_memos._pure(ns["h"]) is pure
