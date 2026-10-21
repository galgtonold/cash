"""`helper_memos`: the memos a statement called, found in the namespace and
in the user's modules, and judged filled by the answers they hold."""

from __future__ import annotations

import functools
import importlib
import sys

from cash.notebook import helper_memos


def _call(ns, code):
    before = helper_memos.watched(ns)
    exec(code, ns)
    return helper_memos.used(before)


def test_a_memo_in_a_module_of_the_users_is_followed(tmp_path, monkeypatch):
    (tmp_path / "memo_mod_hm.py").write_text(
        "import functools\n\n@functools.cache\ndef table(n):\n    return list(range(n))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("memo_mod_hm")
    try:
        ns: dict = {"mod": mod}
        records = _call(ns, "t = mod.table(3)")
        assert records and records[0]["where"] == ["module", "memo_mod_hm", "table"], records
        assert helper_memos.unfilled(records, ns) is None
        mod.table.cache_clear()
        assert helper_memos.unfilled(records, ns), "emptied: a hit would leave it so"
    finally:
        sys.modules.pop("memo_mod_hm", None)


def test_keyword_arguments_key_the_same_in_every_process():
    ns: dict = {}
    exec("import functools\n@functools.cache\ndef f(a, b=1):\n    return a + b", ns)
    records = _call(ns, "x = f(1, b=2)")
    assert records and records[0]["keys"], records
    ns["f"].cache_clear()
    ns["f"](1, b=2)
    assert helper_memos.unfilled(records, ns) is None


def test_a_library_memo_is_not_watched():
    ns = {"split": functools.lru_cache()(str.split)}
    assert helper_memos.watched(ns) is None or all(w[0] != ["ns", "split"] for w in helper_memos.watched(ns))
