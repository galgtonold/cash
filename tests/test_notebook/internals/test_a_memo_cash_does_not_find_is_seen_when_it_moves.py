"""`helper_memos.hidden_moved`: a memo of the user's held where cash does
not look for memos (a closure, a dict, a list, an object's attribute, a
``functools.partial``) is named when a run may have called it.

Such a memo is met by a look through every object. Met for the first time
it is named, since its counters before the run are not known; known, it is
named only when its counters moved since `hidden_counters` read them. A
memo bound in the notebook is found where cash looks and never named.
"""

from __future__ import annotations

import functools

import pytest

from cash.notebook import helper_memos

HOLDERS = {
    "closure": "def make():\n    @functools.cache\n    def cities(n):\n        return n\n    return cities\nheld = make()\n",
    "dict": "H = {'c': functools.cache(_cities)}\nheld = H['c']\n",
    "list": "L = [functools.cache(_cities)]\nheld = L[0]\n",
    "namespace": "import types\nNS = types.SimpleNamespace(c=functools.cache(_cities))\nheld = NS.c\n",
    "instance": "class Box:\n    pass\nB = Box()\nB.c = functools.cache(_cities)\nheld = B.c\n",
    "partial": "P = functools.partial(functools.cache(_cities))\nheld = P.func\n",
}


def _ns(code: str) -> tuple[dict, object]:
    """The namespace *code* leaves, and the memo it holds, which the
    namespace itself does not bind."""
    ns = {"functools": functools, "__name__": "__main__"}
    exec(compile("def _cities(n):\n    return n\n" + code, __file__, "exec"), ns)
    return ns, ns.pop("held", None)


@pytest.mark.parametrize("code", list(HOLDERS.values()), ids=list(HOLDERS))
def test_a_hidden_memo_is_named_when_it_moves(code):
    ns, memo = _ns(code)
    assert helper_memos.hidden_moved({}, ns) is not None, "a memo met for the first time was not named"
    before = helper_memos.hidden_counters()
    assert helper_memos.hidden_moved(before, ns) is None, "a known memo that did not move was named"
    memo(3)
    assert helper_memos.hidden_moved(before, ns) is not None, "a call of the hidden memo went unseen"


def test_a_memo_bound_in_the_notebook_is_never_named():
    ns, _ = _ns("")
    ns["cities"] = functools.cache(ns["_cities"])
    before = helper_memos.hidden_counters()
    ns["cities"](2)
    assert helper_memos.hidden_moved(before, ns) is None
