"""An operator on an object of the user's that a statement reaches through
something not the user's runs the object's special method: what that method
writes is the statement's, and a served statement must not drop it.

``d['k'] += 1`` and ``w = lst[i] + 1`` (held in a builtin container),
``C.x += 1`` (held by a class), ``ns.x + 1`` (held by a library's
namespace), and ``w = bump(x)`` where ``bump`` reads ``v + 1`` by name: each
was served after a restart with ``calls == []``, where a run appends. So
were an item past the first 1,000 of a list read by a variable index or by
``lst[-1]``, a function reading one through a dict, a list, a class, a
namespace or an object (``return d['k'] + 1``), a lambda reading one, and a
method reading one that a function calls (``return kk.m(a)``).
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

SLEEP = f"(time.sleep({F}), 0)[1]"
SETUP = """import time, types
calls = []
class Vec:
    def __add__(self, x):
        calls.append(1)
        return Vec()
class C:
    x = Vec()
    def m(self):
        return 0
v = Vec()
d = {'k': Vec()}
lst = [Vec()]
i = 0
ns = types.SimpleNamespace(x=Vec())
def bump(a):
    return v + a
big = [0] * 1500 + [Vec()]
j = 1500
class Holder:
    def m(self, a):
        return v + a
h = Holder()
h.x = Vec()
def via_dict(a):
    return d['k'] + a
def via_list(a):
    return lst[a] + 1
def via_class(a):
    return C.x + a
def via_namespace(a):
    return ns.x + a
def via_object(a):
    return h.x + a
via_lambda = lambda a: v + a
def via_method(a):
    return h.m(a)
"""
NAMES = (
    "calls", "Vec", "C", "v", "d", "lst", "i", "ns", "bump", "w", "time", "types", "big", "j", "Holder", "h",
    "via_dict", "via_list", "via_class", "via_namespace", "via_object", "via_lambda", "via_method", "make", "W", "P",
)  # fmt: skip


def _restart(magics):
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


@pytest.mark.parametrize(
    "statement",
    [
        f"d['k'] += 1 + {SLEEP}",
        f"w = d['k'] + (1 + {SLEEP})",
        f"lst[0] += 1 + {SLEEP}",
        f"w = lst[i] + (1 + {SLEEP})",
        f"C.x += 1 + {SLEEP}",
        f"w = C.x + (1 + {SLEEP})",
        f"ns.x += 1 + {SLEEP}",
        f"w = ns.x + (1 + {SLEEP})",
        f"w = bump({SLEEP})",
        f"w = big[j] + (1 + {SLEEP})",
        f"big[j] += 1 + {SLEEP}",
        f"w = big[-1] + (1 + {SLEEP})",
        f"w = via_dict({SLEEP})",
        f"w = via_list({SLEEP})",
        f"w = via_class({SLEEP})",
        f"w = via_namespace({SLEEP})",
        f"w = via_object({SLEEP})",
        f"w = via_lambda({SLEEP})",
        f"w = via_method({SLEEP})",
    ],
)
def test_a_served_statement_keeps_what_the_special_method_writes(cash_magics, statement):
    for _ in range(2):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, statement)
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)


def test_a_container_too_big_to_look_through_runs_with_a_reason(cash_magics, monkeypatch):
    """Past `ITEM_TYPES_AT` items an item read by an index not known here
    may be any object: the statement runs each time, and says why."""
    from cash.analysis import handed_callables

    monkeypatch.setattr(handed_callables, "ITEM_TYPES_AT", 100)
    monkeypatch.setattr(handed_callables, "_TOO_BIG_SAID", set())
    for statement in (f"w = big[j] + (1 + {SLEEP})", f"w = (via_list(j), {SLEEP})"):
        run_cash_cell(cash_magics, SETUP + "lst = big\n")
        before = len(cash_magics.shell.user_ns["calls"])
        for _ in range(2):
            run_cash_cell(cash_magics, statement)
        assert len(cash_magics.shell.user_ns["calls"]) - before == 2, statement


def _only_the_notebooks_classes(monkeypatch):
    """The gate sees only what a notebook with no modules of its own has:
    the test harness's classes (``MockShell``) would otherwise count."""
    from cash.analysis import handed_callables
    from cash.notebook import held_functions

    real = handed_callables.live_users_classes
    monkeypatch.setattr(held_functions, "users_module_classes", lambda: [])
    monkeypatch.setattr(
        handed_callables,
        "live_users_classes",
        lambda: [c for c in real() if not (c.__module__ or "").startswith(("tests", "conftest"))],
    )
    monkeypatch.setattr(handed_callables, "ITEM_TYPES_AT", 100)


TOO_BIG_SETUP = "import time\ncalls = []\n"


@pytest.mark.parametrize(
    "made",
    [
        # Made inside a function: no name binds the class.
        "def make():\n    class W:\n        def __add__(self, x):\n            calls.append(1)\n"
        "            return 0\n    return W()\nlst = [0] * 150 + [make()]\n",
        # Deleted from the namespace once its object was made.
        "class W:\n    def __add__(self, x):\n        calls.append(1)\n        return 0\n"
        "lst = [0] * 150 + [W()]\ndel W\n",
    ],
    ids=["made-in-a-function", "deleted"],
)
def test_a_class_no_name_binds_still_makes_a_too_big_container_run(cash_magics, monkeypatch, made):
    import warnings

    from cash.analysis import handed_callables

    _only_the_notebooks_classes(monkeypatch)
    # Said once for each container: the case before this one said it for ``lst``.
    monkeypatch.setattr(handed_callables, "_TOO_BIG_SAID", set())
    said = []
    for _ in range(2):
        run_cash_cell(cash_magics, TOO_BIG_SETUP + made + "i = 150\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_cash_cell(cash_magics, f"w = (lst[i] + 1, {SLEEP})")
        said += [str(w.message) for w in caught]
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)
    assert any("NOTEBOOK-ITEMS-UNREAD" in s for s in said), said


def test_with_no_such_class_a_too_big_container_is_served(cash_magics, monkeypatch):
    import warnings

    from cash.analysis import handed_callables

    _only_the_notebooks_classes(monkeypatch)
    # A class lives on while anything holds it, cash's memos of classes
    # included: the ``W`` the tests above made would still count (and make
    # this run, the cautious way). Only this test's classes are asked about.
    gate = handed_callables.live_users_classes
    monkeypatch.setattr(handed_callables, "live_users_classes", lambda: [c for c in gate() if c.__name__ == "P"])
    run_cash_cell(cash_magics, TOO_BIG_SETUP + "class P:\n    def __add__(self, x):\n        return 0\n"
                  "lst = [0] * 150 + [P()]\ni = 150\n")  # fmt: skip
    statement = f"w = (lst[i] + 1, {SLEEP})"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, statement)
        run_cash_cell(cash_magics, statement)
    assert not any("NOTEBOOK-ITEMS-UNREAD" in str(w.message) for w in caught)
    statuses = [str(s["status"]) for s in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses == ["RESTORED"], statuses


def test_a_class_exec_made_after_the_gate_was_asked_still_counts(cash_magics, monkeypatch):
    """``exec(src, g)`` with no ``__name__`` in ``g`` makes a class of
    module ``builtins``: judged by its own functions each time, not by a
    verdict kept for ``builtins`` when the gate was first asked."""
    _only_the_notebooks_classes(monkeypatch)
    from cash.analysis import handed_callables

    monkeypatch.setattr(handed_callables, "_TOO_BIG_SAID", set())
    # What the first ask records for ``builtins`` from Python's own
    # (``ExceptionGroup``): not the user's.
    monkeypatch.setattr(handed_callables, "_USERS_MODULE_NAMES", {"builtins": False})
    made = []
    for _ in range(2):
        run_cash_cell(cash_magics, "import time\ncalls = []\nbig = [0] * 152\nj = 5\n")
        monkeypatch.setattr(handed_callables, "ITEM_TYPES_AT", 100)
        run_cash_cell(cash_magics, f"w0 = (big[j] + 1, {SLEEP})")
        run_cash_cell(
            cash_magics,
            "src = 'class W:\\n    def __add__(self, x):\\n        calls.append(1)\\n        return 0\\n'\n"
            "g = {'calls': calls}\nexec(src, g)\nbig[151] = g['W']()\ndel g\ni = 151\n",
        )
        run_cash_cell(cash_magics, f"w = (big[i] + 1, {SLEEP})")
        made.append(list(cash_magics.shell.user_ns["calls"]))
        _restart(cash_magics)
        for name in ("big", "j", "w0", "src"):
            cash_magics.shell.user_ns.pop(name, None)
    assert made == [[1], [1]]


@pytest.mark.parametrize(
    "kind",
    [
        "class A:\n    x: int = 0\n",
        "import dataclasses\n@dataclasses.dataclass\nclass A:\n    x: int = 0\n",
        "class A:\n    def __eq__(self, other):\n        return isinstance(other, A)\n    __hash__ = None\n",
        "class A:\n    def __repr__(self):\n        return str(len('a'))\n",
    ],
    ids=["annotated", "dataclass", "eq-calls-isinstance", "repr-calls-len"],
)
def test_a_class_whose_special_methods_change_nothing_outside_lets_a_big_list_be_served(cash_magics, monkeypatch, kind):
    """An annotated class (3.14 writes it an ``__annotate__``), a dataclass,
    or special methods calling only builtins change nothing a statement
    leaves out: a big list of numbers read by an index is served."""
    _only_the_notebooks_classes(monkeypatch)
    from cash.analysis import handed_callables

    monkeypatch.setattr(handed_callables, "ITEM_TYPES_AT", 100)
    monkeypatch.setattr(handed_callables, "_TOO_BIG_SAID", set())
    # The ``W``s of the tests above live on in cash's memos: only ``A`` is asked about.
    gate = handed_callables.live_users_classes
    monkeypatch.setattr(handed_callables, "live_users_classes", lambda: [c for c in gate() if c.__name__ == "A"])
    run_cash_cell(cash_magics, "import time\n" + kind + "a0 = A()\nbig = [0.5] * 151\ni = 7\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, f"w = (big[i] + 1, {SLEEP})")
        run_cash_cell(cash_magics, f"w = (big[i] + 1, {SLEEP})")
    assert not any("NOTEBOOK-ITEMS-UNREAD" in str(w.message) for w in caught)
    assert [str(s["status"]) for s in cash_magics.cash_status("dict")["last_cell"]["statements"]] == ["RESTORED"]


def test_an_object_of_a_class_with_no_module_runs_as_plain_python(cash_magics):
    """``type('W', (), {})`` exec'd in a dict with no ``__name__`` makes a
    class that has no ``__module__`` at all: reading one raised inside cash,
    and the cell failed where plain Python runs it."""
    run_cash_cell(cash_magics, "g = {}\nexec(\"W = type('W', (), {})\", g)\n")
    run_cash_cell(cash_magics, "x = g['W']()\n")
    run_cash_cell(cash_magics, "y = 1\n")
    assert cash_magics.shell.user_ns["y"] == 1
