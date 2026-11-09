"""A weak reference cash keeps to an object does not keep a statement out.

A variable finding an object by a weak reference keeps the statement that
changes the object out of the cache: restored, the object would be a copy
the reference does not find (NOTEBOOK-IDENTITY-HELD). Cash's own memos keep
weak references to the user's objects too -- the decorator's memo of the
arguments it hashed holds one to each -- and a plain ``weakref.ref`` of
cash's was taken for a variable's. Whether the loop below was stored then
hung on what cash's memos held at that moment, which earlier work decided:
in a full test run it ran again now and then. Every weak reference cash
makes is known as cash's (`cash._weak`).
"""

from __future__ import annotations

import ast
import gc
import pathlib
import weakref

import pytest

import cash
from cash import _weak
from cash.notebook.statement.identity_holders import _weakly_found_by_others
from tests._cell_driver import run_cash_cell

SETUP = (
    "import time\n"
    "class Other:\n    registry = []\n"
    "class Model:\n"
    "    def __init__(self, k):\n        self.k = k\n"
    "    def fit(self, X):\n"
    "        time.sleep(0.05)\n"
    "        self.w = [[[[[x * self.k for x in X]]]]]\n"
    "        return self\n"
    "X = [1, 2, 3]"
)
MODELS = "models = {'a': Model(1), 'b': Model(2)}"
LOOP = "for est in models.values():\n    est.fit(X)"


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    gc.collect()  # what earlier tests left: see the loop tests' own fixture


def _statuses(cash_magics):
    return [(str(s["status"]), s.get("uncacheable_reasons")) for s in cash_magics._last_cell_metrics["statements"]]


@pytest.mark.usefixtures("store_everything")
def test_a_loop_whose_objects_a_memo_of_cash_holds_is_stored(cash_magics, mock_shell, cash_instance):
    cells = [SETUP, MODELS, LOOP]
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    # As a cached call given each model would leave them (`ArgHasher.hash_payload`).
    for key in ("a", "b"):
        cash_instance._args._memo_arg_hash(mock_shell.user_ns["models"][key], "lineage", "digest")
    run_cash_cell(cash_magics, LOOP, cells=cells)
    run_cash_cell(cash_magics, MODELS, cells=cells)
    run_cash_cell(cash_magics, LOOP, cells=cells)
    assert "RESTORED" in str(_statuses(cash_magics)), _statuses(cash_magics)


class Held:
    pass


def test_cash_s_weak_references_are_told_from_the_user_s():
    mine = [Held() for _ in range(5)]
    kept = [
        _weak.ref(mine[0]),
        _weak.WeakSet([mine[1]]),
        _weak.WeakKeyDictionary({mine[2]: 1}),
        _weak.WeakValueDictionary({1: mine[3]}),
    ]
    assert not _weakly_found_by_others(mine[:4])
    users = [weakref.ref(mine[4])]
    assert _weakly_found_by_others(mine[4:])
    for made in (weakref.WeakSet, weakref.WeakKeyDictionary):
        other = Held()
        users.append(made([other]) if made is weakref.WeakSet else made({other: 1}))
        assert _weakly_found_by_others([other])
    assert kept and users


_CONSTRUCTORS = {"WeakKeyDictionary", "WeakValueDictionary", "WeakSet"}


def test_cash_makes_no_weak_reference_it_would_not_know():
    """Every ``weakref.ref`` cash makes carries a callback, and every weak
    container is one of `cash._weak`'s: a plain one is taken for the
    user's."""
    root = pathlib.Path(cash.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_weak.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            plain = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "weakref"
            if plain and (
                func.attr in _CONSTRUCTORS or (func.attr == "ref" and len(node.args) + len(node.keywords) < 2)
            ):
                found.append(f"{path.relative_to(root)}:{node.lineno}")
            if isinstance(func, ast.Name) and func.id == "map" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name):
                    if first.value.id == "weakref" and first.attr == "ref":
                        found.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not found, found
