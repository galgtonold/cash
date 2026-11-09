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
import copy
import gc
import pathlib
import re
import weakref

import pytest

import cash
from cash import _native, _weak
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


def test_a_callback_of_cash_s_does_not_make_the_user_s_reference_cash_s(cash_instance):
    """``weakref.ref(m, cached)`` or ``weakref.ref(m, cached.cache_clear)``:
    the user's reference, though its callback is cash's code."""

    @cash_instance.cache
    def cached(*args):
        return 1

    for callback in (cached, cached.cache_clear, cash_instance.cache):
        held = Held()
        reference = weakref.ref(held, callback)
        assert not _weak.is_cash_ref(reference), callback
        assert _weakly_found_by_others([held]), callback
    held = Held()
    proxy = weakref.proxy(held)
    assert not _weak.is_cash_ref(proxy)  # its attributes are its object's
    assert _weakly_found_by_others([held]) and proxy is not None
    mine = Held()
    _weak.finalize(mine, print)
    assert not _weakly_found_by_others([mine])
    users = Held()
    weakref.finalize(users, print)
    assert _weakly_found_by_others([users])


def test_a_copy_of_a_container_of_cash_s_is_cash_s():
    """The standard ``copy()`` (and ``|``, which copies) gives a standard
    container, whose references are not known as cash's."""
    held = Held()
    for container, other in (
        (_weak.WeakKeyDictionary({held: 1}), {}),
        (_weak.WeakValueDictionary({1: held}), {}),
        (_weak.WeakSet([held]), set()),
    ):
        made = [container.copy(), copy.copy(container), container | other]
        if isinstance(other, dict):
            made.append(other | container)
        for each in made:
            assert type(each) is type(container), (type(container), type(each))
            assert len(each) == 1


def test_the_c_extension_s_weak_references_are_cash_s():
    """The C watchers keep a weak reference to each value they watch, for
    as long as it lives (``cwatch.c``)."""
    if not _native.AVAILABLE:
        pytest.skip("no C extension")
    watched = frozenset({1, 2})
    _native.cwatch.arm(watched)
    assert weakref.getweakrefs(watched)
    assert not _weakly_found_by_others([watched])


FINALIZERS = {
    # Bound to a variable.
    "a-function-of-the-user-s": "def cb():\n    pass\nf = weakref.finalize(models['a'], cb)",
    # The object named too, as ``m`` where the models are made: the
    # loop's walk does not find ``f``'s reference, which
    # ``weakref.finalize`` keeps out of sight.
    "the-object-named": "def cb():\n    pass\nf = weakref.finalize(m, cb)",
    # Held by weakref.finalize's own registry alone.
    "held-by-nothing": "weakref.finalize(models['a'], print, 'gone')",
    "a-cached-function": "f = weakref.finalize(models['a'], cleanup)",
}


def _finalized(obj):
    """Whether a ``weakref.finalize`` still watches *obj*."""
    return any(info.weakref() is obj for info in list(weakref.finalize._registry.values()))


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("kind", sorted(FINALIZERS))
def test_a_finalizer_of_the_user_s_keeps_a_loop_out(kind, cash_magics, mock_shell, cash_instance):
    """``weakref.finalize(models['a'], cb)`` watches the object: restored,
    the loop put a copy in its place, and the finalizer watched (and ran
    for) the object that went."""
    mock_shell.user_ns["cash_instance"] = cash_instance
    # Weights cash looks all the way through: a walk cut short would find
    # the finalizer's reference from the object (`_weakly_found_by_others`).
    flat = SETUP.replace("[[[[[x * self.k for x in X]]]]]", "[x * self.k for x in X]")
    setup = flat + "\nimport weakref\n@cash_instance.cache\ndef cleanup(*a):\n    return 1\n"
    models = MODELS + "\nm = models['a']" if kind == "the-object-named" else MODELS
    cells = [setup, models, FINALIZERS[kind], LOOP]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    try:
        assert _finalized(mock_shell.user_ns["models"]["a"]), _statuses(cash_magics)
    finally:
        for finalizer, info in list(weakref.finalize._registry.items()):
            if type(info.weakref()).__name__ == "Model":
                finalizer.detach()


DEEP_REFS = {
    "plain": "r = weakref.ref(models['a'])",
    "a-cached-function-s-cache-clear": "r = weakref.ref(models['a'], cleanup.cache_clear)",
    "a-cached-function": "r = weakref.ref(models['a'], cleanup)",
}


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("kind", sorted(DEEP_REFS))
def test_a_reference_of_the_user_s_past_where_cash_looks_keeps_a_loop_out(kind, cash_magics, mock_shell, cash_instance):
    """Kept deeper than `_WEAK_HOLDER_LEVELS`, the reference is found from
    the object (`_weakly_found_by_others`): with a callback of cash's code,
    it was taken for cash's own."""
    mock_shell.user_ns["cash_instance"] = cash_instance
    setup = SETUP + "\nimport weakref\n@cash_instance.cache\ndef cleanup(*a):\n    return 1\n"
    hold = DEEP_REFS[kind] + "\nholder = {'a': {'b': {'c': {'d': {'e': {'f': r}}}}}}\ndel r"
    cells = [setup, MODELS, hold, LOOP]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    found = mock_shell.user_ns["holder"]["a"]["b"]["c"]["d"]["e"]["f"]()
    assert found is mock_shell.user_ns["models"]["a"], _statuses(cash_magics)


#: What makes a weak reference (or holds some), as ``weakref`` or
#: ``_weakref`` names it.
_MAKERS = {
    "ref",
    "ReferenceType",
    "proxy",
    "KeyedRef",
    "WeakMethod",
    "finalize",
    "WeakKeyDictionary",
    "WeakValueDictionary",
    "WeakSet",
}
_MODULES = {"weakref", "_weakref", "_weakrefset"}
#: Checks of a type, which make no reference: ``isinstance(x, weakref.ref)``.
_TYPE_CHECKS = {"isinstance", "issubclass", "instance_of"}
#: cash's subclasses of a reference type: ``_PinRef`` refers to cash's own
#: marker object only.
_SUBCLASSES = {"_PinRef"}


class _Uses(ast.NodeVisitor):
    """Each place a module of cash names a maker of weak references
    (`_MAKERS`), through ``import weakref [as w]`` or ``from weakref import
    ref [as r]``, other than where it makes none: an annotation, a type's
    subscript, a type check, a comparison, a tuple or set of types, an
    attribute of it (``weakref.finalize._registry``)."""

    def __init__(self) -> None:
        self.modules: set[str] = set()
        self.names: set[str] = set()
        self.found: list[int] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in _MODULES:
                self.modules.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in _MODULES:
            for alias in node.names:
                if alias.name in _MAKERS or alias.name == "*":
                    self.names.add(alias.asname or alias.name)

    def is_maker(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id in self.modules and node.attr in _MAKERS
        return isinstance(node, ast.Name) and node.id in self.names

    def scan(self, tree: ast.Module) -> list[int]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                self.visit(node)
        allowed: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.arg | ast.AnnAssign) and node.annotation is not None:
                allowed.update(map(id, ast.walk(node.annotation)))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
                allowed.update(map(id, ast.walk(node.returns)))
            elif isinstance(node, ast.Subscript):
                allowed.update(map(id, ast.walk(node.slice)))
            elif isinstance(node, ast.Attribute):
                allowed.add(id(node.value))
            elif isinstance(node, ast.Compare):
                allowed.update(id(part) for part in (node.left, *node.comparators))
            elif isinstance(node, ast.Tuple | ast.Set):
                allowed.update(id(part) for part in node.elts)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _TYPE_CHECKS:
                allowed.update(map(id, ast.walk(node.args[1])) if len(node.args) > 1 else ())
            elif isinstance(node, ast.ClassDef) and node.name in _SUBCLASSES:
                allowed.update(id(base) for base in node.bases)
        return sorted(node.lineno for node in ast.walk(tree) if self.is_maker(node) and id(node) not in allowed)


def _made_outside_weak(source: str) -> list[int]:
    return _Uses().scan(ast.parse(source))


def test_the_scan_finds_every_way_to_make_a_reference():
    """`test_cash_makes_no_weak_reference_it_would_not_know`'s scan, on
    each way the standard library makes a weak reference."""
    made = [
        "import weakref\nweakref.ref(x)",
        "import weakref\nweakref.ref(x, None)",
        "import weakref\nweakref.ref(x, lambda r: None)",
        "import weakref as wr\nwr.ref(x, cb)",
        "from weakref import ref\nref(x, cb)",
        "from weakref import WeakSet as S\nS()",
        "import weakref\nweakref.WeakMethod(o.m)",
        "import weakref\nweakref.proxy(x)",
        "import weakref\nweakref.finalize(x, cb)",
        "import functools, weakref\nfunctools.partial(weakref.ref)",
        "import weakref\nlist(map(weakref.ref, xs))",
        "import weakref\nmake = weakref.ref",
        "import weakref\ndef f(make=weakref.ref): pass",
        "import _weakref\n_weakref.ref(x)",
        "import weakref\nweakref.WeakKeyDictionary()",
        "import weakref\nclass Mine(weakref.ref): pass",
    ]
    for source in made:
        assert _made_outside_weak(source), source
    none = [
        "import weakref\nisinstance(x, weakref.ref)",
        "import weakref\nx: weakref.ref = y",
        "import weakref\ndef f(r: weakref.ref) -> weakref.ref: pass",
        "import weakref\nm: dict[int, weakref.ref] = {}",
        "import weakref\nweakref.finalize._registry.get(f)",
        "import weakref\ntype(c) is weakref.finalize",
        "import weakref\nKINDS = (weakref.ref, weakref.ProxyType)",
        "import weakref\nweakref.getweakrefs(x)",
    ]
    for source in none:
        assert not _made_outside_weak(source), source


def test_cash_makes_no_weak_reference_it_would_not_know():
    """Every weak reference cash makes comes from `cash._weak`, which makes
    it known as cash's: one made otherwise is taken for the user's (or,
    with a callback of cash's code, the user's for cash's)."""
    root = pathlib.Path(cash.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_weak.py":
            continue
        found.extend(f"{path.relative_to(root)}:{line}" for line in _made_outside_weak(path.read_text("utf-8")))
    assert not found, found


def test_the_c_extension_makes_no_weak_reference_it_would_not_know():
    """A weak reference the C extension makes carries cash's callback
    (``set_weak_callback``): with none it is the plain one a user's
    ``weakref.ref(x)`` shares, and taken for the user's."""
    root = pathlib.Path(cash.__file__).parent / "_c"
    found = []
    for path in sorted([*root.glob("*.c"), *root.glob("*.h")]):
        text = path.read_text("utf-8")
        for match in re.finditer(r"PyWeakref_New(?:Ref|Proxy)\s*\(([^;]*?)\)\s*;", text):
            if re.search(r",\s*(?:NULL|Py_None)\s*$", match.group(1)) or "Proxy" in match.group(0):
                found.append(f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}")
    assert not found, found
