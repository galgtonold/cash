"""Every module-level memo of the notebook path says how long it lives.

A dict, list, set or memo object kept at a module's top level outlives the
session that filled it. Each one is listed with a scope
(``cash.notebook.memo_lifetimes.LIFETIMES``), and ``cash.reset_session()``
clears the ones a session owns through that list, so a memo added without
an entry fails here instead of carrying one session's facts into the next.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import cash
from cash.notebook import cache_key, lineage_formula

_ROOT = Path(cash.__file__).parent / "notebook"
#: The calls that make an empty container.
_MAKERS = frozenset(
    {
        "dict",
        "list",
        "set",
        "OrderedDict",
        "defaultdict",
        "deque",
        "Counter",
        "WeakKeyDictionary",
        "WeakValueDictionary",
        "WeakSet",
    }
)


def _starts_empty(node: ast.expr) -> bool:
    """Whether *node* makes a container to be filled later: an empty dict,
    list or set, a list of empty slots (``[None]``, ``[[], []]``), or a memo
    object. A table written out in full is data, not a memo."""
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.List):
        return all(_starts_empty(e) or (isinstance(e, ast.Constant) and e.value is None) for e in node.elts)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return (name in _MAKERS and not node.args) or name.endswith("Memo")
    return False


def _module_level_containers() -> dict[str, dict[str, bool]]:
    """``module -> {name: whether it starts as a plain empty list}``."""
    found: dict[str, dict[str, bool]] = {}
    for path in sorted(_ROOT.rglob("*.py")):
        parts = path.relative_to(_ROOT).with_suffix("").parts
        module = ".".join(("cash", "notebook", *(p for p in parts if p != "__init__")))
        for statement in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets, value = [statement.target], statement.value
            else:
                continue
            if not _starts_empty(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    found.setdefault(module, {})[target.id] = isinstance(value, ast.List)
    return found


_FOUND = _module_level_containers()


@pytest.fixture
def registry():
    from cash import _memo_registry

    return _memo_registry


@pytest.fixture
def lifetimes():
    from cash.notebook.memo_lifetimes import LIFETIMES

    return LIFETIMES


def test_the_scan_finds_the_memos():
    """Guards the scan itself: an empty scan would pass every check below."""
    assert len(_FOUND) > 20
    assert "_LINEAGE_ALIASES" in _FOUND["cash.notebook.lineage_formula"]
    assert "_PLACE" in _FOUND["cash.notebook.compiled_source"]


def test_every_module_level_container_has_a_lifetime(lifetimes):
    missing = sorted(
        f"{module}.{name}"
        for module, names in _FOUND.items()
        for name in names
        if name not in lifetimes.get(module, {})
    )
    assert not missing, (
        "module-level containers without a scope in cash/notebook/memo_lifetimes.py "
        f"(statement, cell, session, process or block): {missing}"
    )


def test_every_lifetime_names_a_container_that_exists(lifetimes):
    gone = sorted(
        f"{module}.{name}"
        for module, names in lifetimes.items()
        for name in names
        if name not in _FOUND.get(module, {})
    )
    assert not gone, f"listed in memo_lifetimes.py but no longer a module-level container: {gone}"


def test_every_lifetime_is_registered_with_its_scope(registry, lifetimes):
    registered = registry.registered()
    for module, names in lifetimes.items():
        for name, scope in names.items():
            assert registered[f"{module}.{name}"] == scope


def test_a_memo_a_reset_clears_can_be_cleared(registry, lifetimes):
    """It has a ``clear``, and is not a list of slots read by index
    (``_X[0]``), which emptied would raise where it is read."""
    checked = 0
    for module, names in lifetimes.items():
        for name, scope in names.items():
            if scope not in registry.CLEARED_SCOPES:
                continue
            memo = getattr(importlib.import_module(module), name)
            assert callable(getattr(memo, "clear", None)), (module, name)
            if _FOUND[module][name]:
                assert memo == [] or not any(item is None or item == [] for item in memo), (module, name)
            checked += 1
    assert checked > 10


def test_a_session_reset_empties_what_the_session_learnt(monkeypatch, registry):
    for module, name in (
        (lineage_formula, "_LINEAGE_ALIASES"),
        (lineage_formula, "_SETTING_RUNS"),
        (cache_key, "_DRAW_KEYS"),
    ):
        monkeypatch.setattr(module, name, {})
    lineage_formula.record_lineage_alias("lineage-of-notebook-a", "earlier-lineage")
    lineage_formula.setting_lineage("module-lineage", "setting-1")
    cache_key._remember_draw_key("stmt:random-key", "stable-key")
    kept = lineage_formula._FORWARD_HELPERS  # a session memo that is an object, not a dict
    kept["digest"] = frozenset({"helper"})
    process = cache_key._GLOBAL_NAMES_MEMO
    process["a code object"] = frozenset({"name"})
    try:
        cleared = registry.reset("session")
        assert not lineage_formula._LINEAGE_ALIASES
        assert not lineage_formula._SETTING_RUNS
        assert not cache_key._DRAW_KEYS
        assert len(kept) == 0
        assert "cash.notebook.lineage_formula._LINEAGE_ALIASES" in cleared
        # What is true of the process is kept.
        assert process.peek("a code object") == frozenset({"name"})
        assert "cash.notebook.cache_key._GLOBAL_NAMES_MEMO" not in cleared
    finally:
        process.pop("a code object")


def test_reset_session_resets_the_session_memos(monkeypatch, registry):
    asked = []
    monkeypatch.setattr(registry, "reset", lambda scope="session": asked.append(scope) or [])
    monkeypatch.setattr(cash._active, "set_default_cash", lambda instance: None)
    cash.reset_session()
    assert asked == ["session"]


def test_a_shorter_reset_leaves_the_session_memos(monkeypatch, registry):
    monkeypatch.setattr(lineage_formula, "_LINEAGE_ALIASES", {"a": "b"})
    cleared = registry.reset("cell")
    assert lineage_formula._LINEAGE_ALIASES == {"a": "b"}
    assert all(registry.registered()[name] in ("statement", "cell") for name in cleared)


def test_a_clear_that_raises_does_not_keep_the_others(monkeypatch, registry):
    monkeypatch.setattr(registry, "_MEMOS", dict(registry._MEMOS))
    seen = []

    def broken():
        raise RuntimeError("cannot clear")

    registry.register_memo("tests.first", broken, "session")
    registry.register_memo("tests.second", lambda: seen.append("second"), "session")
    cleared = registry.reset("session")
    assert seen == ["second"]
    assert "tests.second" in cleared and "tests.first" not in cleared


def test_a_memo_needs_a_known_scope_and_a_clear(registry):
    with pytest.raises(ValueError, match="unknown memo scope"):
        registry.register_memo("tests.memo", dict().clear, "kernel")
    with pytest.raises(ValueError, match="needs a clear"):
        registry.register_memo("tests.memo", None, "session")
    with pytest.raises(ValueError, match="a reset clears"):
        registry.reset("process")
