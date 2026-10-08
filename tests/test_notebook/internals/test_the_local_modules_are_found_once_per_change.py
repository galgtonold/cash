"""The local modules a store into a module's object is looked up in are found
again only when ``sys.modules`` changes, not on every statement.

A local module can sit in ``sys.modules`` under another key than its
``__name__``: ``multiprocessing`` keeps a script's ``__main__`` as
``__mp_main__``, and IPython then puts its own ``__main__`` in the name's
place. Checked by name, the memo was stale for good and every statement
walked every loaded module again (the speed set's file loop ran 2-3x slower
after a decorator scenario had loaded ``multiprocessing``). Pins the work, not
the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import sys
import types

import pytest

from cash.notebook import callee_reach


@pytest.fixture
def aliased_module(tmp_path):
    """A local module held only under a key that is not its name."""
    path = tmp_path / "pc_aliased_helpers.py"
    path.write_text("K = 1\n", encoding="utf-8")
    module = types.ModuleType("pc_aliased_helpers_name")
    module.__file__ = str(path)
    key = "__pc_alias_of_pc_aliased_helpers__"
    sys.modules[key] = module
    try:
        yield key, module
    finally:
        sys.modules.pop(key, None)


def _scans(monkeypatch) -> list[object]:
    seen: list[object] = []
    real = callee_reach._is_local
    monkeypatch.setattr(callee_reach, "_is_local", lambda m: seen.append(m) or real(m))
    return seen


def test_a_module_under_another_key_does_not_make_every_call_walk_the_modules(aliased_module, monkeypatch):
    _key, module = aliased_module
    assert module in callee_reach._local_modules()
    seen = _scans(monkeypatch)
    for _ in range(3):
        assert module in callee_reach._local_modules()
    assert seen == []


def test_the_modules_are_found_again_when_the_entry_holds_another_module(aliased_module, tmp_path):
    key, module = aliased_module
    assert module in callee_reach._local_modules()
    other = types.ModuleType("pc_aliased_helpers_name")
    other.__file__ = module.__file__
    sys.modules[key] = other
    found = callee_reach._local_modules()
    assert other in found
    assert module not in found
