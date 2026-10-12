"""A helper whose code is swapped in place is keyed by the code it runs now.

IPython's ``%autoreload 2`` does not rebind a reloaded function: it keeps
the old function object (every ``from mylib import helper`` still holds it)
and sets ``old.__code__ = new.__code__``, ``old.__dict__ = new.__dict__``;
a class keeps its object and each method's code is swapped the same way.
A memo per function or class object then still described the old code,
and a cached function calling the edited helper served the old result.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core


def _module(tmp_path, monkeypatch, name: str, text: str):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(text), encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _caller(tmp_path, monkeypatch, cash_instance, name: str, imports: str, body: str):
    """A cached ``run(x)`` in a module of its own that reads the helper as a
    global, as a project module (or a notebook cell) does."""
    mod = _module(tmp_path, monkeypatch, name, f"{imports}\n\ndef run(x):\n    return {body}\n")
    return cash_instance.cache(mod.run)


def _swap_code(tmp_path, mod, text: str) -> dict:
    """Edit *mod*'s file and patch its functions in place, as autoreload does."""
    path = tmp_path / f"{mod.__name__}.py"
    path.write_text(textwrap.dedent(text), encoding="utf-8", newline="\n")
    ns: dict = {"__name__": mod.__name__}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


def test_a_helper_with_new_code_is_run_again(tmp_path, monkeypatch, cash_instance):
    mod = _module(tmp_path, monkeypatch, "swap_fn_mod", "def helper(v):\n    return v * 10\n")
    run = _caller(tmp_path, monkeypatch, cash_instance, "swap_fn_user", "from swap_fn_mod import helper", "helper(x)")

    assert run(3) == 30
    new = _swap_code(tmp_path, mod, "def helper(v):\n    return v * 20 + 0\n")
    mod.helper.__code__ = new["helper"].__code__
    assert run(3) == run.__wrapped__(3) == 60


def test_a_method_with_new_code_is_run_again(tmp_path, monkeypatch, cash_instance):
    old = "class M:\n    @staticmethod\n    def helper(v):\n        return v * 10\n"
    mod = _module(tmp_path, monkeypatch, "swap_cls_mod", old)
    run = _caller(tmp_path, monkeypatch, cash_instance, "swap_cls_user", "from swap_cls_mod import M", "M.helper(x)")

    assert run(3) == 30
    new = _swap_code(tmp_path, mod, old.replace("v * 10", "v * 20"))
    mod.M.__dict__["helper"].__func__.__code__ = new["M"].__dict__["helper"].__func__.__code__
    assert run(3) == run.__wrapped__(3) == 60


def test_what_the_new_code_calls_is_keyed(tmp_path, monkeypatch, cash_instance):
    text = "K = 1\ndef inner(v):\n    return v + K\ndef helper(v):\n    return v * 10\n"
    mod = _module(tmp_path, monkeypatch, "swap_reach_mod", text)
    run = _caller(tmp_path, monkeypatch, cash_instance, "swap_reach_user", "from swap_reach_mod import helper", "helper(x)")

    assert run(3) == 30
    new = _swap_code(tmp_path, mod, text.replace("return v * 10", "return inner(v) * 20"))
    mod.helper.__code__ = new["helper"].__code__
    assert run(3) == 80
    # The helper the new code calls reads K: its value is in the key now.
    monkeypatch.setattr(mod, "K", 5)
    assert run(3) == run.__wrapped__(3) == 160


def test_a_wrapped_helper_with_new_code_is_run_again(tmp_path, monkeypatch, cash_instance):
    text = "def helper(v):\n    return v * 10\n"
    mod = _module(tmp_path, monkeypatch, "swap_wraps_mod", text)

    imports = (
        "import functools\nimport swap_wraps_mod\n\n"
        "def logged(fn):\n    @functools.wraps(fn)\n    def wrapper(*a):\n        return fn(*a)\n    return wrapper\n\n"
        "helper = logged(swap_wraps_mod.helper)"
    )
    run = _caller(tmp_path, monkeypatch, cash_instance, "swap_wraps_user", imports, "helper(x)")

    assert run(3) == 30
    new = _swap_code(tmp_path, mod, text.replace("v * 10", "v * 20"))
    # The wrapped function itself is patched, as autoreload patches the
    # function object it finds in the module.
    mod.helper.__code__ = new["helper"].__code__
    assert run(3) == run.__wrapped__(3) == 60
