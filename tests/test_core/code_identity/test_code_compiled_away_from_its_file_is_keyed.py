"""Code compiled from other text than its file at its line is keyed by its
own definition.

IPython 9's ``%autoreload 2`` (deduperreload) recompiles an edited function
from ``ast.unparse`` of its definition alone and swaps the code into the old
function object. The code then says it starts on line 1 (or 2) of the
module, and a ``@cash.cache`` function's of the decorator's file, so reading
the source at that line gave ``import cash`` for every version of every
patched function: the helper's key stopped moving after its first edit, and
the cached function's own text held no call to the helper at all, so the
helper left its key. A plain ``importlib.reload`` (a script) compiles the
whole file and was never affected.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import textwrap
import time

import pytest

from cash import source_reading
from cash.code_digest import callable_identity

pytestmark = pytest.mark.core


def _write(path, text: str, stamp: list[float]) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8", newline="\n")
    # Distinct and settled: each version has its own (mtime, size) and is
    # old enough for cash to memoise what it reads from it.
    stamp[0] += 10
    os.utime(path, (stamp[0], stamp[0]))


def _import(tmp_path, monkeypatch, name: str):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _compiled_alone(module, func_name: str, filename: str):
    """*func_name*'s definition in *module*'s file, compiled on its own as
    deduperreload compiles it: line 1, under *filename*."""
    with open(module.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    node.decorator_list = []
    namespace: dict = {}
    exec(compile(ast.unparse(node), filename, "exec"), vars(module), namespace)
    return namespace[func_name].__code__


HELPER = """\
    import os

    SCALE = 1

    def helper(v):
        return v * {k}
"""


def test_the_source_of_a_function_compiled_alone_is_its_definition(tmp_path, monkeypatch):
    stamp = [time.time() - 600]
    path = tmp_path / "alone_src_mod.py"
    _write(path, HELPER.format(k=10), stamp)
    mod = _import(tmp_path, monkeypatch, "alone_src_mod")

    _write(path, HELPER.format(k=20), stamp)
    mod.helper.__code__ = _compiled_alone(mod, "helper", str(path))
    assert mod.helper.__code__.co_firstlineno == 1
    assert source_reading.getsource(mod.helper) == "def helper(v):\n    return v * 20\n"
    assert source_reading.defining_file(mod.helper) == str(path)


def test_a_function_compiled_under_another_file_is_found_in_its_module(tmp_path, monkeypatch):
    stamp = [time.time() - 600]
    path = tmp_path / "alone_file_mod.py"
    other = tmp_path / "alone_decorators.py"
    _write(other, '"""A decorator module."""\n\ndef deco(f):\n    return f\n', stamp)
    _write(path, HELPER.format(k=10), stamp)
    mod = _import(tmp_path, monkeypatch, "alone_file_mod")

    _write(path, HELPER.format(k=30), stamp)
    mod.helper.__code__ = _compiled_alone(mod, "helper", str(other))
    assert source_reading.getsource(mod.helper) == "def helper(v):\n    return v * 30\n"
    assert source_reading.defining_file(mod.helper) == str(path)


def test_code_found_in_no_file_has_no_source(tmp_path, monkeypatch):
    """Code no file holds is keyed by its bytecode, not by the text at its line."""
    stamp = [time.time() - 600]
    path = tmp_path / "alone_none_mod.py"
    _write(path, HELPER.format(k=10), stamp)
    mod = _import(tmp_path, monkeypatch, "alone_none_mod")

    mod.helper.__code__ = compile("def helper(v):\n    return v * 99\n", str(path), "exec").co_consts[0]
    with pytest.raises(OSError):
        source_reading.getsource(mod.helper)
    assert source_reading.defining_file(mod.helper) is None
    before = callable_identity(mod.helper)
    mod.helper.__code__ = compile("def helper(v):\n    return v * 98\n", str(path), "exec").co_consts[0]
    assert callable_identity(mod.helper) != before


def test_each_edit_of_a_helper_compiled_alone_is_run(tmp_path, monkeypatch, cash_instance):
    stamp = [time.time() - 600]
    path = tmp_path / "alone_seq_mod.py"
    _write(path, "def _ratio(a, b):\n    return a / b + 0\n", stamp)
    mod = _import(tmp_path, monkeypatch, "alone_seq_mod")
    namespace = {"_ratio": mod._ratio}
    exec("def feat(x):\n    return _ratio(x, 2) + 1\n", namespace)
    feat = cash_instance.cache(namespace["feat"])
    assert feat(4) == 3

    # The first edit moved the text read at line 1; the second did not.
    for h in (100, 200, 300):
        _write(path, f"def _ratio(a, b):\n    return a / b + {h}\n", stamp)
        mod._ratio.__code__ = _compiled_alone(mod, "_ratio", str(path))
        assert feat(4) == feat.__wrapped__(4) == 3 + h


# --- IPython's own deduplicating reload ------------------------------------

MODULE = (
    "from alone_holder import CASH\n"
    "{head}\n"
    "@CASH.cache\n"
    "def feat(x):\n"
    "    return {call} + {f}\n"
    "\n"
    "@CASH.cache\n"
    "def build(x):\n"
    "    return feat(x) * 10\n"
)

#: name -> (the module's helper text, the call feat makes, extra modules).
SHAPES = {
    "helper": ("def _ratio(a, b):\n    return a / b + {h}\n", "_ratio(x, 2)", {}),
    "method": ("class K:\n    def r(self, a, b):\n        return a / b + {h}\n", "K().r(x, 2)", {}),
    "constant": ("H = {h}\n\ndef _ratio(a, b):\n    return a / b + H\n", "_ratio(x, 2)", {}),
    "other_module": (
        "import alone_other_{tag}\n",
        "alone_other_{tag}._ratio(x, 2)",
        {"alone_other_{tag}": "def _ratio(a, b):\n    return a / b + {h}\n"},
    ),
}

#: (helper constant, feat constant) per edit, in the order of the edit sequence that went stale.
EDITS = [(0, 2), (100, 2), (200, 2), (300, 2), (300, 3), (400, 3), (500, 3)]


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_deduperreload_edits_reach_the_key(tmp_path, monkeypatch, cash_instance, shape):
    deduper = pytest.importorskip("IPython.extensions.deduperreload.deduperreload")
    head, call, extra = SHAPES[shape]
    tag = shape
    stamp = [time.time() - 600]
    (tmp_path / "alone_holder.py").write_text("CASH = None\n", encoding="utf-8")
    name = f"alone_dd_{tag}"
    paths = {name: tmp_path / f"{name}.py"}
    paths.update({mod.format(tag=tag): tmp_path / f"{mod.format(tag=tag)}.py" for mod in extra})

    def write(h: int, f: int) -> None:
        texts = {name: MODULE.format(head=head.format(h=h, tag=tag), call=call.format(tag=tag), f=f)}
        texts.update({mod.format(tag=tag): text.format(h=h) for mod, text in extra.items()})
        for mod_name, text in texts.items():
            text = textwrap.dedent(text)
            if paths[mod_name].exists() and paths[mod_name].read_text(encoding="utf-8") == text:
                continue
            _write(paths[mod_name], text, stamp)

    write(0, 1)
    holder = _import(tmp_path, monkeypatch, "alone_holder")
    monkeypatch.setattr(holder, "CASH", cash_instance)
    mod = importlib.import_module(name)
    reloader = deduper.DeduperReloader()
    reloader.update_sources()
    assert (mod.feat(4), mod.build(4)) == (3, 30)

    for h, f in EDITS:
        write(h, f)
        for mod_name in paths:
            if mod_name != name:
                assert reloader.maybe_reload_module(sys.modules[mod_name])
        assert reloader.maybe_reload_module(mod)
        want = 2 + h + f
        assert (mod.feat(4), mod.build(4)) == (want, want * 10), (h, f)
        assert (mod.feat(4), mod.build(4)) == (want, want * 10), (h, f)
