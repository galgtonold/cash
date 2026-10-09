"""A helper edited or rebound later is still followed as it is now.

Analysing a chain of helpers remembers, for one analysis pass, which code
can reach a module namespace by name, and checks loaded code against its
file through an index built once per file version. Neither may hide a
change: a helper edited on disk still reads as edited, and a helper rebound
after a pass is followed as it is bound in the next, so a global it reads
by name is keyed.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap

import pytest

import cash.loaded_code as loaded_code_module
from cash.decorator.closure_fold import iter_code_scopes
from cash.decorator.global_reads import reaches_namespace_by_name

pytestmark = pytest.mark.core

CHAIN = 20


def _chain_module(tmp_path, monkeypatch, name: str, extra: str = "") -> object:
    lines = [f"def h{i}(x):\n    return h{i + 1}(x) + 1\n" for i in range(CHAIN - 1)]
    lines.append(f"def h{CHAIN - 1}(x):\n    return x\n")
    lines.append("def top(x):\n    return h0(x)\n")
    path = tmp_path / f"{name}.py"
    path.write_text("\n\n".join(lines) + textwrap.dedent(extra), encoding="utf-8")
    # Settled, as a file the user saved a while ago is: its compiled form is
    # then kept per file version.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 60_000_000_000))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # no .pyc: loaded code is checked by compiling
    importlib.invalidate_caches()
    return importlib.import_module(name)


def test_an_edited_helper_in_the_chain_still_reads_as_edited(tmp_path, monkeypatch):
    mod = _chain_module(tmp_path, monkeypatch, "chain_edited_mod")
    path = tmp_path / "chain_edited_mod.py"
    for i in range(CHAIN):
        assert loaded_code_module.loaded_code_matches_disk(getattr(mod, f"h{i}"))
    text = path.read_text(encoding="utf-8")
    edited = text.replace(f"def h{CHAIN - 1}(x):\n    return x\n", f"def h{CHAIN - 1}(x):\n    return x * 7\n")
    path.write_text(edited, encoding="utf-8")
    # A new file version: the index is built from the new text.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 30_000_000_000))
    assert not loaded_code_module.loaded_code_matches_disk(getattr(mod, f"h{CHAIN - 1}"))
    assert loaded_code_module.loaded_code_matches_disk(mod.h0)


READER = """

def plain(name):
    return name


def reader(name):
    return globals()[name]


relay = plain
K = 1


def via(name):
    return relay(name)


def outer():
    return via("K")
"""


def test_a_helper_rebound_after_a_pass_is_followed_as_it_is_now(tmp_path, monkeypatch):
    mod = _chain_module(tmp_path, monkeypatch, "chain_rebound_mod", READER)
    scopes = tuple(iter_code_scopes(mod.via.__code__))
    assert not reaches_namespace_by_name(scopes, vars(mod))
    mod.relay = mod.reader
    assert reaches_namespace_by_name(scopes, vars(mod))


def test_a_global_named_by_a_string_through_a_rebound_helper_is_keyed(tmp_path, monkeypatch, disk_cash):
    mod = _chain_module(tmp_path, monkeypatch, "chain_rebound_e2e_mod", READER)
    assert disk_cash.cache(mod.via)("x") == "x"  # one pass walks `via` while `relay` is plain
    mod.relay = mod.reader
    outer = disk_cash.cache(mod.outer)  # reaches `via`, which now reaches globals()
    assert outer() == 1
    mod.K = 2
    assert outer() == 2


@pytest.fixture(autouse=True)
def _forget_modules():
    yield
    for name in [n for n in sys.modules if n.startswith("chain_")]:
        del sys.modules[name]
