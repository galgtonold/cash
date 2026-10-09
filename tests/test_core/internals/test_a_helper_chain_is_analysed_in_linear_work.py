"""A chain of helpers costs analysis work in proportion to its length.

Each helper's "can a string here name a global?" walk followed every helper
it reaches, reading each one's bytecode again, and checking whether a
helper's loaded code still matched its file walked every code object of the
recompiled module once per helper: a chain of 300 helpers took seconds
before the first call in every new process. What a code object names is now
read once, a walk that finds nothing is remembered for the rest of one
analysis pass, and the recompiled module's functions are looked up by name.

The memo must never hide a helper: an edited helper still reads as edited,
and a helper rebound after one pass is followed as it is in the next
(``test_core/code_identity/test_a_helper_rebound_later_is_followed.py``).
"""

from __future__ import annotations

import dis
import importlib
import os
import sys
import textwrap

import pytest

import cash.decorator.global_reads as global_reads_module
import cash.loaded_code as loaded_code_module

CHAIN = 80


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


def test_the_reach_walk_reads_each_helper_s_bytecode_about_once(tmp_path, monkeypatch, disk_cash):
    mod = _chain_module(tmp_path, monkeypatch, "chain_reach_mod")
    reads = []
    real = dis.get_instructions

    def counting(code, *args, **kwargs):
        reads.append(code)
        return real(code, *args, **kwargs)

    monkeypatch.setattr(global_reads_module.dis, "get_instructions", counting)
    cached = disk_cash.cache(mod.top)
    assert cached(1) == CHAIN
    # Before: each helper's walk re-read every helper below it, about
    # CHAIN**2 / 2 = 3200 reads here. A few reads per helper are the
    # function's own folds (written and loaded globals, attribute pairs).
    assert len(reads) <= 8 * CHAIN, len(reads)


def test_loaded_code_is_matched_by_name_not_by_walking_the_module(tmp_path, monkeypatch, disk_cash):
    mod = _chain_module(tmp_path, monkeypatch, "chain_loaded_mod")
    walks = []
    real = loaded_code_module._code_objects

    def counting(code):
        if isinstance(code.co_consts, tuple) and code.co_name == "<module>":
            walks.append(code)  # one walk of a whole module
        yield from real(code)

    monkeypatch.setattr(loaded_code_module, "_code_objects", counting)
    for i in range(CHAIN):
        assert loaded_code_module.loaded_code_matches_disk(getattr(mod, f"h{i}"))
    # One walk of the module per file version, not one per helper.
    assert len(walks) == 1, len(walks)


@pytest.fixture(autouse=True)
def _forget_modules():
    yield
    for name in [n for n in sys.modules if n.startswith("chain_")]:
        del sys.modules[name]
