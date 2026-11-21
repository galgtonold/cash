"""Building a miss's trace reads the data its class watches probe once.

A call's trace names the classes its code ran in, and each class's digest
is served under its watch only while the watch holds. A watch whose class
reaches module data no watcher sees (a list, a library's object) reads it
whole to tell. lmfit's module of line shapes holds four scipy
distributions, which each of a dozen class watches read again: about
15 ms of every miss. Building the trace is now one round of checks
(`hit_memo.probe_round`), as checking it on a hit is, and a watch made
inside a round starts from what the round read.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import class_memo, code_trace, hit_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
TABLE = [1, 2, 3]


def _make_hook(step):
    def hook(x):
        return x + step

    return hook


class Base:
    def run(self, x):
        return x


class Plain(Base):
    def run(self, x):
        return super().run(x) + TABLE[0]


class Hooked:
    hook = staticmethod(_make_hook(2))

    def run(self, x):
        return self.hook(x) * TABLE[1]
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"trace_digest_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def test_building_a_trace_reads_the_data_class_watches_probe_once(mod, tmp_path, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(left, right, x):
        ran.append(x)
        return left.run(x) + right.run(x)

    plain, hooked = mod.Plain(), mod.Hooked()
    for x in range(3):
        apply(plain, hooked, x)
    reads = []
    real_read = hit_memo._plain_content

    def counting(v):
        if v is mod.TABLE:
            reads.append(1)
        return real_read(v)

    monkeypatch.setattr(hit_memo, "_plain_content", counting)
    in_build = []
    real_build = code_trace.build

    def build(*args, **kwargs):
        before = len(reads)
        try:
            return real_build(*args, **kwargs)
        finally:
            in_build.append(len(reads) - before)

    monkeypatch.setattr(code_trace, "build", build)
    apply(plain, hooked, 10)
    assert ran == [0, 1, 2, 10]
    assert in_build, "the call was not traced"
    assert in_build[0] <= 1, "each class watch read the shared list again while the trace was built"


def test_a_change_to_the_probed_data_after_a_miss_still_reaches_the_next_call(mod, tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(left, right, x):
        ran.append(x)
        return left.run(x) + right.run(x)

    plain, hooked = mod.Plain(), mod.Hooked()
    for x in range(3):
        apply(plain, hooked, x)
    assert apply(plain, hooked, 1) == (1 + 1) + (1 + 2) * 2
    mod.TABLE[1] = 5
    assert apply(plain, hooked, 1) == (1 + 1) + (1 + 2) * 5
    assert ran == [0, 1, 2, 1]
