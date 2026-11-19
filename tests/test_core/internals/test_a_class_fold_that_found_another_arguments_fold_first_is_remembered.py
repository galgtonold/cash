"""A class fold that found what another argument's fold went through first
is remembered, for builds that find the same.

Two arguments whose classes share a base and the helpers it calls (an
lmfit study holding a composite model of Gaussian and linear models): the
second class's fold finds the base and the helpers' globals folded already
and leaves them out. It was never remembered, so every call folded it
again: 36 ms of a 74 ms hit on an lmfit fit that takes 27 ms. It is now
served to a build that answers its questions alike, and folded again for
one that does not, and every edit still reaches the key.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import class_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
RATE = 2


class Bound:
    def __init__(self, low):
        self.low = low


def scale(x):
    return x * RATE


class Base:
    limits = {"k": [Bound(0)]}

    def shift(self, x):
        return scale(x) + self.limits["k"][0].low


class First(Base):
    def run(self, x):
        return self.shift(x) + 1


class Second(Base):
    def run(self, x):
        return self.shift(x) * 2


class Other:
    def run(self, x):
        return x
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"shared_fold_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def apply(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(left, right, x):
        ran.append(x)
        return left.run(x) + right.run(x)

    apply.ran = ran
    return apply


def _served(cls):
    proof = class_memo._proofs.get((id(cls), "carrier"))
    return 0 if proof is None else proof.served


def _folded_afresh(monkeypatch, fn, *args):
    with monkeypatch.context() as m:
        m.setattr(class_memo, "ENABLED", False)
        return fn.explain(*args).cache_key


def test_the_second_arguments_fold_is_served(mod, apply):
    first, second = mod.First(), mod.Second()
    for _ in range(3):
        apply(first, second, 10)
    before = _served(mod.Second)
    assert apply(first, second, 10) == first.run(10) + second.run(10)
    assert _served(mod.Second) > before, "the fold that found the first argument's was folded again"
    assert len(apply.ran) == 1


def test_it_keys_as_folding_again_whatever_was_folded_before_it(mod, apply, monkeypatch):
    first, second, other = mod.First(), mod.Second(), mod.Other()
    calls = [(first, second, 1), (other, second, 1), (first, second, 1), (second, first, 1), (other, second, 1)]
    for _ in range(2):
        for args in calls:
            apply(*args)
            assert apply.explain(*args).cache_key == _folded_afresh(monkeypatch, apply, *args)


@pytest.mark.parametrize("edit", ["global", "helper", "held-instance"])
def test_an_edit_the_shared_base_reaches_still_reaches_the_key(mod, apply, edit):
    first, second = mod.First(), mod.Second()
    for _ in range(3):
        apply(first, second, 10)
    key = apply.explain(first, second, 10).cache_key
    if edit == "global":
        mod.RATE = 3
    elif edit == "helper":

        def scale(x):
            return x * RATE + 1  # noqa: F821 - compiled for its code only

        mod.scale.__code__ = scale.__code__
    else:
        mod.Base.limits["k"][0].low = 5
    assert apply.explain(first, second, 10).cache_key != key
    assert apply(first, second, 10) == first.run(10) + second.run(10)
    assert len(apply.ran) == 2
