"""What a class folds to is remembered in the process while its world
provably stands still (`cash.decorator.class_memo`), and folded again the
moment anything it holds or reaches moves.

Two folds read the same class on every call that meets it: the code an
argument's class carries (a sampler handed to every call cost 8-20 ms a
call, hits included) and the digest of a class a call's trace names. Each
edit below must still reach the key; the sabotaged arm, whose watch never
breaks, shows the edits are seen by the watch and not by luck.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import class_memo
from cash.tracking.trace_items import remembered_class_digest as class_digest

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


class Sampler:
    constraints = {"k": [Bound(0)]}

    def __init__(self, k):
        self.k = k

    def run(self, x):
        return scale(x) + self.k + self.constraints["k"][0].low
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"class_memo_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def _edits(mod):
    def rebind_global():
        mod.RATE = 3

    def replace_method_code():
        def run(self, x):
            return scale(x) - self.k  # noqa: F821 - compiled for its code only

        mod.Sampler.run.__code__ = run.__code__

    def edit_helper_code():
        def scale(x):
            return x * RATE + 1  # noqa: F821 - compiled for its code only

        mod.scale.__code__ = scale.__code__

    def mutate_held_instance():
        mod.Sampler.constraints["k"][0].low = 5

    return [rebind_global, replace_method_code, edit_helper_code, mutate_held_instance]


@pytest.mark.parametrize("edit", range(4), ids=["global", "method", "helper", "held-instance"])
def test_an_argument_class_is_folded_again_after_an_edit(tmp_path, mod, edit):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(sampler, x):
        ran.append(x)
        return sampler.run(x)

    s = mod.Sampler(1)
    first = apply(s, 10)
    # The first miss moves what cash learned; the hit after it watches anew.
    apply(s, 10)
    before = class_memo.STATS["served"]
    assert apply(s, 10) == first
    assert class_memo.STATS["served"] > before, "the argument's class was folded again"
    assert len(ran) == 1
    key = apply.explain(s, 10).cache_key
    _edits(mod)[edit]()
    assert apply.explain(s, 10).cache_key != key, "an edit the argument's class reaches did not reach the key"
    second = apply(s, 10)
    assert len(ran) == 2, "an edit the argument's class reaches did not reach the key"
    assert second == mod.Sampler.run(s, 10)


def test_a_sabotaged_watch_would_serve_the_stale_fold(tmp_path, mod, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(sampler, x):
        ran.append(x)
        return sampler.run(x)

    s = mod.Sampler(1)
    for _ in range(3):
        apply(s, 10)
    # A watch that never breaks: what only the class's fold reads (a global
    # its method reads, data its class holds) is no longer seen, so what the
    # tests above see is the watch. (Its methods' own code is keyed apart.)
    key = apply.explain(s, 10).cache_key
    monkeypatch.setattr(class_memo._Proof, "holds", lambda self: self.world >= 0)
    rebind_global, _method, _helper, mutate_held_instance = _edits(mod)
    rebind_global()
    mutate_held_instance()
    assert apply.explain(s, 10).cache_key == key
    monkeypatch.undo()
    assert apply.explain(s, 10).cache_key != key


def test_a_remembered_fold_is_served_into_a_build_that_folded_nothing_of_it(tmp_path, mod):
    """A fold that found the class folded already by the build (its own
    globals read the class) is not remembered for a build that did not."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    Sampler = mod.Sampler

    @c.cache
    def reads_the_class_too(sampler, x):
        return Sampler.run(sampler, x)

    @c.cache
    def only_the_argument(sampler, x):
        return sampler.run(x)

    s = mod.Sampler(1)
    keys = [only_the_argument.explain(s, 1).cache_key]
    reads_the_class_too(s, 1)
    reads_the_class_too(s, 1)
    keys.append(only_the_argument.explain(s, 1).cache_key)
    assert keys[0] == keys[1]


@pytest.mark.parametrize("edit", range(4), ids=["global", "method", "helper", "held-instance"])
def test_a_class_digest_moves_with_an_edit(mod, edit):
    before = class_digest(mod.Sampler)
    assert class_digest(mod.Sampler) == before
    _edits(mod)[edit]()
    if edit in (0, 2):
        # What a method's code reads is no part of the class's digest.
        assert class_digest(mod.Sampler) == before
    else:
        assert class_digest(mod.Sampler) != before


def test_a_class_digest_is_served_and_a_sabotaged_watch_would_keep_it(mod, monkeypatch):
    class_digest(mod.Sampler)
    before = class_memo.STATS["served"]
    stale = class_digest(mod.Sampler)
    assert class_memo.STATS["served"] > before
    monkeypatch.setattr(class_memo._Proof, "holds", lambda self: self.world >= 0)
    mod.Sampler.constraints["k"][0].low = 5
    assert class_digest(mod.Sampler) == stale
    monkeypatch.undo()
    assert class_digest(mod.Sampler) != stale


def test_a_class_defined_anew_lets_the_remembered_one_go(tmp_path):
    """A notebook cell run again defines its class anew: the memo must not
    keep the old one alive, which ``__subclasses__`` would still list."""
    import gc
    import weakref

    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    class Base:
        pass

    @c.cache
    def apply(step, x):
        return step.out(x)

    ns = {"Base": Base, "__name__": __name__}
    olds = []
    served = class_memo.STATS["served"]
    for k in (2, 3):
        exec(f"class Step(Base):\n    def out(self, x):\n        return x * {k}\n", ns)
        olds.append(weakref.ref(ns["Step"]))
        for _ in range(3):
            assert apply(ns["Step"](), 1) == k
        class_digest(ns["Step"])
        class_digest(ns["Step"])
    assert class_memo.STATS["served"] > served
    gc.collect()
    assert olds[0]() is None
    assert Base.__subclasses__() == [ns["Step"]]
