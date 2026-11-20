"""A class whose world reaches a cached function is remembered too, while
its fold asks for no cached function's state; one whose fold does is
folded every time.

A method of the user's own class, cached, takes the instance as ``self``:
the class's world reaches the method's wrapper, and the class memo refused
it outright, so every hit folded the class and the user code it reaches
again (an lmfit study, 27 ms of a 55 ms hit on a 30 ms fit). The fold of
the class keys the wrapper by the code it wraps, which the watch covers;
only a fold that keys a cached function by its whole state
(`KeyBuilder.callee_state`), which the watch does not cover, is not
remembered. Code that reads ``sys.modules`` no longer refuses the watch
either: walked, it was every module loaded, and one of a class of its own
(``six.moves``) made the whole world unwatchable.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import types

import pytest

from cash import Cash
from cash.decorator import class_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
import sys

RATE = 2
LEVEL = 1


def scale(x):
    return x * RATE


def level(x):
    return x + LEVEL


class Study:
    def __init__(self, k):
        self.k = k

    def fit(self, x):
        return scale(x) + self.k

    def loaded(self, prefix):
        return sorted(name for name in sys.modules if name.startswith(prefix))


class Caller:
    def run(self, x):
        return leveled(x) * 2
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"cached_world_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.fixture
def study(mod, cash):
    """``Study.fit`` cached, as ``@cash.cache`` on the method would leave it."""
    mod.Study.fit = cash.cache(mod.Study.fit)
    return mod.Study(3)


class _LazyModule(types.ModuleType):
    """A module of a class of its own, as ``six.moves`` is."""


@pytest.fixture
def lazy_module(tmp_path):
    name = f"lazy_{tmp_path.name.replace('-', '_')}"
    sys.modules[name] = _LazyModule(name)
    yield name
    sys.modules.pop(name, None)


def _fit_edited(self, x):
    return scale(x) - self.k  # noqa: F821 - compiled for its code only


def _loaded_edited(self, prefix):
    return []


def _served(cls, purpose):
    proof = class_memo._proofs.get((id(cls), purpose))
    return 0 if proof is None else proof.served


def _folded_afresh(monkeypatch, fn, *args):
    with monkeypatch.context() as m:
        m.setattr(class_memo, "ENABLED", False)
        return fn.explain(*args).cache_key


def test_the_class_of_a_cached_methods_instance_is_served(mod, study, lazy_module):
    for _ in range(3):
        study.fit(10)
    before = (_served(mod.Study, "carrier"), _served(mod.Study, "surface"))
    assert study.fit(10) == 23
    assert _served(mod.Study, "carrier") > before[0], "the class's fold was folded again"
    assert _served(mod.Study, "surface") > before[1], "the code the class reaches was walked again"


def test_it_keys_as_folding_again(mod, study, lazy_module, monkeypatch):
    for x in (1, 2, 1, 3, 2):
        study.fit(x)
        assert mod.Study.fit.explain(study, x).cache_key == _folded_afresh(monkeypatch, mod.Study.fit, study, x)


@pytest.mark.parametrize("edit", ["global", "helper", "method", "sibling-method"])
def test_an_edit_the_class_reaches_still_reaches_the_key(mod, study, lazy_module, edit):
    for _ in range(3):
        study.fit(10)
    key = mod.Study.fit.explain(study, 10).cache_key
    if edit == "global":
        mod.RATE = 3
    elif edit == "helper":

        def scale(x):
            return x * RATE + 1  # noqa: F821 - compiled for its code only

        mod.scale.__code__ = scale.__code__
    elif edit == "method":
        mod.Study.fit.__wrapped__.__code__ = _fit_edited.__code__
    else:
        mod.Study.loaded.__code__ = _loaded_edited.__code__
    assert mod.Study.fit.explain(study, 10).cache_key != key
    assert study.fit(10) == mod.Study.fit.__wrapped__(study, 10)


def test_a_fold_that_keys_a_cached_functions_state_is_not_remembered(mod, cash, monkeypatch):
    mod.leveled = cash.cache(mod.level)
    ran = []

    @cash.cache
    def apply(caller, x):
        ran.append(x)
        return caller.run(x)

    caller = mod.Caller()
    for _ in range(3):
        apply(caller, 1)
    # The code the class reaches folds the cached function's state; what
    # its methods read does not.
    assert _served(mod.Caller, "surface") == 0, "a fold holding a cached function's state was remembered"
    assert _served(mod.Caller, "carrier") > 0
    key = apply.explain(caller, 1).cache_key
    assert key == _folded_afresh(monkeypatch, apply, caller, 1)
    mod.LEVEL = 5
    assert apply.explain(caller, 1).cache_key != key
    assert apply(caller, 1) == 12
    assert ran == [1, 1]
