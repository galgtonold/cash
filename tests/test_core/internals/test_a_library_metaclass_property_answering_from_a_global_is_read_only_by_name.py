"""A library class whose metaclass has a property answering from a module
global (bokeh's ``MetaHasProps.model_class_reverse_map``) no longer keeps
the state segment of every function whose world reaches it from being
remembered.

A key reads a library class's attributes from ``vars()`` along its MRO,
never through its metaclass, so such a property runs only when the cached
code reads it by name. The state memo's scan (`cash.decorator.hit_memo`)
notes the property's name and refuses only code that names it; code that
does not is reused, and edits elsewhere still move its segment. A user
metaclass's property is refused as before.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

LIBRARY = textwrap.dedent("""
    _REVERSE = {"a": 1}


    class MetaHasProps(type):
        @property
        def reverse_map(cls):
            return dict(_REVERSE)


    class Model(metaclass=MetaHasProps):
        scale = 2


    def register(name, value):
        _REVERSE[name] = value
""")

USER = textwrap.dedent("""
    from {lib} import Model

    OFFSET = 10


    class UserMeta(type):
        @property
        def reverse_map(cls):
            return dict(_USER_REVERSE)


    _USER_REVERSE = {{"a": 1}}


    class UserModel(metaclass=UserMeta):
        scale = 2


    def reads_property(x):
        return x * Model.scale + len(Model.reverse_map)


    def holds_class(x, model=Model):
        return x * model.scale + OFFSET


    def reads_user_property(x):
        return x + len(UserModel.reverse_map)
""")


@pytest.fixture
def mod(tmp_path, monkeypatch):
    tag = tmp_path.name.replace("-", "_")
    lib = f"metalib_{tag}"
    site = tmp_path / "site-packages"
    (site / lib).mkdir(parents=True)
    (site / lib / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    name = f"metauser_{tag}"
    (tmp_path / f"{name}.py").write_text(USER.format(lib=lib), encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name)
    sys.modules.pop(name, None)
    sys.modules.pop(lib, None)


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _state(c, f):
    """The segment through the memo, and a fresh fold of it."""
    spec = c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]
    return c._keys._code_state(spec, [], note=True), c._keys._code_state_now(spec, [], note=True)


def _reuses(c, f, times=4):
    """How many of *times* asks for *f*'s segment were served remembered;
    each one agrees with a fresh fold."""
    before = state_memo.STATS["reused"]
    for _ in range(times):
        got, fresh = _state(c, f)
        assert got == fresh
    return state_memo.STATS["reused"] - before


def test_a_library_metaclass_property_read_by_name_is_not_reused(mod, c):
    f = c.cache(mod.reads_property)
    assert f(3) == f.__wrapped__(3)
    assert _reuses(c, f) == 0
    sys.modules[mod.Model.__module__].register("b", 2)
    assert f.__wrapped__(3) == 8
    assert _reuses(c, f) == 0


def test_a_library_metaclass_property_not_read_is_reused(mod, c, monkeypatch):
    monkeypatch.setattr(state_memo, "VERIFY", True)
    f = c.cache(mod.holds_class)
    assert f(3) == f.__wrapped__(3) == 16
    assert _reuses(c, f) > 0, "the segment is not being reused: the test proves nothing"
    old, _ = _state(c, f)
    assert state_memo.STATS["verified"] > 0
    # An edit elsewhere still moves the segment, and the call misses.
    mod.OFFSET = 20
    got, fresh = _state(c, f)
    assert got == fresh != old
    assert f(3) == f.__wrapped__(3) == 26


def test_a_user_metaclass_property_is_not_reused(mod, c):
    f = c.cache(mod.reads_user_property)
    assert f(3) == f.__wrapped__(3) == 4
    assert _reuses(c, f) == 0
    mod._USER_REVERSE["b"] = 2
    assert f(3) == f.__wrapped__(3) == 5
