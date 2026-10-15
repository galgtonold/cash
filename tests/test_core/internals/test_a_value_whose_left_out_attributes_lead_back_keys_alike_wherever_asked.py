"""A value whose left-out attributes lead back to the object leaving them
out keys to one digest, however deep the stack is when it is keyed and
whatever was keyed before it in the same keying.

``sd`` leaves out its slots (its ``__getstate__`` saves none), and one of
them holds an object holding ``sd``: telling whether that one pickles
pickled ``sd`` again, and told its attributes again, without end. With
each object's answer remembered for the keying (`kept_state.one_keying`),
the answers taken where the stack ran out were remembered too, so the
digest depended on how many frames sat above the call (three digests in
the watcher fuzz, seed 217).
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from cash import kept_state
from cash.value_hash import compute_hash

SOURCE = (
    "class A:\n"
    "    def __init__(self, **kw):\n"
    "        self.__dict__.update(kw)\n"
    "class S:\n"
    "    __slots__ = ('a', 'b', '__weakref__')\n"
    "    def __getstate__(self):\n"
    "        return (None, {})\n"
    "class SD(S):\n"
    "    __slots__ = ('c', '__dict__')\n"
    "def g(x):\n"
    "    return x\n"
)


@pytest.fixture
def value(monkeypatch):
    name = "_left_out_leads_back"
    module = types.ModuleType(name)
    module.__file__ = None
    exec(SOURCE, module.__dict__)
    for cls in (module.A, module.S, module.SD):
        cls.__module__ = name
    monkeypatch.setitem(sys.modules, name, module)
    sd, s = module.SD(), module.S()
    sd.a, sd.b, sd.c, sd.extra = True, s, module.A(back=sd, fn=module.g), "x"
    s.a, s.b = 1, sd
    module.g.__qualname__ = "renamed"  # pickled by a name that no longer finds it
    return [{"k": [sd]}, [], [module.A, sd, s], "x"], s


def _deeper(frames: int, value):
    return _deeper(frames - 1, value) if frames else compute_hash(value)


def test_the_digest_does_not_depend_on_the_frames_above(value):
    root, _s = value
    assert len({_deeper(frames, root) for frames in range(60)}) == 1


def test_what_was_keyed_before_in_the_keying_does_not_change_it(value):
    root, s = value
    alone = compute_hash(root)

    afters = set()
    for frames in range(0, 60, 7):
        with kept_state.one_keying():
            _deeper(frames, [s])  # tells ``s`` and what it leads to first
            afters.add(compute_hash(root))

    assert afters == {alone}


def test_the_digest_is_the_one_nothing_remembered_gives(value, monkeypatch):
    """Control: with nothing remembered but which objects are being told
    (without which the telling has no end), the digest is the same."""
    root, _s = value
    remembered = compute_hash(root)
    monkeypatch.setattr(kept_state, "_Keying", _Forgetful, raising=False)

    assert compute_hash(root) == remembered


def test_a_keying_is_its_own_threads():
    seen = []
    with kept_state.one_keying():
        thread = threading.Thread(target=lambda: seen.append(kept_state._keying()))
        thread.start()
        thread.join()
        assert kept_state._keying() is not None
    assert seen == [None]


class _Forgetful:
    def __init__(self):
        self.left_out = _OnlyTelling()
        self.picklable = _OnlyTelling()


class _OnlyTelling(dict):
    def __setitem__(self, key, value):
        if value[1] is getattr(kept_state, "_TELLING", None):
            super().__setitem__(key, value)
        else:
            self.pop(key, None)
