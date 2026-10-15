"""A chain of objects whose class chooses its state (``__getstate__``) is
keyed in work linear in its depth, to the digest it always had.

The key of such an object keeps the attributes its state leaves out
(`kept_state.left_out_attrs`), and telling whether one can be pickled
pickles it. An attribute that is itself such an object had its own left-out
attributes pickled to tell, and was pickled again as part of the state: a
chain was pickled 2**depth times (16 deep, a second; 30 deep, hours).
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from cash import kept_state
from cash.value_hash import compute_hash

SOURCE = (
    "class Chosen:\n"
    "    def __getstate__(self):\n"
    "        return {'k': 1}\n"
    "class Slotted:\n"
    "    __slots__ = ('x', 'y')\n"
    "    def __getstate__(self):\n"
    "        return (None, {'x': 0})\n"
)


@pytest.fixture
def classes(monkeypatch):
    name = "_chosen_state_chain"
    module = types.ModuleType(name)
    module.__file__ = None
    exec(SOURCE, module.__dict__)
    for cls in (module.Chosen, module.Slotted):
        cls.__module__ = name
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _chain(module, depth: int, mixed: bool = False):
    node = root = module.Chosen()
    for i in range(depth):
        child = module.Slotted() if mixed and i % 3 == 0 else module.Chosen()
        if isinstance(child, module.Slotted):
            child.x = i
        if isinstance(node, module.Slotted):
            node.y = child
        else:
            node.child = child
        node = child
    node.lock = threading.Lock()  # left out, and no pickle can read it
    return root


class _TooMuchWork(BaseException):
    """Stops a keying past its bound: what catches ``Exception`` lets it by."""


def _counting_dumps(monkeypatch, bound: int):
    calls = [0]
    real = kept_state._dumps

    def counting(*args, **kwargs):
        calls[0] += 1
        if calls[0] > bound:
            raise _TooMuchWork
        return real(*args, **kwargs)

    monkeypatch.setattr(kept_state, "_dumps", counting)
    return calls


@pytest.mark.parametrize("depth", [20, 200])
def test_the_work_grows_with_the_depth_not_its_power(classes, monkeypatch, depth):
    bound = 2 * depth + 10
    calls = _counting_dumps(monkeypatch, bound)
    start = time.perf_counter()
    try:
        compute_hash(_chain(classes, depth))
    except _TooMuchWork:
        pytest.fail(f"more than {bound} pickles to key a chain {depth} deep")
    assert calls[0] > depth, "each object's left-out attributes are told"
    assert time.perf_counter() - start < 5


@pytest.mark.parametrize("mixed", [False, True], ids=["one_class", "with_slots"])
def test_the_digest_is_the_one_every_attribute_told_alone_gives(classes, monkeypatch, mixed):
    """Control: told with a pickle of its own each and nothing remembered,
    as before, every left-out attribute gives the same digest."""
    value = _chain(classes, 9, mixed)
    remembered = compute_hash(value)

    def alone(attr):
        try:
            kept_state._dumps(attr, kept_state.pickle.DEFAULT_PROTOCOL, False, None, None, None, True)
        except Exception:
            return False
        return True

    monkeypatch.setattr(kept_state, "_picklable", alone, raising=False)
    monkeypatch.setattr(kept_state, "_LEFT_OUT", _Forgetful(), raising=False)

    assert compute_hash(value) == remembered


class _Forgetful(list):
    """A memo stack whose memo remembers nothing."""

    def append(self, _memo):
        super().append(_Nothing())


class _Nothing(dict):
    def __setitem__(self, key, value):
        pass
