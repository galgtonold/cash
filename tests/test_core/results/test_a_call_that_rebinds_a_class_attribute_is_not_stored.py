"""A call that rebinds an attribute of a class is not stored: a hit would
leave the class as it was, where plain Python leaves the new binding.

``Cfg.items = Cfg.items + [x]``, ``Cfg.count += 1``, ``Cfg.items =
list(Cfg.items)`` (an equal new list: ``Cfg.items is held`` tells them
apart), ``setattr(Cfg, "items", ...)``, a classmethod's ``cls.items = ...``
and a new attribute bound on the class were stored with no more than the
IMPURE-SIDE-EFFECTS line of the first call, which a later process does not
print again. A hit in the same process, and every call of a later process,
then left the class as it was: a helper reading ``Cfg.items`` computed from
the old list, ``Cfg.count`` stayed 0, and code holding the old list saw the
class still hold it. What each user class the call can reach binds is now
compared, by identity, before and after the body
(`class_writes.ClassState`): a call that rebound one is not stored, with an
IMPURE-SCOPE-MUTATION warning. ``# @cash:assume-safe`` on the line that
writes, ``with cash.assume_safe():`` around it and ``assume_safe=True``
keep the old behaviour: stored, and a hit skips the write.

The reference is plain Python: the same script with ``cash.cache`` an
identity decorator, run in a fresh process.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

_HELPERS = """
class Cfg:
    items = [1, 2]
    count = 0

    @classmethod
    def bump(cls, x):
        cls.items = cls.items + [x]


def total():
    return sum(Cfg.items)
"""

_SCRIPT = """
import json
import sys
import warnings

import cash
import helpers
from helpers import Cfg, total

variant = sys.argv[1]
if sys.argv[2] == "plain":
    cache = lambda f: f  # noqa: E731
else:
    cache = cash.Cash(register_magic=False).cache

RUNS = []

if variant == "grow":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        Cfg.items = Cfg.items + [x]
        return total() + x
elif variant == "count":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        Cfg.count += 1
        return x
elif variant == "equal":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        Cfg.items = list(Cfg.items)
        return total() + x
elif variant == "setattr":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        setattr(Cfg, "items", Cfg.items + [x])
        return x
elif variant == "classmethod":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        Cfg.bump(x)
        return total()
elif variant == "adds":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        Cfg.seen = x
        return x
elif variant == "waived_line":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        Cfg.count += 1  # @cash:assume-safe
        return x
elif variant == "assume_safe":
    @cash.Cash(register_magic=False).cache(assume_safe=True)
    def f(x):
        RUNS.append(1)
        Cfg.count += 1
        return x
elif variant == "waived_block":
    @cache
    def f(x):
        RUNS.append(1)  # @cash:assume-safe
        with cash.assume_safe():
            Cfg.count += 1
        return x

warnings.simplefilter("ignore")
held = Cfg.items
out = [f(3), f(3)]
state = {
    "out": out,
    "items": Cfg.items,
    "count": Cfg.count,
    "seen": getattr(Cfg, "seen", None),
    "same_list": Cfg.items is held,
}
held.append(99)
state["old_list_reaches_class"] = 99 in Cfg.items
print(json.dumps({"state": state, "runs": len(RUNS)}))
"""


@pytest.fixture
def run(tmp_path):
    (tmp_path / "helpers.py").write_text(textwrap.dedent(_HELPERS), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")

    def _run(variant: str, mode: str) -> dict:
        out = run_python("main.py", variant, mode, cwd=tmp_path)
        return json.loads(out.stdout.strip().splitlines()[-1])

    return _run


@pytest.mark.parametrize("variant", ["grow", "count", "equal", "setattr", "classmethod", "adds"])
def test_every_call_leaves_the_class_as_plain_python_does(run, variant):
    plain = run(variant, "plain")["state"]
    first = run(variant, "cached")
    second = run(variant, "cached")
    assert first["state"] == plain
    assert second["state"] == plain
    if variant != "adds":  # binding the int it already holds is no change
        assert (first["runs"], second["runs"]) == (2, 2)


@pytest.mark.parametrize("variant", ["waived_line", "waived_block", "assume_safe"])
def test_a_waived_write_is_stored_and_a_hit_skips_it(run, variant):
    """Waived, the write is the user's word that a hit may skip it: the
    call is stored (the first call of a process may still run, its key
    having read the count it moved), and a later process is served it."""
    first, second = run(variant, "cached"), run(variant, "cached")
    assert first["runs"] + second["runs"] < 4
    assert second["state"]["count"] < 2


def test_the_warning_names_the_attribute(disk_cash):
    class Box:
        items = [1]

    @disk_cash.cache
    def bump(x):
        Box.items = list(Box.items)
        return x

    with pytest.warns(Warning) as caught:
        bump(1)
    said = [str(w.message) for w in caught if "IMPURE-SCOPE-MUTATION" in str(w.message)]
    assert said
    assert any("Box.items" in s for s in said)
    assert bump.explain(1).reason != "hit"


def test_a_call_that_copies_an_instance_of_a_slotted_class_is_stored(disk_cash):
    """``copy.deepcopy`` caches ``__slotnames__`` on the class the first
    time: cash's bookkeeping of a protocol, not the user's state."""
    import copy

    class Slotted:
        __slots__ = ("v",)

        def __init__(self, v):
            self.v = v

    runs = []

    @disk_cash.cache(assume_safe=False)
    def dup(x):
        runs.append(1)  # @cash:assume-safe
        return copy.deepcopy(Slotted(x)).v

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert dup(4) == 4
        assert dup(4) == 4
    assert len(runs) == 1
