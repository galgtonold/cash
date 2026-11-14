"""A set a hit restores iterates in the order the run that stored it did.

A set iterates in its hash table's slot order, and a set rebuilt from its
elements -- unpickled, or copied by the RAM tier -- put many of them in other
slots. ``codes = {c: i for i, c in enumerate(kinds)}``, a label encoding,
numbered the categories differently after a restart than in the run that
stored ``kinds``, and so did everything built from the codes. The store keeps
where each set's elements sit, and the hit puts them back (`set_order`): sets
in plain containers, and sets an object or a container's subclass holds -- a
defaultdict(set), an OrderedDict, a namedtuple, a dataclass, ``__slots__``.
"""

from __future__ import annotations

import pytest

from cash.backends import set_order
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = pytest.mark.skipif(set_order._KERNELS is None, reason="no C extension for this Python")

MAKE = f"""import random, time
def make_kinds():
    r = random.Random(1)
    kinds = set(r.sample(range(1000), 100))
    groups = [set(r.sample(range(100000), 50)) for _ in range(5)]
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return kinds, groups"""
LOAD = "kinds, groups = make_kinds()"
ENCODE = "codes = {c: i for i, c in enumerate(kinds)}\ngroup_of = {n: i for i, g in enumerate(groups) for n in g}\nshown = repr(kinds)"


def _plain():
    ns: dict = {}
    for cell in (MAKE, LOAD, ENCODE):
        exec(cell, ns)
    return ns


def _restart(magics) -> None:
    magics.tracking_state.reset_session_state()
    simulator = magics._upstream_checker.simulator
    simulator.cache.reset()
    simulator.probe.reset()
    ns = magics.shell.user_ns
    for name in [n for n in ns if not n.startswith("_") and n not in ("get_ipython", "exit", "quit")]:
        ns.pop(name, None)


def test_a_label_encoding_of_a_restored_set_numbers_as_the_run_did(cash_magics):
    plain = _plain()
    cells = [MAKE, LOAD, ENCODE]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    _restart(cash_magics)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["kinds"] is not None and ns["kinds"] == plain["kinds"]
    assert ns["shown"] == plain["shown"]
    assert list(ns["codes"].items()) == list(plain["codes"].items())
    assert list(ns["group_of"].items()) == list(plain["group_of"].items())


HOLDERS = f"""import collections, dataclasses, random, time
Pair = collections.namedtuple("Pair", "name members")
class SetOf(set):
    pass
class Plain:
    def __init__(self, members):
        self.members = members
@dataclasses.dataclass
class Data:
    name: str
    members: set
class Slotted:
    __slots__ = ("members",)
    def __init__(self, members):
        self.members = members
def colliding(seed):
    r = random.Random(seed)
    s = set(r.sample(range(4000), 600))
    for n in r.sample(sorted(s), 200):
        s.discard(n)
    s.update(r.sample(range(4000, 8000), 100))
    return s
def make_holders():
    by_group = collections.defaultdict(set)
    by_group["a"] |= colliding(1)
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return dict(
        subclass=SetOf(colliding(2)),
        by_group=by_group,
        ordered=collections.OrderedDict(x=colliding(3)),
        pair=Pair("p", colliding(4)),
        plain=Plain(colliding(5)),
        data=Data("d", colliding(6)),
        slotted=Slotted(colliding(7)),
        two_deep=[Plain(Data("inner", colliding(8)))],
    )"""
LOAD_HOLDERS = "holders = make_holders()"
NUMBER = """def sets_of(h):
    return [h["subclass"], h["by_group"]["a"], h["ordered"]["x"], h["pair"].members, h["plain"].members,
            h["data"].members, h["slotted"].members, h["two_deep"][0].members.members]
numbers = [{n: i for i, n in enumerate(s)} for s in sets_of(holders)]"""


def _numbers(ns):
    return [list(d.items()) for d in ns["numbers"]]


def test_sets_held_by_objects_and_subclasses_number_as_the_run_did(cash_magics):
    cells = [HOLDERS, LOAD_HOLDERS, NUMBER]
    plain: dict = {}
    for cell in cells:
        exec(cell, plain)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    _restart(cash_magics)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["holders"]["plain"].members == plain["holders"]["plain"].members
    assert _numbers(ns) == _numbers(plain)
