"""A class of the user's whose metaclass has a ``__getattribute__`` of its
own, alive anywhere in the process, is not asked anything by cash while a
statement runs.

Two of cash's walks over every class asked each one through Python's
attribute lookup: the walk for memos in the user's modules (``vars(cls)``)
and the walk for what ABCs registered (``isinstance(k, abc.ABCMeta)``,
which asks ``k.__class__``). Each ran the metaclass's hook, where plain
Python never does, so the hook's write (here ``RAN.append``) changed the
user's state, and inside the statement it was taken as a write of the
statement's that no stored result repeats: ``G = build()`` was never
served. It needed only such a class alive in a module of the user's and
the ABC registries to have moved since cash last read them.
"""

from __future__ import annotations

import abc
import dataclasses
import uuid

from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

RAN: list[str] = []


class Watching(type):
    def __getattribute__(cls, name):
        RAN.append(name)
        return super().__getattribute__(name)


class Kept(metaclass=Watching):
    pass


SETUP = f"""
import time

class Graph:
    def __init__(self):
        self.rows = list(range(1000))

def build():
    # {uuid.uuid4().hex}: a text this process has not looked up before
    time.sleep(0.2)
    return Graph()
"""


def _move_the_abc_registries():
    class Some(abc.ABC):
        pass

    Some.register(type("Registered", (), {}))


def test_a_statement_with_such_a_class_alive_is_served_and_the_hook_never_runs(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    assert type(Kept) is Watching
    RAN.clear()
    run_cash_cell(cash_magics, SETUP)
    _move_the_abc_registries()
    run_cash_cell(cash_magics, "G = build()")
    _move_the_abc_registries()
    run_cash_cell(cash_magics, "G = build()")
    assert RAN == []
    assert cash_magics._last_cell_metrics["statements"][-1]["status"] == CacheStatus.RESTORED


class Counted(metaclass=Watching):
    """A class of the user's whose metaclass hooks every lookup on it."""

    step = 2

    def __init__(self):
        self.base = 40

    def total(self):
        return self.base + self.step


@dataclasses.dataclass
class Point(metaclass=Watching):
    x: int = 1
    y: int = 2


def _served_twice(cash_magics, statement):
    """*statement* run, then run again and served, with nothing of the
    user's asking the class anything: only an instance's lookups run."""
    RAN.clear()
    run_cash_cell(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert cash_magics._last_cell_metrics["statements"][-1]["status"] == CacheStatus.RESTORED
    return list(RAN)


def test_a_function_using_such_a_class_of_a_module_is_keyed_without_asking_it(cash_magics, monkeypatch):
    # The key of `use()` walks what `use` reaches: the class, its methods,
    # its namespace, its module; the store sizes and digests the result.
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cash_magics.shell.user_ns["Counted"] = Counted
    run_cash_cell(cash_magics, f"def use():\n    # {uuid.uuid4().hex}\n    return Counted().total()")
    assert _served_twice(cash_magics, "n = use()") == []
    assert cash_magics.shell.user_ns["n"] == 42


def test_a_class_a_cell_defines_with_such_a_metaclass_is_keyed_without_asking_it(cash_magics, monkeypatch):
    # The class is the notebook's own: its code is part of the cell's
    # lineage, read off the class as `type` reads it.
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cash_magics.shell.user_ns["Watching"] = Watching
    run_cash_cell(
        cash_magics,
        f"class Local(metaclass=Watching):\n    # {uuid.uuid4().hex}\n    step = 3\n"
        "    def total(self):\n        return 39 + self.step",
    )
    run_cash_cell(cash_magics, "def use_local():\n    return Local().total()")
    assert _served_twice(cash_magics, "m = use_local()") == []
    assert cash_magics.shell.user_ns["m"] == 42


def test_a_dataclass_with_such_a_metaclass_is_keyed_without_asking_it(cash_magics, monkeypatch):
    # Its fields (`__dataclass_fields__`) and annotations are read as
    # `type` reads them.
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cash_magics.shell.user_ns["Point"] = Point
    run_cash_cell(
        cash_magics, f"def norm():\n    # {uuid.uuid4().hex}\n    p = Point(3, 4)\n    return p.x * p.x + p.y * p.y"
    )
    assert _served_twice(cash_magics, "d = norm()") == []
    assert cash_magics.shell.user_ns["d"] == 25
