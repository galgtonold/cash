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
