"""An object's ``__dict__`` shows what it holds, on Python 3.14.0-3.14.3 too.

There ``gc.get_referents(p.__dict__)`` gave ``[]`` for an object that keeps
its attributes inline, so the restore check read nothing in ``p`` and a loop
restored ``p`` while the notebook held a weak reference to ``p.child``:
``held() is p.child`` gave False where plain Python gives True. Cash reads
such a dict through its values (`cash._referents`).
"""

from __future__ import annotations

import weakref

from cash._referents import get_referents
from cash.notebook.statement import in_place


class Child:
    pass


class Owner:
    def __init__(self) -> None:
        self.n = 0
        self.seen: list[int] = []


def test_an_objects_dict_gives_its_values():
    p = Owner()
    p.child = Child()
    found = get_referents(p.__dict__)
    assert any(item is p.child for item in found)
    assert any(item is p.seen for item in found)


def test_several_objects_are_read_at_once():
    p, q = Owner(), Owner()
    p.child, q.child = Child(), Child()
    found = get_referents(p.__dict__, q.__dict__)
    assert any(item is p.child for item in found) and any(item is q.child for item in found)


def test_a_weak_reference_to_a_part_refuses_the_restore():
    p = Owner()
    p.child = Child()
    held = weakref.ref(p.child)
    assert in_place.refusal(p, {"p": p, "held": held}) is not None
