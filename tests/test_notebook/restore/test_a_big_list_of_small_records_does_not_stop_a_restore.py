"""A loop whose notebook keeps a big list of small records of a class
written in Python is restored into: the records are read at C speed.

E10's text-cleaning loops (mlbook ch08) ran every time because the
notebook kept the ``TarFile`` it extracted the IMDb archive with, whose
100,000 ``TarInfo`` members were too many to look through one at a time.
"""

from __future__ import annotations

import tarfile
import weakref

from cash.notebook.statement.in_place import refusal


class Child:
    def __init__(self):
        self.v = 0


class P:
    def __init__(self):
        self.n = 0
        self.seen = []
        self.child = Child()


class Record:
    def __init__(self, i, held=None):
        self.i = i
        self.name = f"r{i}"
        self.extra = {}
        self.held = held


def _members(count):
    found = []
    for i in range(count):
        info = tarfile.TarInfo(f"aclImdb/train/pos/{i}_7.txt")
        info.size = i
        found.append(info)
    return found


def test_a_big_list_of_tar_members_does_not_stop_the_restore():
    p = P()

    assert refusal(p, {"p": p, "members": _members(20000)}) is None


def test_a_big_list_of_records_of_the_users_class_does_not_stop_the_restore():
    p = P()

    assert refusal(p, {"p": p, "records": [Record(i) for i in range(20000)]}) is None


def test_a_record_keeping_an_id_of_a_part_is_found():
    p = P()
    records = [Record(i) for i in range(20000)]
    records[12345].held = id(p.seen)

    assert "id()" in refusal(p, {"p": p, "records": records})


def test_an_id_kept_as_text_deeper_in_a_record_is_found():
    p = P()
    records = [Record(i) for i in range(20000)]
    records[777].extra["key"] = hex(id(p.child))

    assert "id()" in refusal(p, {"p": p, "records": records})


def test_a_record_keeping_a_weak_reference_to_a_part_is_found():
    p = P()
    records = [Record(i) for i in range(20000)]
    records[5].held = weakref.ref(p.child)

    assert refusal(p, {"p": p, "records": records}) is not None


def test_a_big_list_of_functions_is_still_too_big_to_look_through():
    p = P()
    namespace = {"p": p, "big": [lambda j=j: j for j in range(20000)]}

    assert "too big to look through ('big')" in refusal(p, namespace)
