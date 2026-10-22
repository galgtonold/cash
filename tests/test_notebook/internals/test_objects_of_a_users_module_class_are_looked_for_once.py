"""Whether an object of a class of the user's modules exists is asked by
scanning every object the collector tracks (``gc.get_referrers``), about
17 ms per million objects. A statement's keys asked it many times: late in
a long test run, with a big heap on a loaded machine, one statement took
over 30 s. A yes is kept until code runs from an ``exec`` again: kept
stale (the object collected, a class id reused), it can only add to a key,
since it lets a walk run, which finds what is there. A no is asked afresh.
"""

from __future__ import annotations

import gc

from cash.notebook import held_functions


class _OfAUsersModule:
    pass


def test_objects_of_a_users_module_class_are_looked_for_once_until_code_runs(monkeypatch):
    # Each statement's keys asked this many times, and each asked was a scan
    # of every object the collector tracks (~17 ms per million): late in a
    # long test run one statement took over 30 s. A yes makes a walk run,
    # which finds what is there, so it is kept until code runs again.
    scans = []
    real = gc.get_referrers

    def counting(*objects):
        scans.append(len(objects))
        return real(*objects)

    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [_OfAUsersModule])
    monkeypatch.setattr(held_functions.gc, "get_referrers", counting)
    kept = _OfAUsersModule()
    assert all(held_functions.module_class_objects_exist() for _ in range(5))
    assert len(scans) == 1, scans
    exec("pass", {})  # the next statement
    assert held_functions.module_class_objects_exist()
    assert len(scans) == 2, scans
    del kept


def test_no_object_is_looked_for_again_each_time(monkeypatch):
    # A no is not kept: running code can make an object with no new code.
    scans = []
    real = gc.get_referrers

    def counting(*objects):
        scans.append(len(objects))
        return real(*objects)

    class Fresh:
        pass

    monkeypatch.setattr(held_functions, "_users_module_classes", lambda: [Fresh])
    monkeypatch.setattr(held_functions.gc, "get_referrers", counting)
    assert not held_functions.module_class_objects_exist()
    made = Fresh()
    assert held_functions.module_class_objects_exist()
    assert len(scans) == 2, scans
    del made
