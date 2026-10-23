"""A miss is not stored for a list another thread changed before its body.

After a miss, the check for a change made by the body skips its second
hash of a list when the list is shown unchanged: by its identities for a
list of plain values, by its watchers' token for a list of records. Both
were taken as the body began, after the key was built. Another thread
that changed the list and called in between made them stand for its
content, so the body ran on that content and its result was stored
under the key of the content before. Both are now taken by the key
build, the identities before its walk, so a write after them shows,
however many threads are alive when the body begins: the racing thread
here has ended by then, and the count of threads is made to say one.
"""

from __future__ import annotations

import threading

import pytest

from cash import Cash
from cash.decorator import purity_checks

pytestmark = pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning")


class Rec:
    def __init__(self, x):
        self.x = x


def _records():
    return [Rec(i) for i in range(500)]


def _set_records(lst, i, value):
    lst[i].x = value


def _ints():
    return list(range(500))


def _set_ints(lst, i, value):
    lst[i] = value


def _sum_records(lst):
    return sum(p.x for p in lst)


def _many_ints():
    return list(range(5000))  # past what the memo key holds by value: watched


def _nested():
    return [[i, i] for i in range(100)]


def _set_nested(lst, i, value):
    lst[i][0] = value


def _sum_nested(lst):
    return sum(a for a, _ in lst)


KINDS = {
    "records": (_records, _set_records, _sum_records),
    "ints": (_ints, _set_ints, sum),
    "many ints": (_many_ints, _set_ints, sum),
    "nested": (_nested, _set_nested, _sum_nested),
}


def _cash(tmp_path, backend):
    if backend == "memory":
        return Cash(backend="memory", register_magic=False)  # no thread of cash's own
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.fixture
def one_thread_counted(monkeypatch):
    """The count says one thread, as in a process the racing thread has
    left: nothing may hang on it."""
    monkeypatch.setattr(threading, "active_count", lambda: 1)


@pytest.mark.parametrize("backend", ["disk", "memory"])
@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_a_miss_whose_list_another_thread_changed_and_keyed(
    tmp_path, watching, monkeypatch, one_thread_counted, kind, backend
):
    make, set_item, truth = kind

    def total(lst):
        return truth(lst)

    f = _cash(tmp_path, backend).cache(total)
    lst = make()
    watching["lst"] = lst
    set_item(lst, 7, -1)
    for _ in range(3):
        f(lst)  # the watchers warm, on another content
    set_item(lst, 7, 7)  # the content the next call keys, never stored

    real = purity_checks.PurityChecks.argument_identities
    main, done = threading.current_thread(), []

    def other_thread():
        set_item(lst, 5, 9)
        done.append(f(lst))

    def argument_identities(self, *args, **kwargs):
        if threading.current_thread() is main and not done:
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join(30)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(purity_checks.PurityChecks, "argument_identities", argument_identities)
    f(lst)  # keyed before the change, its body runs after it
    monkeypatch.setattr(purity_checks.PurityChecks, "argument_identities", real)
    assert done == [truth(lst)]

    set_item(lst, 5, 5)  # back to the content the call was keyed by
    assert f(lst) == truth(lst)
    assert f(lst) == truth(lst)


@pytest.mark.parametrize("backend", ["disk", "memory"])
def test_a_record_swapped_in_and_back_during_the_key_walk(tmp_path, watching, monkeypatch, one_thread_counted, backend):
    """Another thread swaps a record into the list while its key is walked
    and puts the old one back before the walk ends. The key is of a copy
    taken before the walk, so it is of the content the body runs on."""
    from cash.decorator import arg_hashing

    def total(lst):
        return _sum_records(lst)

    f = _cash(tmp_path, backend).cache(total)
    lst = _records()
    watching["lst"] = lst
    for _ in range(3):
        f(lst)
    _set_records(lst, 7, -1)
    other = Rec(10**6)
    real = arg_hashing.canonical_bytes
    fired = []

    def canonical_bytes(form, *args, **kwargs):
        # The list, or a copy of it the key build walks instead.
        if not fired and any(type(x) is list and x and x[0] is lst[0] for x in [*form[0], *form[1].values()]):
            fired.append(True)
            old = lst[5]
            lst[5] = other
            try:
                return real(form, *args, **kwargs)
            finally:
                lst[5] = old
        return real(form, *args, **kwargs)

    monkeypatch.setattr(arg_hashing, "canonical_bytes", canonical_bytes)
    assert f(lst) == _sum_records(lst)
    assert fired
    monkeypatch.setattr(arg_hashing, "canonical_bytes", real)
    lst[5] = other
    assert f(lst) == _sum_records(lst)
