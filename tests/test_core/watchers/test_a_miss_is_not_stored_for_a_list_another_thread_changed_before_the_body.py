"""A miss is not stored for a list another thread changed before its body.

After a miss, the check for a change made by the body skips its second
hash of a list when the list is shown unchanged: by its identities for a
list of plain values, by its watchers' token for a list of records. Both
were taken as the body began, after the key was built. Another thread
that changed the list and called in between made them stand for its
content, so the body ran on that content and its result was stored
under the key of the content before. The token is now the one the key
build took, and a list of plain values counts only while the watchers
still give the key's digest.
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


KINDS = {
    "records": (_records, _set_records, _sum_records),
    "ints": (_ints, _set_ints, sum),
    "many ints": (_many_ints, _set_ints, sum),
}


@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_a_miss_whose_list_another_thread_changed_and_keyed(tmp_path, watching, monkeypatch, kind):
    make, set_item, truth = kind

    def total(lst):
        return truth(lst)

    f = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(total)
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
    monkeypatch.undo()
    assert done == [truth(lst)]

    set_item(lst, 5, 5)  # back to the content the call was keyed by
    assert f(lst) == truth(lst)
    assert f(lst) == truth(lst)
