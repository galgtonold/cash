"""A miss of a function that returns a number does not pay for checks
such a result cannot need, nor for answers that hold from call to call.

Each of these ran on every miss of ``def f(x): return x + 1``: the
function's globals copied for a look the result never reaches, the
arguments bound and the globals read to find what a number shares with
them, each of cash's own threads judged again by the code it runs, and
the index of the stored keys built again from the whole record.
"""

from __future__ import annotations

import threading

from cash.decorator import purity_checks, stored_keys
from cash.tracking import trace_recorder
from tests._work_counts import method_calls

CALLS = 150


def _misses(cash, make, count=CALLS):
    f = cash.cache(make)
    for i in range(count):
        f(i)
    return f


def test_the_holders_of_an_id_are_not_gathered_for_a_plain_result(disk_cash):
    with method_calls(purity_checks, "_identity_holders") as gathered:
        _misses(disk_cash, lambda x: x + 1)
    assert gathered.calls == 0, f"the function's globals were copied on {gathered.calls} of {CALLS} misses"


def test_the_holders_are_gathered_when_the_result_could_name_an_object(disk_cash):
    class Thing:
        pass

    # The address of an object, spelled before the call: a text that
    # could name one of the objects the function can reach.
    spelled = str(id(Thing()))
    with method_calls(purity_checks, "_identity_holders") as gathered:
        _misses(disk_cash, lambda x: [spelled, x], count=3)
        _misses(disk_cash, lambda x: f"{spelled}-{x}", count=3)
    assert gathered.calls == 6


def test_a_number_or_a_text_is_not_searched_for_what_it_shares(disk_cash):
    with method_calls(purity_checks, "_supplied") as bound:
        _misses(disk_cash, lambda x: x + 1)
        _misses(disk_cash, lambda x: f"row {x}")
        _misses(disk_cash, lambda x: None if x % 2 else x / 3)
    assert bound.calls == 0, f"the arguments were bound for the share check on {bound.calls} misses"


def test_a_container_is_still_searched_for_what_it_shares(disk_cash):
    with method_calls(purity_checks, "_supplied") as bound:
        _misses(disk_cash, lambda x: [x], count=3)
    assert bound.calls == 3


def test_the_index_of_stored_keys_is_built_once_for_a_loop_of_misses(disk_cash):
    built = []
    real = stored_keys._View.__init__
    with __import__("pytest").MonkeyPatch.context() as patch:
        patch.setattr(stored_keys._View, "__init__", lambda self, *a: (built.append(1), real(self, *a))[1])
        _misses(disk_cash, lambda x: x + 1)
    # Once, and once more for each time another reader's stamp of the file moved.
    assert len(built) <= 3, f"the index was built {len(built)} times in {CALLS} misses"


def test_each_thread_is_judged_once_by_the_code_it_runs(disk_cash):
    f = _misses(disk_cash, lambda x: x + 1, count=5)  # cash's writer threads are running now
    with method_calls(trace_recorder, "_counts_as_another") as judged:
        for i in range(100, 100 + CALLS):
            f(i)
    assert judged.calls <= 2 * threading.active_count(), (
        f"cash's threads were judged {judged.calls} times in {CALLS} misses"
    )


def test_a_thread_of_the_users_started_later_is_still_counted():
    stop = threading.Event()
    assert not trace_recorder.other_thread_alive(None, working=True)
    worker = threading.Thread(target=stop.wait, args=(30,))
    worker.start()
    try:
        assert trace_recorder.other_thread_alive(None, working=True)
        assert trace_recorder.other_thread_alive(None, working=True), "the kept answer for the thread is the same"
    finally:
        stop.set()
        worker.join()
    assert not trace_recorder.other_thread_alive(None, working=True)
