"""A refusal verdict (a call that changed its argument in a way a hit cannot
replay runs uncached without its key, `refusal_verdicts`) is kept for a
closure whose captures can all be named, and for a method handed to
``cash.cache`` bound to an object. It is never reused for another capture
or another type of object: a closure whose capture was rebound so that it
no longer writes its argument, or a subclass whose method does not, is
stored and hit as if no verdict had been kept, and each call leaves its
argument as plain Python does."""

from __future__ import annotations

import functools
import logging
import warnings


class Box:
    __slots__ = ("log",)

    def __init__(self):
        self.log = []


def _quiet(calls):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calls()


def test_a_rebound_capture_that_stops_the_write_is_stored(disk_cash):
    writes = True

    def plain(box):
        if writes:
            box.log.append(1)
        return 7

    fill = disk_cash.cache(plain)

    def run():
        boxes, plain_boxes = [Box(), Box()], [Box(), Box()]
        assert [fill(b) for b in boxes] == [plain(b) for b in plain_boxes]
        assert [b.log for b in boxes] == [b.log for b in plain_boxes]

    _quiet(run)
    assert fill.cache_info()["hits"] == 0
    writes = False
    _quiet(run)
    assert fill.cache_info()["hits"] == 1, "the verdict of the writing capture was reused"


def _wrapping(logger, step):
    def inner(function):
        @functools.wraps(function)
        def wrapped(box):
            logger.debug("step %s", step)
            return function(box)

        return wrapped

    return inner


def test_a_wrapper_around_another_function_is_stored_when_that_one_does_not_write(disk_cash):
    def writes(box):
        box.log.append(1)
        return 1

    def reads(box):
        return len(box.log)

    logger = logging.getLogger("cash-test")
    wrapped = _wrapping(logger, 1)(writes)
    fill = disk_cash.cache(wrapped)
    boxes = [Box(), Box()]
    assert _quiet(lambda: [fill(b) for b in boxes]) == [1, 1]
    assert [b.log for b in boxes] == [[1], [1]]
    # The same wrapper, the function it captured swapped.
    cell = wrapped.__closure__[wrapped.__code__.co_freevars.index("function")]
    cell.cell_contents = reads
    assert _quiet(lambda: [fill(Box()) for _ in range(2)]) == [0, 0]
    assert fill.cache_info()["hits"] == 1, "the verdict of the writing function was reused"


class Estimator:
    writes = True

    def fit(self, box):
        if self.writes:
            box.log.append(type(self).__name__)
        return 3


class Writing(Estimator):
    pass


class Reading(Estimator):
    writes = False


def test_a_method_bound_to_a_type_that_does_not_write_is_stored(disk_cash):
    writing, reading = disk_cash.cache(Writing().fit), disk_cash.cache(Reading().fit)
    boxes = [Box(), Box()]
    assert _quiet(lambda: [writing(b) for b in boxes]) == [3, 3]
    assert [b.log for b in boxes] == [["Writing"], ["Writing"]]
    assert writing.cache_info()["hits"] == 0
    boxes = [Box(), Box()]
    assert _quiet(lambda: [reading(b) for b in boxes]) == [3, 3]
    assert [b.log for b in boxes] == [[], []]
    assert reading.cache_info()["hits"] == 1, "the verdict of the writing type was reused"
