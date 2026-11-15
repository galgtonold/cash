"""A call whose verdict says it changes its argument in a way a hit cannot
replay runs without building its key, except every `REPROBE_EVERY`-th call,
which is keyed and checked again. The verdict is kept from the first such
call when no later call could write the argument back; a closure keeps one
while each value it captures can be named, and a bound method one per type
of the object it is bound to."""

from __future__ import annotations

import functools
import logging
import sys
import warnings

import pytest

from cash import Cash
from cash.decorator.refusal_verdicts import REPROBE_EVERY


class Box:
    __slots__ = ("log",)

    def __init__(self):
        self.log = []


def test_only_the_probes_build_a_key(disk_cash, monkeypatch):
    @disk_cash.cache
    def fill(box):
        box.log.append(1)
        return len(box.log)

    builds = []
    real = disk_cash._keys.build
    monkeypatch.setattr(disk_cash._keys, "build", lambda *a, **k: builds.append(1) or real(*a, **k))
    boxes = [Box() for _ in range(2 * REPROBE_EVERY + 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = [fill(b) for b in boxes]
    assert results == [1] * len(boxes)
    assert all(b.log == [1] for b in boxes)
    # The first call, then one probe per REPROBE_EVERY calls.
    assert len(builds) == 3, len(builds)


def test_a_call_that_was_stored_leaves_no_verdict(disk_cash):
    @disk_cash.cache
    def size(box):
        return len(box.log)

    for _ in range(3):
        assert size(Box()) == 0
    assert size.cache_info()["hits"] == 2


def _counting_builds(cash, monkeypatch):
    builds = []
    real = cash._keys.build
    monkeypatch.setattr(cash._keys, "build", lambda *a, **k: builds.append(1) or real(*a, **k))
    return builds


def test_a_first_change_no_later_call_can_write_back_keeps_its_verdict(disk_cash, monkeypatch):
    # A slotted object is no argument a hit writes back: the second call is
    # refused as the first was, so it runs without building its key.
    @disk_cash.cache
    def fill(box):
        box.log.append(1)
        return len(box.log)

    def plain(box):
        box.log.append(1)
        return len(box.log)

    builds = _counting_builds(disk_cash, monkeypatch)
    boxes, plain_boxes = [Box(), Box()], [Box(), Box()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert [fill(b) for b in boxes] == [plain(b) for b in plain_boxes]
    assert [b.log for b in boxes] == [b.log for b in plain_boxes]
    assert len(builds) == 1, len(builds)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="what a call ran is recorded with sys.monitoring, Python 3.12+; below it a call that changes its "
    "argument in place runs every time, with a warning (the tests above)",
)
def test_a_first_change_a_later_call_writes_back_keeps_no_verdict(tmp_path, monkeypatch):
    # A registered hasher names the array by what it returns: the first
    # call is not stored, and the next takes its state first and is.
    np = pytest.importorskip("numpy")
    hashed = Cash(cache_dir=str(tmp_path / "hashed"), register_magic=False)
    hashed.register_hasher(np.ndarray, lambda a: a.tobytes().hex(), override=True)

    @hashed.cache
    def clip(a):
        np.clip(a, 0, 1, out=a)
        return float(a.sum())

    builds = _counting_builds(hashed, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert [clip(np.array([-1.0, 2.0])) for _ in range(3)] == [1.0] * 3
    assert clip.cache_info()["hits"] == 1
    assert len(builds) == 3, len(builds)


def _memmap(logger, n_files):
    """A wrapper as NiMARE's ``use_memmap`` is: it closes over the function
    it wraps, a logger and a count."""

    def inner(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            logger.debug("%d files", n_files)
            return function(*args, **kwargs)

        return wrapped

    return inner


def test_a_wrapper_closing_over_nameable_values_keeps_a_verdict(disk_cash, monkeypatch):
    @disk_cash.cache
    @_memmap(logging.getLogger("cash-test"), 2)
    def fill(box):
        box.log.append(1)
        return len(box.log)

    builds = _counting_builds(disk_cash, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert [fill(Box()) for _ in range(REPROBE_EVERY)] == [1] * REPROBE_EVERY
    assert len(builds) == 1, len(builds)


def test_a_closure_over_a_value_with_no_name_keeps_no_verdict(disk_cash, monkeypatch):
    held = object()

    @disk_cash.cache
    def fill(box):
        box.log.append(held)
        return len(box.log)

    builds = _counting_builds(disk_cash, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert [fill(Box()) for _ in range(3)] == [1, 1, 1]
    assert len(builds) == 3, len(builds)


def test_a_capture_rebound_names_another_verdict(disk_cash, monkeypatch):
    mode = "a"

    @disk_cash.cache
    def fill(box):
        box.log.append(mode)
        return len(box.log)

    builds = _counting_builds(disk_cash, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fill(Box())
        fill(Box())
        assert len(builds) == 1
        mode = "b"
        fill(Box())
    assert len(builds) == 2, len(builds)


class _Fit:
    def fit(self, box):
        box.log.append(type(self).__name__)
        return len(box.log)


class _Ale(_Fit):
    pass


class _Kda(_Fit):
    pass


def test_a_method_bound_to_another_type_keeps_its_own_verdict(disk_cash, monkeypatch):
    builds = _counting_builds(disk_cash, monkeypatch)
    ale, kda = disk_cash.cache(_Ale().fit), disk_cash.cache(_Kda().fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ale(Box())
        ale(Box())
        assert len(builds) == 1
        kda(Box())
        kda(Box())
    assert len(builds) == 2, len(builds)
