"""A call whose verdict says it changes its argument in a way a hit cannot
replay runs without building its key, except every `REPROBE_EVERY`-th call,
which is keyed and checked again."""

from __future__ import annotations

import warnings

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
