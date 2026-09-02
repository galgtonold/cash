"""A cached generator must stream, not buffer.

Adding `@cash.cache` should not change how a function behaves, and for a
generator it did: cash consumed the whole thing before returning anything, so
the caller saw nothing until the last item was produced. Measured on a fake
LLM stream -- 494ms to first token uncached, 2444ms cached, and the 2444ms was
the entire completion arriving at once.

That is the wrong trade for the shape people actually cache here. A streaming
LLM call cached for its replay value silently loses the streaming it was
written for.

What must hold at the same time, and is why this is not a two-line change:

* a file read LAZILY, between yields, is still recorded as a dependency --
  the reason the old code consumed inside the tracker in the first place;
* the CALLER's own reads, in its loop body, are not attributed to the
  function;
* a partially consumed generator stores nothing -- a truncated result cached
  under the full result's key is a wrong answer, not a slow one. This is the
  one place streaming COSTS something: today cash produces the whole result
  regardless of what the caller takes, so abandoning the loop still leaves a
  complete entry behind. Streaming only produces what is consumed, so
  abandoning it abandons the entry too. Accepted deliberately -- the
  alternative is caching something nobody computed;
* execution time is what the PRODUCER spent, not wall-clock across the
  caller's loop, or a slow consumer would make a trivial generator look
  expensive enough to persist.
"""
from __future__ import annotations

import time
import warnings

import pytest

from cash import Cash


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _drain(make_iter):
    """Return (arrival times, items), timing from BEFORE the call.

    Before the call, because that is where a buffering implementation does its
    work -- time only the loop and a buffered miss looks instant.
    """
    start = time.perf_counter()
    arrivals, items = [], []
    for item in make_iter():
        arrivals.append(time.perf_counter() - start)
        items.append(item)
    return arrivals, items


def test_the_first_item_arrives_before_the_last_is_produced(c):
    @c.cache
    def slow_stream():
        for i in range(4):
            time.sleep(0.1)
            yield i

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arrivals, items = _drain(slow_stream)

    assert items == [0, 1, 2, 3]
    assert arrivals[0] < 0.25, (
        f"first item at {arrivals[0]:.2f}s -- the generator was buffered"
    )
    assert arrivals[-1] >= 0.35, "the last item cannot arrive before it is produced"


def test_the_replay_still_hits_and_is_complete(c):
    calls = []

    @c.cache
    def stream():
        calls.append(1)
        for i in range(4):
            time.sleep(0.1)
            yield i

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(stream()) == [0, 1, 2, 3]
        assert list(stream()) == [0, 1, 2, 3]
    assert len(calls) == 1, "the second call must be served from cache"


def test_a_partially_consumed_generator_stores_nothing(c):
    calls = []

    @c.cache
    def stream():
        calls.append(1)
        for i in range(6):
            time.sleep(0.05)
            yield i

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item in stream():
            if item == 2:
                break                      # abandon it
        # A truncated entry would serve [0, 1, 2] here. It must recompute.
        assert list(stream()) == [0, 1, 2, 3, 4, 5]
    assert len(calls) == 2, "the abandoned run must not have been stored"


def test_a_generator_that_raises_midway_stores_nothing(c):
    calls = []

    @c.cache
    def stream():
        calls.append(1)
        yield 0
        raise RuntimeError("provider died")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            list(stream())
        with pytest.raises(RuntimeError):
            list(stream())
    assert len(calls) == 2, "a failed stream must not be cached"


def test_a_lazily_read_file_is_still_a_dependency(c, tmp_path):
    """The reason the old code buffered inside the tracker. Read between
    yields, so it only happens while the caller iterates."""
    data = tmp_path / "lazy.txt"
    data.write_text("v1", encoding="utf-8")

    @c.cache
    def stream():
        yield "start"
        time.sleep(0.12)
        yield data.read_text(encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(stream()) == ["start", "v1"]
        data.write_text("v2", encoding="utf-8")
        assert list(stream()) == ["start", "v2"], (
            "editing a file the generator reads lazily must invalidate"
        )


def test_the_callers_own_file_reads_are_not_attributed(c, tmp_path):
    """The other half: keeping the tracker open across the caller's loop would
    make the caller's I/O a dependency of the cached function."""
    theirs = tmp_path / "callers.txt"
    theirs.write_text("a", encoding="utf-8")
    calls = []

    @c.cache
    def stream():
        calls.append(1)
        time.sleep(0.12)
        yield 1
        yield 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in stream():
            theirs.read_text(encoding="utf-8")
        theirs.write_text("b", encoding="utf-8")
        list(stream())
    assert len(calls) == 1, "the caller's file must not invalidate the function"


def test_a_slow_consumer_does_not_inflate_the_recorded_time(c):
    """Execution time drives the persistence decision, so it has to be the
    producer's, not wall-clock across a slow loop."""
    @c.cache
    def quick_stream():
        yield from range(3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in quick_stream():
            time.sleep(0.2)                # the CALLER is slow, not the function
    saved = quick_stream.cache_info()["total_time_saved"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        list(quick_stream())
    assert quick_stream.cache_info()["total_time_saved"] - saved < 0.3, (
        "the caller's sleep was charged to the function"
    )
