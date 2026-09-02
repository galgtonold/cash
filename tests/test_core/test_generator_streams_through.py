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

import itertools
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


def test_the_producer_never_runs_ahead_of_the_consumer(c):
    """The streaming assertion, without a clock.

    A wall-clock threshold is the one flake class this suite keeps
    rediscovering, and it is not needed: buffering and streaming differ in
    OBSERVABLE ORDER, not just in timing. If the generator were drained first,
    `produced` would already hold every item when the consumer sees the first.
    """
    produced: list[int] = []

    @c.cache
    def stream():
        for i in range(5):
            produced.append(i)
            yield i

    seen: list[int] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item in stream():
            seen.append(item)
            assert len(produced) == len(seen), (
                f"producer is {len(produced) - len(seen)} items ahead -- "
                f"it was drained, not streamed"
            )

    assert seen == [0, 1, 2, 3, 4]


def test_the_first_item_arrives_before_the_last_is_produced(c):
    """The same property in wall-clock terms, kept because it is what a user
    actually feels. Generous threshold: it only has to separate 'streamed'
    from 'the whole 0.4s of work happened first'."""
    @c.cache
    def slow_stream():
        for i in range(4):
            time.sleep(0.1)
            yield i

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arrivals, items = _drain(slow_stream)

    assert items == [0, 1, 2, 3]
    assert arrivals[0] < arrivals[-1] / 2, (
        f"first item at {arrivals[0]:.2f}s of {arrivals[-1]:.2f}s -- buffered"
    )


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


# --------------------------------------------------------------------------
# The rest of the hazard list. Each of these was named as a risk when the
# change was designed; a named risk with no test is just a comment.
# --------------------------------------------------------------------------
def test_the_caller_keeps_items_received_before_an_exception(c):
    """Streaming changes error semantics, and this pins the new ones.

    Buffering meant the caller saw nothing when the producer raised. It now
    sees everything produced up to the failure -- which is what the uncached
    generator does, and the whole point of not changing behaviour.
    """
    @c.cache
    def stream():
        yield 0
        yield 1
        raise RuntimeError("provider died")

    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            for item in stream():
                seen.append(item)
    assert seen == [0, 1], "items produced before the failure must reach the caller"


def test_abandoning_a_multi_chunk_stream_leaves_no_chunks_behind(c):
    """Chunks written before the caller gave up are unreferenced -- no manifest
    names them -- so they are dropped. Otherwise every abandoned run leaks."""
    @c.cache(chunk_max_items=2)
    def stream():
        yield from range(20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        it = stream()
        for _ in range(7):          # forces at least three chunk writes
            next(it)
        it.close()

    leftover = [e for e in c.backend.list_entries()
                if "chunk_" in str(e.get("key", ""))]
    assert not leftover, f"orphan chunks left behind: {leftover}"


def test_an_infinite_generator_streams_and_simply_never_caches(c):
    """This used to be a documented footgun: 'the first call never returns',
    because caching drained it. It streams now; it just never commits."""
    calls = []

    @c.cache
    def forever():
        calls.append(1)
        i = 0
        while True:
            yield i
            i += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = list(itertools.islice(forever(), 5))
        second = list(itertools.islice(forever(), 5))

    assert first == second == [0, 1, 2, 3, 4]
    assert len(calls) == 2, "nothing can be cached -- it never finishes"


def test_a_cached_generator_consuming_another_one_streams(c):
    """Nested cached generators. The tracker keeps a token stack, so the inner
    suspend/resume nests inside the outer one -- asserted rather than assumed.
    """
    inner_produced: list[int] = []

    @c.cache
    def inner():
        for i in range(4):
            inner_produced.append(i)
            yield i

    @c.cache
    def outer():
        for value in inner():
            yield value * 10

    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item in outer():
            seen.append(item)
            assert len(inner_produced) == len(seen), "the inner one was drained"

    assert seen == [0, 10, 20, 30]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(outer()) == [0, 10, 20, 30], "and it still caches"


def test_argument_mutation_is_still_reported_for_a_generator(c):
    """`_check_argument_mutation` used to run right after the call. It now runs
    at exhaustion, so it needs to be shown still running at all."""
    @c.cache
    def stream(rows):
        rows.append("mutated")
        yield len(rows)

    rows = ["a"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(stream(rows))
        messages = [str(w.message) for w in caught]

    assert any("mutat" in m.lower() for m in messages), messages


def test_a_write_inside_a_LIBRARY_is_still_observed(c, tmp_path):
    """Same move for the runtime effect observer: it reports at exhaustion now.

    Two things are load-bearing in the shape below. `shutil.copyfile` rather
    than `path.write_text`, because the analyzer catches the latter by name
    and then suppresses the observer as a duplicate. And its return value is
    USED, because a bare call is caught by the discarded-return heuristic and
    suppressed the same way. Both earlier versions of this test passed while
    proving only that the STATIC pass runs.
    """
    import shutil

    source = tmp_path / "src.txt"
    source.write_text("payload", encoding="utf-8")
    target = tmp_path / "copied.txt"

    @c.cache
    def stream():
        yield 1
        yield str(shutil.copyfile(source, target))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(list(stream())) == 2
        messages = [str(w.message) for w in caught]

    assert target.exists()
    assert any("static analysis did not see" in m for m in messages), messages


# --------------------------------------------------------------------------
# Cross-process survival. Caching a generator exists for the SECOND process;
# this is the property that makes the feature worth having at all.
# --------------------------------------------------------------------------
def test_a_cached_generator_survives_a_fresh_cash_instance(tmp_path):
    """The bug a user found by re-running a demo and noticing it stayed slow.

    Chunks were written with `execution_time=0`, which put them under the
    smart-persistence compute floor, so they stayed RAM-only while the
    manifest -- carrying the real time -- went to disk. A fresh process found
    a manifest with nothing behind it and, because a missing chunk terminates
    iteration, returned an EMPTY iterator. Not slow: wrong, and silent.
    """
    runs = []

    def build(cash_dir):
        c = Cash(cache_dir=str(cash_dir), register_magic=False)

        @c.cache
        def stream():
            runs.append(1)
            for i in range(5):
                time.sleep(0.03)
                yield i
        return stream

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(build(tmp_path / "c")()) == [0, 1, 2, 3, 4]
        # A brand-new instance over the same directory: no shared RAM tier,
        # which is what makes this stand in for a second process.
        assert list(build(tmp_path / "c")()) == [0, 1, 2, 3, 4]

    assert len(runs) == 1, "the second instance recomputed instead of restoring"


def test_a_manifest_whose_chunks_vanished_is_a_miss_not_a_short_answer(tmp_path):
    """Eviction can still reach a chunk without reaching its manifest.

    The reader stops at the first missing chunk, so the entry would serve
    fewer items than it stored. Recomputing is the only honest answer.
    """
    runs = []
    c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)

    @c.cache(chunk_max_items=2)
    def stream():
        runs.append(1)
        for i in range(6):
            time.sleep(0.02)
            yield i

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(stream()) == [0, 1, 2, 3, 4, 5]
        assert list(stream()) == [0, 1, 2, 3, 4, 5]
        assert len(runs) == 1

        # Lose one chunk, as eviction would.
        key = next(e["key"] for e in c.backend.list_entries()
                   if str(e["key"]).endswith(":chunk_1"))
        c.backend.delete(key)

        assert list(stream()) == [0, 1, 2, 3, 4, 5], "a short answer was served"
    assert len(runs) == 2, "the damaged entry must recompute"
