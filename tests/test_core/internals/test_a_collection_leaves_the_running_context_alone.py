"""A garbage collection's finalizers do not change the context of the code
the collection interrupted.

On Python 3.10 and 3.11 the collector can start inside the allocation a
``ContextVar.set`` or ``reset`` makes. A finalizer that set or reset a
variable then lost its change, or the interrupted call read a mapping freed
under it: cash's next reset of its own variable failed with ``LookupError``
(a CI worker then failed every notebook test after it), or the process
crashed. Each collection now runs in a fresh context of its own
(`cash._gc_context`); 3.12 runs the collector only between bytecodes.
"""

from __future__ import annotations

import contextvars
import gc
import subprocess
import sys
import textwrap

import pytest

import cash  # noqa: F401 - importing cash is what sets collections apart

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 12), reason="3.12 runs the collector between bytecodes, never inside a ContextVar call"
)

RUNNING = contextvars.ContextVar("running", default="none")


class _Garbage:
    seen: list[str] = []

    def __init__(self) -> None:
        self.me = self  # a cycle: only the collector frees it

    def __del__(self) -> None:
        _Garbage.seen.append(RUNNING.get())
        RUNNING.set("finalizer")


def test_a_finalizer_sees_and_leaves_nothing_of_the_running_code():
    _Garbage.seen.clear()
    token = RUNNING.set("the code")
    try:
        _Garbage()
        gc.collect()
        assert _Garbage.seen == ["none"], "the finalizer ran in the interrupted code's context"
        assert RUNNING.get() == "the code", "the finalizer's change reached the interrupted code"
    finally:
        RUNNING.reset(token)


STRESS = textwrap.dedent(
    """
    import contextvars, gc
    import cash
    from cash.tracking.tracker_context import active_tracker, untracked

    OTHER = contextvars.ContextVar("other", default=None)

    class Garbage:
        def __init__(self):
            self.me = self
        def __del__(self):
            token = OTHER.set("finalizer")
            with untracked():
                pass
            OTHER.reset(token)

    gc.set_threshold(1, 1, 1)
    for i in range(5000):
        Garbage()
        token = active_tracker.set(i)
        other = OTHER.set(i)
        with untracked():
            Garbage()
        assert active_tracker.get() == i and OTHER.get() == i
        OTHER.reset(other)
        active_tracker.reset(token)
    print("ok")
    """
)


def test_collections_inside_context_variable_calls_lose_nothing():
    """Every allocation may collect, and every collection runs finalizers
    that set and reset variables: cash's own sets and resets all hold."""
    result = subprocess.run([sys.executable, "-c", STRESS], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and result.stdout.strip() == "ok", (result.returncode, result.stderr[-2000:])


DEEP = textwrap.dedent(
    """
    import contextvars, gc, sys
    import cash

    RUNNING = contextvars.ContextVar("running", default="none")
    seen = []

    class Garbage:
        def __init__(self):
            self.me = self
        def __del__(self):
            seen.append(RUNNING.get())

    def down(n):
        if n:
            return down(n - 1)
        Garbage()
        gc.collect()

    RUNNING.set("the code")
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    # Ever closer to the limit, until the collection itself can no longer
    # be called: its callback then runs a frame or two short of it.
    ran = 0
    for spare in range(30, 0, -1):
        seen.clear()
        try:
            down(sys.getrecursionlimit() - depth - spare)
        except RecursionError:
            break
        # Nearest the limit the finalizer itself may not get a frame: as
        # without cash, it is then not run.
        assert seen in (["none"], []), (spare, seen)
        ran += 1
    assert ran, "no collection ran"
    print("ok")
    """
)


def test_a_collection_near_the_recursion_limit_still_runs_in_its_own_context():
    """A collection can start a frame short of the recursion limit (a pickler
    deep in a 20,000-object chain): entering its context raised there, the
    finalizers ran in the interrupted code's context, and a test failed on
    the unraisable error."""
    result = subprocess.run([sys.executable, "-c", DEEP], capture_output=True, text=True, timeout=120)
    assert "own_context" not in result.stderr, result.stderr[-2000:]
    assert result.returncode == 0 and result.stdout.strip() == "ok", (result.returncode, result.stderr[-2000:])


DEEP_WITH_EVERY_ALLOCATION = textwrap.dedent(
    """
    import contextvars, gc, sys
    import cash

    RUNNING = contextvars.ContextVar("running", default="none")
    errors = []
    sys.unraisablehook = lambda u: errors.append(u.exc_type.__name__)

    class Garbage:
        def __init__(self):
            self.me = self

    def down(n):
        Garbage()  # with a threshold of 1 every allocation collects
        return n if n == 0 else down(n - 1) + 0

    token = RUNNING.set("the code")
    limit = sys.getrecursionlimit()
    gc.set_threshold(1, 1, 1)
    try:
        down(100_000)
    except RecursionError:
        pass
    gc.set_threshold(700, 10, 10)
    assert RUNNING.get() == "the code", "a collection left its own context entered"
    gc.collect()  # puts back a limit a collection deep in the recursion could not
    assert sys.getrecursionlimit() == limit, (sys.getrecursionlimit(), limit)
    RUNNING.reset(token)
    print("ok" if not errors else errors[:3])
    """
)


def test_a_collection_at_the_recursion_limit_raises_nothing():
    """A collection that starts a few frames short of the recursion limit
    (deep in a pickle or a deepcopy) is one ctypes cannot enter a context
    for: it runs in the interrupted context, as without cash, rather than
    raising an unraisable error, which pytest reports as a failure of the
    test it fell in (macOS 3.10, copying a 20,000-node chain)."""
    result = subprocess.run(
        [sys.executable, "-c", DEEP_WITH_EVERY_ALLOCATION], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", (result.stdout[-500:], result.stderr[-2000:])
