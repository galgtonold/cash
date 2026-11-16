"""A name some cached code reads moves only the memos that read it.

``warnings.catch_warnings`` sets ``warnings.filters`` (and the display
hooks) and puts them back on exit; ``contextlib.redirect_stderr`` does the
same to ``sys.stderr``. pandas runs a ``catch_warnings`` in
``DataFrame.to_string`` and many other calls. Once any code cash scanned in
the process named ``filters`` or ``stderr`` (a field of a user's own
object is enough), each of those writes moved the whole world, and no
cached function of the process was served from its fast path again: a hit
cost a scan instead of a microsecond. A change to such a name now moves the
memos whose own code names it, and putting back the very object it replaced
brings them back; a rebinding that stays still moves the key.
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
]

cw = hit_memo._cw

# Names a library sets and puts back, read here by user code of its own.
filters = ("a",)
stderr = "plain"


def unrelated(x):
    return x + 1


def reads_filters(x):
    return (x, len(filters))


def reads_stderr(x):
    return (x, stderr)


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


@pytest.fixture(autouse=True)
def _names_kept():
    global filters, stderr
    saved = filters, stderr
    # As after any code that named them: the worst case, always.
    cw.world_add_names(("filters", "stderr", "showwarning", "_showwarnmsg_impl"))
    yield
    filters, stderr = saved


def _fast(cached, *args):
    """Call *cached*; was the call served from the fast path?"""
    before = hit_memo.status()["fast_hits"]
    result = cached(*args)
    return result, hit_memo.status()["fast_hits"] > before


def _settle(cached, *args):
    """Call until a call is served from the fast path. Another test's
    leftovers (xdist) can move the world now and then, and the memo then
    waits before it looks again (`HitMemo.RESCAN_AFTER_S`)."""
    deadline = time.monotonic() + 5 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while not _fast(cached, *args)[1]:
        assert time.monotonic() < deadline, "not served from the fast path: the test proves nothing"
        time.sleep(0.01)


def _library_round_trip():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        pass
    with contextlib.redirect_stderr(io.StringIO()):
        pass


def test_a_library_setting_and_putting_back_keeps_every_fast_path(c):
    f, g, h = c.cache(unrelated), c.cache(reads_filters), c.cache(reads_stderr)
    for cached in (f, g, h):
        _settle(cached, 1)
    for _ in range(5):
        _library_round_trip()
        assert _fast(f, 1) == (2, True)
        # Put back as it was: the names these read are too.
        assert _fast(g, 1) == ((1, 1), True)
        assert _fast(h, 1) == ((1, "plain"), True)


def test_pandas_to_string_keeps_an_unrelated_fast_path(c):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    df.to_string()  # its first call imports and defines what it uses
    f = c.cache(unrelated)
    _settle(f, 1)
    for _ in range(5):
        df.to_string()
        assert _fast(f, 1) == (2, True)


def test_a_name_left_changed_moves_only_the_memos_that_read_it(c):
    global filters, stderr
    f, g, h = c.cache(unrelated), c.cache(reads_filters), c.cache(reads_stderr)
    for cached in (f, g, h):
        _settle(cached, 1)
    saved = filters
    filters = ("a", "b", "c")
    with warnings.catch_warnings():  # in between: undone, not the change above
        pass
    assert _fast(f, 1) == (2, True)
    assert _fast(h, 1) == ((1, "plain"), True)
    assert g(1) == (1, 3)
    _settle(g, 1)
    assert _fast(g, 1) == ((1, 3), True)
    # The very object it held before: the old entry stands again.
    filters = saved
    assert g(1) == (1, 1)
    stderr = "rebound"
    assert h(1) == (1, "rebound")
    assert _fast(f, 1) == (2, True)
    stderr = "plain"
    assert h(1) == (1, "plain")


def test_a_put_back_after_another_change_is_a_new_state():
    d1, d2 = {"filters": 1}, {"filters": 2}
    cw.world_watch_dict(d1, False)
    cw.world_watch_dict(d2, False)
    names = {"filters"}
    a, b = object(), object()
    d1["filters"] = a
    start = cw.scoped_cells(names)
    d1["filters"] = b
    moved = cw.scoped_cells(names)
    assert moved != start
    d1["filters"] = a  # undone, last in first out
    assert cw.scoped_cells(names) == start
    # Changed in one dict, then in another, then put back in the first:
    # not as it was, so a new epoch, never an earlier one.
    d1["filters"] = b
    d2["filters"] = b
    d1["filters"] = a
    now = cw.scoped_cells(names)
    assert now not in (start, moved)
    d2["filters"] = 2
    assert cw.scoped_cells(names) not in (start, moved, now)
