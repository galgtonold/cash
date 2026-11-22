"""What a ``warnings.warn`` does is checked before serving a call that warned.

``warnings`` is library code, so no key held any of it, yet the filters and
the display hooks a program rebinds decide what a warning does at the time
of the call: ``simplefilter("error")`` makes it an exception, and a user
function installed as ``warnings.showwarning`` or ``warnings.formatwarning``
runs and may raise or write what the call returns. A cached ``f(1)`` that
warned under one setting was served under another, where plain Python
raised or returned something else. A run that warned now keeps the part
of the warnings state a result can see, and a lookup serves it only while
that stands (`warning_state`); a run that did not warn is served whatever
the state, in a script and a notebook alike.

Each scenario changes one of them between calls of one function and checks
every call returns (or raises) what plain Python does, on the ordinary
path, through the remembered state segment (`state_memo`) and on the fast
hit path (`hit_memo`), where it first checks the call IS served from the
fast path. What only decides whether or how a line is printed -- a filter
behind the last ``"error"`` one, ``catch_warnings`` in the body, a
library's display hook -- leaves the entry served, the fast path too.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import sys
import time
import warnings

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo, warning_state

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]


class Hooked(Exception):
    pass


def raising_hook(message, category, filename, lineno, file=None, line=None):
    if str(message) == "w-key":
        raise Hooked(str(message))


def silent_hook(message, category, filename, lineno, file=None, line=None):
    pass


def custom_format(message, category, filename, lineno, line=None):
    return f"CUSTOM {message}\n"


def hooked(x):
    try:
        warnings.warn("w-key")
    except Hooked:
        return (x, "hooked")
    return (x, "shown")


def caught(x):
    try:
        warnings.warn("w-key")
    except UserWarning:
        return (x, "raised")
    return (x, "passed")


def uncaught(x):
    warnings.warn("w-key")
    return x + 1


def printed(x):
    # The line ``warnings`` shows it as, which a rebound ``formatwarning`` writes.
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        warnings.warn("w-key")
    return (x, buf.getvalue().splitlines()[0] if buf.getvalue() else "")


def quiet(x):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        warnings.warn("w-key")
    return x + 1


@pytest.fixture(params=["ordinary", "state memo", "fast"])
def path(request, monkeypatch):
    if request.param == "ordinary":
        monkeypatch.setattr(hit_memo, "ENABLED", False)
        monkeypatch.setattr(state_memo, "ENABLED", False)
    elif request.param == "state memo":
        if not state_memo.ENABLED:
            pytest.skip("the state memo needs the C extension")
        monkeypatch.setattr(hit_memo, "ENABLED", False)
    elif not hit_memo.ENABLED:
        pytest.skip("the fast hit path needs the C extension")
    return request.param


@pytest.fixture(autouse=True)
def _own_warnings():
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("always")
        saved = warnings.showwarning, warnings.formatwarning
        warnings.showwarning = warnings._showwarning_orig
        warnings.formatwarning = warnings._formatwarning_orig
        try:
            yield
        finally:
            warnings.showwarning, warnings.formatwarning = saved


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _plain(fn, *args):
    try:
        return ("returned", fn(*args))
    except Exception as e:  # what plain Python does, raising included
        return ("raised", type(e))


def _same(cached, fn, *args):
    """Call *cached* and plain *fn* under the same warnings state."""
    want = _plain(fn, *args)
    got = _plain(cached, *args)
    assert got == want
    return got


def _settle(cached, fn, path, *args):
    """Call until a call is served without running the body; on the fast
    path, until it is served from the memo, and check that it is. Another
    test's leftovers in the process (xdist) can move the world now and
    then, and the memo then waits before it looks again
    (`HitMemo.RESCAN_AFTER_S`): it is given a few of those.

    Plain *fn* runs once, before the wait: its body sets and puts back
    ``sys.stderr`` or ``warnings.filters``, and once any code cash scanned in
    the process names that key, each such write moves the world and no
    memo is served until it is scanned again. Run between the cached
    calls, it kept the memo from ever arming."""
    for _ in range(3):
        _same(cached, fn, *args)
    if path != "fast":
        return
    # As after a test whose cached code named them: the worst case, always.
    hit_memo._cw.world_add_names(("stderr", "filters"))
    want = _plain(fn, *args)
    deadline = time.monotonic() + 5 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while True:
        before = hit_memo.status()["fast_hits"]
        assert _plain(cached, *args) == want
        if hit_memo.status()["fast_hits"] > before:
            return
        assert time.monotonic() < deadline, "not served from the fast path: the test proves nothing"
        time.sleep(0.01)


def test_a_showwarning_hook_rebound_moves_the_key(c, path):
    f = c.cache(hooked)
    _settle(f, hooked, path, 1)
    warnings.showwarning = raising_hook
    assert _same(f, hooked, 1) == ("returned", (1, "hooked"))
    _settle(f, hooked, path, 1)
    warnings.showwarning = silent_hook
    assert _same(f, hooked, 1) == ("returned", (1, "shown"))
    warnings.showwarning = raising_hook
    assert _same(f, hooked, 1) == ("returned", (1, "hooked"))
    warnings.showwarning = warnings._showwarning_orig
    assert _same(f, hooked, 1) == ("returned", (1, "shown"))


def test_a_filter_that_raises_moves_the_key(c, path):
    f, g = c.cache(caught), c.cache(uncaught)
    _settle(f, caught, path, 1)
    _settle(g, uncaught, path, 1)
    warnings.simplefilter("error")
    assert _same(f, caught, 1) == ("returned", (1, "raised"))
    assert _same(g, uncaught, 1) == ("raised", UserWarning)
    assert _same(g, uncaught, 1) == ("raised", UserWarning)
    # An "ignore" in front shadows it: plain Python warns and goes on.
    warnings.filterwarnings("ignore", message="w-key")
    assert _same(f, caught, 1) == ("returned", (1, "passed"))
    assert _same(g, uncaught, 1) == ("returned", 2)
    warnings.resetwarnings()
    warnings.simplefilter("always")
    assert _same(f, caught, 1) == ("returned", (1, "passed"))
    warnings.defaultaction = "error"
    try:
        warnings.resetwarnings()
        assert _same(f, caught, 1) == ("returned", (1, "raised"))
    finally:
        warnings.defaultaction = "default"
    warnings.simplefilter("always")
    assert _same(f, caught, 1) == ("returned", (1, "passed"))


def _stdlib_show():
    """``warnings._showwarnmsg_impl`` as the standard library defines it,
    which pytest's recorder replaced: the one that writes to ``sys.stderr``."""
    home = sys.modules.get("_py_warnings")
    if home is not None and hasattr(home, "_showwarnmsg_impl"):
        return home._showwarnmsg_impl
    tree = ast.parse(inspect.getsource(warnings))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_showwarnmsg_impl")
    space = {}
    exec(compile(ast.Module([node], []), warnings.__file__, "exec"), warnings.__dict__, space)
    return space["_showwarnmsg_impl"]


@pytest.fixture
def stdlib_display(monkeypatch):
    monkeypatch.setattr(warnings, "_showwarnmsg_impl", _stdlib_show())


def test_a_formatwarning_hook_rebound_moves_the_key(c, path, stdlib_display):
    f = c.cache(printed)
    _settle(f, printed, path, 1)
    warnings.formatwarning = custom_format
    assert _same(f, printed, 1) == ("returned", (1, "CUSTOM w-key"))
    _settle(f, printed, path, 1)
    warnings.formatwarning = warnings._formatwarning_orig
    got = _same(f, printed, 1)
    assert "UserWarning: w-key" in got[1][1]


def test_a_filter_that_only_decides_printing_keeps_the_key(c, path):
    f = c.cache(caught)
    _settle(f, caught, path, 1)
    misses = f.cache_info()["misses"]
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.simplefilter("always", ResourceWarning, append=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        _settle(f, caught, path, 1)
    _settle(f, caught, path, 1)
    assert f.cache_info()["misses"] == misses


def test_any_filter_that_raises_moves_the_key(c, path):
    # Whether a warning of the call's would match it is not known before
    # the call: one that raises is folded whatever it matches.
    f = c.cache(caught)
    _settle(f, caught, path, 1)
    misses = f.cache_info()["misses"]
    warnings.filterwarnings("error", category=DeprecationWarning, append=True)
    assert _same(f, caught, 1) == ("returned", (1, "passed"))
    assert f.cache_info()["misses"] == misses + 1
    _settle(f, caught, path, 1)


def test_catch_warnings_in_the_body_keeps_the_fast_path(c, path):
    f = c.cache(quiet)
    _settle(f, quiet, path, 1)
    assert warning_state.fingerprint() is None
    misses = f.cache_info()["misses"]
    _settle(f, quiet, path, 1)
    assert f.cache_info()["misses"] == misses


def test_plain_defaults_fold_nothing():
    assert warning_state.fingerprint() is None
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    assert warning_state.fingerprint() is None
    warnings.filterwarnings("error", category=DeprecationWarning, append=True)
    assert warning_state.fingerprint() is not None


@pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension")
def test_the_fast_path_goes_through_the_warnings_check(c):
    """With the check told every change is none, a fast hit serves the old
    result: the scenarios above go through it."""
    cw = hit_memo._cw
    f = c.cache(caught)
    _settle(f, caught, "fast", 1)
    cw.warn_watch(warnings.__dict__, None, lambda: None)
    try:
        warnings.simplefilter("error")
        assert f(1) == (1, "passed")
        assert caught(1) == (1, "raised")
    finally:
        warning_state.watch(cw)
    assert _same(f, caught, 1) == ("returned", (1, "raised"))


def silent(x):
    return x * 3


def test_a_run_that_did_not_warn_is_served_under_any_filters(c, path):
    f = c.cache(silent)
    _settle(f, silent, path, 2)
    misses = f.cache_info()["misses"]
    warnings.simplefilter("error")
    warnings.showwarning = raising_hook
    _settle(f, silent, path, 2)
    assert f.cache_info()["misses"] == misses


def test_a_library_display_hook_shares_entries(c, path):
    # ``logging.captureWarnings`` shows warnings through a logger: how a
    # line is shown, as IPython's hook does, is no part of what is served.
    import logging

    f = c.cache(caught)
    _settle(f, caught, path, 1)
    misses = f.cache_info()["misses"]
    logging.captureWarnings(True)
    try:
        assert warning_state.fingerprint() is None
        _settle(f, caught, path, 1)
    finally:
        logging.captureWarnings(False)
    assert f.cache_info()["misses"] == misses


CHILD = """
import sys, warnings
from cash import Cash

c = Cash(cache_dir=sys.argv[1], register_magic=False)


@c.cache
def checked(x):
    try:
        warnings.warn("w-key")
    except UserWarning:
        return (x, "raised")
    return (x, "passed")


@c.cache
def plain(x):
    return x + 1


print(checked(1), plain(1), checked.cache_info()["misses"], plain.cache_info()["misses"])
"""


def test_an_entry_of_a_run_that_warned_is_checked_in_another_process(tmp_path):
    import subprocess

    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    store = str(tmp_path / ".cash")

    def run(*flags):
        if sys.version_info < (3, 12):
            # The first call a process stores says this once there; it is
            # not the warning the body raises.
            flags = (*flags, "-W", "ignore:[KEY-LIBRARY-STATE-UNSEEN]")
        out = subprocess.run(
            [sys.executable, *flags, str(script), store], capture_output=True, text=True, check=True, cwd=tmp_path
        )
        return out.stdout.strip()

    assert run("-W", "always") == "(1, 'passed') 2 1 1"
    assert run("-W", "always") == "(1, 'passed') 2 0 0"
    # Raising now: the run that warned is not served; the one that did not is.
    assert run("-W", "error::UserWarning") == "(1, 'raised') 2 1 0"
    assert run("-W", "error::UserWarning") == "(1, 'raised') 2 0 0"


def test_a_nested_run_that_warned_counts_for_the_outer_one(c, path):
    inner = c.cache(caught)

    def outer(x):
        return (inner(x), x)

    f = c.cache(outer)
    _settle(f, outer, path, 1)
    warnings.simplefilter("error")
    assert _same(f, outer, 1) == ("returned", ((1, "raised"), 1))
    warnings.simplefilter("always")
    assert _same(f, outer, 1) == ("returned", ((1, "passed"), 1))


def test_concurrent_bodies_do_not_see_each_other_change_the_filters(c):
    # The filter one body's run puts first is not a change another body
    # made to the warnings filters (`process_registries`): all four cache.
    from concurrent.futures import ThreadPoolExecutor

    def slow(x):
        time.sleep(0.1)  # the others start while the first still runs
        return x * x

    f = c.cache(slow)
    with ThreadPoolExecutor(max_workers=4) as ex:
        assert list(ex.map(f, [1, 2, 3, 4])) == [1, 4, 9, 16]
    assert [f(x) for x in (1, 2, 3, 4)] == [1, 4, 9, 16]
    assert f.cache_info()["misses"] == 4


def copied_filters(x):
    # As scikit-learn hands the filters to its workers: a pattern by its text.
    import re

    texts = [m.pattern for _a, m, _c, _m, _l in warnings.filters if m is not None and not isinstance(m, str)]
    return (x, [re.compile(t).match("w-key") is not None for t in texts])


def test_the_filter_a_run_puts_first_reads_as_a_pattern_that_matches_nothing(c):
    f = c.cache(copied_filters)
    x, matched = f(1)
    assert x == 1 and matched and not any(matched)
