"""What a ``warnings.warn`` does is part of a cached call's key.

``warnings`` is library code, so no key held any of it, yet the filters and
the display hooks a program rebinds decide what a warning does at the time
of the call: ``simplefilter("error")`` makes it an exception, and a user
function installed as ``warnings.showwarning`` or ``warnings.formatwarning``
runs and may raise or write what the call returns. A cached ``f(1)`` that
warned under one setting was served under another, where plain Python
raised or returned something else.

Each scenario changes one of them between calls of one function and checks
every call returns (or raises) what plain Python does, on the ordinary
path, through the remembered state segment (`state_memo`) and on the fast
hit path (`hit_memo`), where it first checks the call IS served from the
fast path. What only decides whether a line is printed -- a filter behind
the last ``"error"`` one, ``catch_warnings`` in the body -- moves no key
and leaves the fast path serving.
"""

from __future__ import annotations

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
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warnings.warn("w-key")
    return (x, warnings._formatwarnmsg(seen[0]).splitlines()[0])


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
    path, until it is served from the memo, and check that it is."""
    for _ in range(6):
        _same(cached, fn, *args)
    if path == "fast":
        before = hit_memo.status()["fast_hits"]
        _same(cached, fn, *args)
        assert hit_memo.status()["fast_hits"] > before, "not served from the fast path: the test proves nothing"


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


def test_a_formatwarning_hook_rebound_moves_the_key(c, path):
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
