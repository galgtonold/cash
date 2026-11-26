"""A library's settings a cached body reads as it runs are checked on lookup.

A key reads a library by its installed version alone, so its settings were in
no key: ``sc.settings.n_jobs`` (a property of the settings class's metaclass
returning ``cls._n_jobs``), ``sklearn.set_config(...)``, ``pd.set_option(...)``,
NumPy's print options and error handling, ``decimal``'s context, a
``ContextVar`` of the user's or a library's. Changing one between two calls of
a cached function served the first call's result. Now the code a body runs
records the chains it reads from its module globals, and the entry is served
only while each ends on what it ended on (`library_reads`).

Each scenario changes one setting between calls and checks every call returns
what plain Python does, on the ordinary path, through the remembered state
segment (`state_memo`) and on the fast hit path (`hit_memo`), where it first
checks the call IS served from the fast path; and in a second process.
"""

from __future__ import annotations

import contextvars
import decimal
import importlib
import subprocess
import sys
import textwrap
import time
import types
import warnings

import pytest

from cash import Cash
from cash.decorator import hit_memo, library_reads, state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]


LIBRARY = """
import contextvars


class _Meta(type):
    @property
    def n_jobs(cls):
        return cls._n_jobs

    @n_jobs.setter
    def n_jobs(cls, v):
        cls._n_jobs = v


class settings(metaclass=_Meta):
    _n_jobs = 1


class _Options:
    def __init__(self):
        self.scale = 1
        self._mode = "a"

    @property
    def mode(self):
        return self._mode

    def get(self, name):
        return getattr(self, name)


options = _Options()
_config = {"display": {"precision": 6}}
THRESHOLD = 10
VAR = contextvars.ContextVar("cash_ut_libset_var", default=1)


def by_metaclass(x):
    return x * settings.n_jobs


def by_instance(x):
    return x * options.scale


def by_property(x):
    return f"{x}:{options.mode}"


def by_method(x):
    return x * options.get("scale")


def by_dict(x):
    return round(x / 7, _config["display"]["precision"])


def by_constant(x):
    return x > THRESHOLD


def by_var(x):
    return x * VAR.get()


def quiet(x):
    return len(str(x))


def set_all(v):
    settings.n_jobs = v
    options.scale = v
    options._mode = "m" * v
    _config["display"]["precision"] = v
    global THRESHOLD
    THRESHOLD = v
    VAR.set(v)
"""


@pytest.fixture
def lib(tmp_path, monkeypatch):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_libset.py").write_text(LIBRARY, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    sys.modules.pop("cash_ut_libset", None)
    mod = importlib.import_module("cash_ut_libset")
    yield mod
    sys.modules.pop("cash_ut_libset", None)
    sys.modules.pop("cash_ut_libset_user", None)


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


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _settle(cached, plain, path, *args, fast=True):
    """Call until a call is served without running the body; on the fast
    path, until it is served from the memo, and check that it is (*fast*:
    for a function the fast path serves; one reading an object's or a
    class's attributes is not).

    The calls follow each other with no pause. Once the world moved, the
    memo scans again, remembers the next call and serves the one after: the
    world has to hold still across those three calls. Another test's
    leftovers (xdist) can move it every few milliseconds, and with a pause
    between the calls no three of them ever met a still world."""
    for _ in range(3):
        assert cached(*args) == plain(*args)
    if path != "fast" or not fast:
        return
    deadline = time.monotonic() + 20 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while True:
        before = hit_memo.status()["fast_hits"]
        assert cached(*args) == plain(*args)
        if hit_memo.status()["fast_hits"] > before:
            return
        assert time.monotonic() < deadline, (
            "not served from the fast path: the test proves nothing",
            hit_memo.status(),
        )


USER_VAR = contextvars.ContextVar("cash_ut_user_var", default=1)


def _scenarios(lib):
    # Code of the user's reading the library's module: what a script's or a
    # cell's own ``lib.settings.n_jobs`` reads.
    user = types.ModuleType("cash_ut_libset_user")
    sys.modules[user.__name__] = user
    user.lib = lib
    exec("def direct(x):\n    return x * lib.settings.n_jobs\n", user.__dict__)
    direct = user.direct

    def user_var(x):
        return x * USER_VAR.get()

    def dec(x):
        return str(decimal.Decimal(1) / decimal.Decimal(x))

    def setter(v):
        lib.set_all(v)
        USER_VAR.set(v)
        decimal.getcontext().prec = 5 + v

    calls = [
        lib.by_metaclass,
        direct,
        lib.by_instance,
        lib.by_property,
        lib.by_dict,
        lib.by_constant,
        lib.by_var,
        user_var,
        dec,
    ]
    # Served from the fast path, whose hits check the settings in Python.
    fast = {lib.by_dict, lib.by_constant, dec}
    return calls, setter, fast


def test_a_changed_library_setting_is_seen(lib, c, path):
    calls, setter, fast = _scenarios(lib)
    ctx = contextvars.copy_context()

    def run():
        for plain in calls:
            cached = c.cache(plain)
            setter(1)
            _settle(cached, plain, path, 7, fast=plain in fast)
            for v in (2, 3, 2, 1):
                setter(v)
                assert cached(7) == plain(7), (plain.__name__, v)
                assert cached(7) == plain(7), (plain.__name__, v)
            assert cached.cache_info()["hits"] > 0, plain.__name__

    saved = decimal.getcontext().prec
    try:
        ctx.run(run)
    finally:
        decimal.getcontext().prec = saved


def test_the_miss_names_the_setting(lib, c):
    f = c.cache(lib.by_metaclass)
    lib.settings.n_jobs = 1
    f(7)
    lib.settings.n_jobs = 2
    assert f(7) == 14
    reasons = str(f.cache_info())
    assert "library setting changed" in reasons
    lib.settings.n_jobs = 1


def test_library_code_that_reads_no_setting_is_not_watched_again(lib, c):
    f = c.cache(lib.quiet)
    for i in range(3):
        f(i)
    assert lib.quiet.__code__ in library_reads.QUIET
    assert f(1) == 1
    assert f.cache_info()["hits"] >= 1


def holds_an_object(x):
    return x * OBJ_VAR.get().k


class _Box:
    k = 3


OBJ_VAR = contextvars.ContextVar("cash_ut_obj_var")


def test_a_context_variable_holding_an_object_runs_uncached_with_a_warning(c):
    ctx = contextvars.copy_context()

    def run():
        OBJ_VAR.set(_Box())
        f = c.cache(holds_an_object)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            assert f(2) == 6
            assert f(2) == 6
        assert any("KEY-LIBRARY-STATE-UNREAD" in str(w.message) for w in seen)
        assert f.cache_info()["hits"] == 0

    ctx.run(run)


def test_numpy_print_options_and_error_handling(c, path):
    np = pytest.importorskip("numpy")

    def shown(n):
        return repr(np.arange(n) * 1.123456789)

    def divided(n):
        return float(np.float64(n) / np.float64(0))

    def plain(fn, n):
        try:
            return ("returned", fn(n))
        except Exception as e:  # plain Python raises under errstate(divide="raise")
            return ("raised", type(e))

    s, d = c.cache(shown), c.cache(divided)
    with np.errstate(all="ignore"):
        _settle(s, shown, path, 3)
        _settle(d, divided, path, 1)
        with np.printoptions(precision=2):
            assert s(3) == shown(3)
    with np.errstate(divide="raise"):
        assert plain(d, 1) == plain(divided, 1) == ("raised", FloatingPointError)
    with np.errstate(all="ignore"):
        assert s(3) == shown(3)
        assert d(1) == divided(1)


SCRIPT = """
import sys
sys.path.insert(0, {site!r})
import cash_ut_libset as lib
from cash import Cash
c = Cash(cache_dir={cache!r}, register_magic=False)
f = c.cache(lib.by_metaclass)
lib.settings.n_jobs = int(sys.argv[1])
print(f(7), f.cache_info()["hits"])
"""


def test_a_setting_is_checked_in_another_process(tmp_path):
    site = tmp_path / "venv" / "lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "cash_ut_libset.py").write_text(LIBRARY, encoding="utf-8")
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(SCRIPT.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run(v):
        out = subprocess.run(
            [sys.executable, str(script), str(v)], capture_output=True, text=True, check=True, timeout=120
        )
        return out.stdout.split()

    assert run(1) == ["7", "0"]
    assert run(2) == ["14", "0"]
    assert run(2) == ["14", "1"]
    assert run(1) == ["7", "0"]


# Where a body is not recorded (CASH_TRACE=0, a coroutine) the
# settings it reads are not checked: the first stored call says so, once a
# process, whatever its own code names. A library reached through an
# argument's method (``df.to_string()``) is named nowhere in that code.

UNSEEN = "KEY-LIBRARY-STATE-UNSEEN"


@pytest.fixture
def unrecorded(monkeypatch):
    monkeypatch.setattr(library_reads, "UNSEEN_SAID", False)
    monkeypatch.setenv("CASH_TRACE", "0")


def _said(seen):
    return [w for w in seen if UNSEEN in str(w.message)]


def adds_one(x):
    return x + 1


def roots(x):
    np = pytest.importorskip("numpy")
    return float(np.sqrt(x))


def norms(x):
    np = pytest.importorskip("numpy")
    return float(np.linalg.norm([x, x]))


def as_text(frame):
    return frame.to_string()


def test_the_first_unrecorded_call_stored_says_so_once(c, unrecorded):
    pytest.importorskip("numpy")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        own = c.cache(adds_one)
        assert own(1) == 2
        f, g = c.cache(roots), c.cache(norms)
        for x in (4, 9, 4):
            assert f(x) == roots(x)
            assert g(x) == norms(x)
    said = _said(seen)
    assert len(said) == 1
    assert said[0].message.code == UNSEEN
    assert "adds_one" in str(said[0].message)
    # Otherwise as before: the entry is served by its key.
    assert f.cache_info()["hits"] == 1
    assert g.cache_info()["hits"] == 1


def _fix_of(message) -> str:
    return str(message).partition("Fix: ")[2]


def test_the_way_out_named_fits_why_nothing_was_recorded(c, unrecorded):
    """With the recording off by ``CASH_TRACE=0`` the
    fix names that, and not a way out for coroutines."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert c.cache(adds_one)(1) == 2
    [said] = _said(seen)
    fix = _fix_of(said.message)
    assert fix.startswith("pass the setting in as an argument, ")
    assert ("unset CASH_TRACE") in fix
    assert "sync function" not in fix and "iterator" not in fix


async def doubles_later(x):
    return x * 2


def test_a_coroutine_is_not_told_to_do_what_it_already_does(c, monkeypatch):
    """A coroutine's body is never recorded: with tracing on,
    the first fix named used to be "run with tracing on"."""
    import asyncio

    monkeypatch.setattr(library_reads, "UNSEEN_SAID", False)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert asyncio.run(c.cache(doubles_later)(2)) == 4
    [said] = _said(seen)
    assert "a coroutine's body is not recorded" in str(said.message)
    fix = _fix_of(said.message)
    assert "3.12" not in fix and "tracing" not in fix and "CASH_TRACE" not in fix, fix
    assert "pass the setting in as an argument" in fix
    assert "into a sync function and cache that one instead" in fix


def test_a_library_reached_through_an_arguments_method_is_said(c, unrecorded):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": [1.23456]})
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        f = c.cache(as_text)
        assert f(frame) == as_text(frame)
    said = _said(seen)
    assert len(said) == 1
    assert "as_text" in str(said[0].message)
    assert "pandas options" in str(said[0].message)


def test_an_unrecorded_call_not_stored_says_nothing(c, unrecorded):
    # The note is about entries served later: a call run uncached has none.
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        f = c.cache(adds_one, cache_if=lambda r: False)
        assert f(1) == 2
    assert not _said(seen)
    assert library_reads.UNSEEN_SAID is False


def test_a_coroutine_reaching_a_library_says_so(c, monkeypatch):
    pytest.importorskip("numpy")
    import asyncio

    monkeypatch.setattr(library_reads, "UNSEEN_SAID", False)

    async def aroots(x):
        return roots(x)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        f = c.cache(aroots)
        assert asyncio.run(f(4)) == 2.0
        assert asyncio.run(f(4)) == 2.0
    said = _said(seen)
    assert len(said) == 1
    assert "coroutine" in str(said[0].message)


def test_a_recorded_call_says_nothing(c, monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(library_reads, "UNSEEN_SAID", False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        f = c.cache(roots)
        f(4)
        f(4)
    assert not _said(seen)
    assert library_reads.UNSEEN_SAID is False


def test_the_note_can_be_silenced(c, unrecorded):
    pytest.importorskip("numpy")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=r"\[KEY-LIBRARY-STATE-UNSEEN\]")
        f = c.cache(roots)
        assert f(4) == 2.0
        assert f(4) == 2.0
    assert not _said(seen)
    assert f.cache_info()["hits"] == 1
