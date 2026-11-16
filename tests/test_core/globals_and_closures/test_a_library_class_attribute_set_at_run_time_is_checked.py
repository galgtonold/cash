"""An attribute of an installed library's class, set by the program, is checked.

A library is keyed by its installed version, so ``somelib.Model.scale = 5``
set by a script before a cached call whose body runs ``Model(x).compute()``
(a method reading ``self.scale``), ``Model.ccompute(x)`` (``cls.scale``) or
a library function doing so was in no key: the call was served the result
made under the old value, in the same process and in the next, on every
path. Now the methods that ran are recorded with the class they ran on, and
the plain class attributes they read off ``self``/``cls`` are kept with the
entry (`library_reads`), checked again on lookup as any library setting.

Below Python 3.12 (and with ``CASH_TRACE=0``) nothing is recorded: the first
stored call reaching the library says so once per process with
KEY-LIBRARY-STATE-UNSEEN.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash
from cash.decorator import hit_memo, library_reads, library_versions, state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheIneffectiveWarning"),
]

RECORDED = sys.version_info >= (3, 12)
needs_recording = pytest.mark.skipif(
    not RECORDED, reason="a body's reads are recorded with sys.monitoring, Python 3.12 or newer"
)

LIBRARY = """
class Model:
    scale = 1
    _memo = {}

    def __init__(self, x):
        self.x = x

    def compute(self):
        return self.x * self.scale

    @classmethod
    def ccompute(cls, x):
        return x * cls.scale

    def remembered(self):
        # A class-level cache its method fills: not a setting.
        if self.x not in self._memo:
            self._memo[self.x] = self.x + 1
        return self._memo[self.x]


class Sub(Model):
    pass


def run(x):
    return Model(x).compute()
"""


def _install(root):
    site = root / "venv" / "lib" / "site-packages"
    (site / "cash_ut_libcls").mkdir(parents=True)
    (site / "cash_ut_libcls" / "__init__.py").write_text(LIBRARY, encoding="utf-8")
    info = site / "cash_ut_libcls-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: cash-ut-libcls\nVersion: 1.0\n", encoding="utf-8")
    (info / "top_level.txt").write_text("cash_ut_libcls\n", encoding="utf-8")
    (info / "RECORD").write_text(
        "cash_ut_libcls/__init__.py,,\ncash_ut_libcls-1.0.dist-info/METADATA,,\n", encoding="utf-8"
    )
    return site


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_install(tmp_path)))
    sys.modules.pop("cash_ut_libcls", None)
    mod = importlib.import_module("cash_ut_libcls")
    yield mod
    sys.modules.pop("cash_ut_libcls", None)


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


def _scenarios(lib):
    Model, Sub = lib.Model, lib.Sub

    class Mine(Model):
        """A user's subclass: its method reads the library's attribute."""

        def mine(self):
            return self.x + self.scale

    def instance(x):
        return Model(x).compute()

    def classmethod_(x):
        return Model.ccompute(x)

    def library_function(x):
        return lib.run(x)

    def subclass(x):
        return Sub(x).compute()

    def users_subclass(x):
        return Mine(x).mine()

    return [
        (instance, Model),
        (classmethod_, Model),
        (library_function, Model),
        (subclass, Model),
        (subclass, Sub),
        (users_subclass, Model),
    ]


def _reset(lib):
    lib.Model.scale = 1
    if "scale" in vars(lib.Sub):
        del lib.Sub.scale


def _settle(cached, plain, path):
    """Call until served without running the body; on the fast path, until
    it is served from the memo, and check that it is."""
    for _ in range(3):
        assert cached(7) == plain(7)
    if path != "fast":
        return
    deadline = time.monotonic() + 5 * hit_memo.HitMemo.RESCAN_AFTER_S + 1.0
    while True:
        before = hit_memo.status()["fast_hits"]
        assert cached(7) == plain(7)
        if hit_memo.status()["fast_hits"] > before:
            return
        assert time.monotonic() < deadline, "not served from the fast path: the test proves nothing"
        time.sleep(0.01)


@needs_recording
def test_a_class_attribute_set_between_calls_is_seen(lib, c, path):
    try:
        for plain, target in _scenarios(lib):
            _reset(lib)
            cached = c.cache(plain)
            _settle(cached, plain, path)
            for v in (5, 2, 5, 1):
                target.scale = v
                assert cached(7) == plain(7), (plain.__name__, target.__name__, v)
                assert cached(7) == plain(7), (plain.__name__, target.__name__, v)
            if target is lib.Sub:
                del lib.Sub.scale
                assert cached(7) == plain(7), (plain.__name__, "Sub.scale deleted")
            assert cached.cache_info()["hits"] > 0, plain.__name__
    finally:
        _reset(lib)


@needs_recording
def test_the_miss_names_the_attribute(lib, c):
    def library_function(x):
        return lib.run(x)

    f = c.cache(library_function)
    try:
        assert f(7) == 7
        lib.Model.scale = 3
        assert f(7) == 21
        assert f.cache_info()["miss_reasons"].get("library setting changed") == 1
    finally:
        _reset(lib)


@needs_recording
def test_a_class_cache_a_method_fills_is_not_a_setting(lib, c):
    def remembered(x):
        return lib.Model(x).remembered()

    f = c.cache(remembered)
    assert f(7) == 8
    assert f(8) == 9
    assert f(7) == 8
    assert f.cache_info()["hits"] == 1


SCRIPT = """
import sys
sys.path.insert(0, {site!r})
import cash_ut_libcls as lib
if sys.argv[1] != "unset":
    lib.Model.scale = int(sys.argv[1])
from cash import Cash
c = Cash(cache_dir={cache!r}, register_magic=False)


@c.cache
def f(x):
    return lib.run(x)


@c.cache
def g(x):
    return lib.Model(x).remembered()


print(f(7), f.cache_info()["hits"], g(7), g.cache_info()["hits"])
"""


@needs_recording
def test_an_attribute_set_by_the_script_is_checked_in_another_process(tmp_path):
    site = _install(tmp_path)
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(SCRIPT.format(site=str(site), cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run(v):
        out = subprocess.run([sys.executable, str(script), v], capture_output=True, text=True, check=True, timeout=120)
        return out.stdout.split()

    assert run("unset") == ["7", "0", "8", "0"]
    assert run("5") == ["35", "0", "8", "1"]
    assert run("5") == ["35", "1", "8", "1"]
    assert run("2") == ["14", "0", "8", "1"]
    assert run("unset") == ["7", "0", "8", "1"]
    assert run("unset") == ["7", "1", "8", "1"]


LIB = None


def through_global(x):
    return LIB.run(x)


def classmethod_through_global(x):
    return LIB.Model.ccompute(x)


def test_unrecorded_a_call_reaching_the_class_says_so_once(lib, c, monkeypatch, tmp_path):
    # Named by a global: the key finds the library the call reaches.
    monkeypatch.setitem(globals(), "LIB", lib)
    # The distributions read afresh: an earlier test's look did not list this one.
    library_versions.forget()
    monkeypatch.setattr(library_versions, "per_user_cache_root", lambda: tmp_path / "user")
    monkeypatch.setattr(library_reads, "UNSEEN_SAID", False)
    if RECORDED:
        monkeypatch.setenv("CASH_TRACE", "0")
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            f, g = c.cache(through_global), c.cache(classmethod_through_global)
            assert f(7) == 7
            assert g(7) == 7
            lib.Model.scale = 5
            # Not seen without a recording: served by the key, as said.
            assert f(7) == 7
        said = [w for w in seen if "KEY-LIBRARY-STATE-UNSEEN" in str(w.message)]
        assert len(said) == 1
        assert "an attribute set on a library's class" in str(said[0].message)
    finally:
        _reset(lib)
        library_versions.forget()
