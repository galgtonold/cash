"""A helper is keyed by what the CALLER's name is bound to, not by where it came from.

Round 18 (r18s3): a test passed on another test's cached answer with the most
common mocking pattern. Helpers were re-resolved per call through the helper
OBJECT's home (``sievelib.sieve``), so rebinding the caller's name
(``primes._sieve = fake``, ``monkeypatch.setattr``, ``mock.patch``) changed
nothing the key could see; and the helper walk ran once per process, so
whatever was bound at the first call decided -- a helper mocked first stayed
"the helper" after the real one was restored.

Now each helper is re-resolved through the name its caller uses, at every
level of the helper tree; a binding that no longer holds the analysed object
re-analyses the tree below it; a cached function called through a rebound name
is treated the same way; and a binding to a mock -- which has no code to key
-- runs that call uncached.

Every arm compares the cached call with ``__wrapped__`` (the undecorated
function, resolving the same possibly-patched names) as the oracle.
"""
from __future__ import annotations

import importlib
import sys
import textwrap
import uuid
from unittest import mock

import pytest

import cash

pytestmark = pytest.mark.core


@pytest.fixture
def mods(tmp_path, monkeypatch):
    """Write modules under unique names and import them; returns a loader."""
    tag = "m" + uuid.uuid4().hex[:8]
    monkeypatch.syspath_prepend(str(tmp_path))
    created: list[str] = []

    def load(files: dict[str, str], main: str):
        for name, src in files.items():
            src = textwrap.dedent(src).replace("PFX_", f"{tag}_")
            (tmp_path / f"{tag}_{name}.py").write_text(src, encoding="utf-8")
            created.append(f"{tag}_{name}")
        importlib.invalidate_caches()
        return [importlib.import_module(f"{tag}_{n}") for n in main.split(",")]

    yield load
    for name in created:
        sys.modules.pop(name, None)


@pytest.fixture
def c(tmp_path):
    return cash.Cash(cache_dir=str(tmp_path / "cache"))


def _check(fn, *args):
    got, want = fn(*args), fn.__wrapped__(*args)
    assert got == want, f"cached {got!r}, undecorated {want!r}"
    return got


LIB = {
    "sievelib": """
        def sieve(n):
            return n + 1
    """,
}


def _primes(c, mods, body_import):
    lib, app = mods({**LIB, "app": f"""
        {body_import}
        def count(n):
            return _sieve(n) * 10
    """}, "sievelib,app")
    app.count = c.cache(app.count)
    return lib, app


def test_imported_alias_patched_after_a_real_call(c, mods):
    """r18s3 F12: `from sievelib import sieve as _sieve`, patch `app._sieve`."""
    _, app = _primes(c, mods, "from PFX_sievelib import sieve as _sieve")
    assert _check(app.count, 1) == 20
    with mock.patch.object(app, "_sieve", lambda n: -1):
        assert _check(app.count, 1) == -10
    assert _check(app.count, 1) == 20


def test_same_module_helper_mocked_first_then_restored(c, mods):
    """r18s3 F5: the first call's binding decided for the rest of the process."""
    (app,) = mods({"app": """
        def _sieve(n):
            return n + 1
        def count(n):
            return _sieve(n) * 10
    """}, "app")
    app.count = c.cache(app.count)
    with mock.patch.object(app, "_sieve", lambda n: -1):
        assert _check(app.count, 1) == -10
    assert _check(app.count, 1) == 20


def test_monkeypatch_setattr(c, mods, monkeypatch):
    _, app = _primes(c, mods, "from PFX_sievelib import sieve as _sieve")
    assert _check(app.count, 2) == 30
    monkeypatch.setattr(app, "_sieve", lambda n: 0)
    assert _check(app.count, 2) == 0


def test_a_patch_two_levels_down(c, mods):
    """The helper's helper, bound in the helper's module: `lvl1._h2`."""
    lvl1, app = mods({
        "lvl2": """
            def h2(x):
                return x + 1
        """,
        "lvl1": """
            from PFX_lvl2 import h2 as _h2
            def h1(x):
                return _h2(x) * 10
        """,
        "app": """
            from PFX_lvl1 import h1
            def f(x):
                return h1(x)
        """,
    }, "lvl1,app")
    app.f = c.cache(app.f)
    assert _check(app.f, 1) == 20
    with mock.patch.object(lvl1, "_h2", lambda x: -1):
        assert _check(app.f, 1) == -10
    assert _check(app.f, 1) == 20


def test_a_mock_runs_the_call_uncached(c, mods):
    """A MagicMock has no code to key: that call bypasses the cache entirely."""
    _, app = _primes(c, mods, "from PFX_sievelib import sieve as _sieve")
    assert _check(app.count, 1) == 20
    with mock.patch.object(app, "_sieve", return_value=5):
        assert _check(app.count, 1) == 50
    with mock.patch.object(app, "_sieve", return_value=7):
        assert _check(app.count, 1) == 70      # a second mock is not the first one's entry
    assert _check(app.count, 1) == 20


def test_a_mocked_call_says_why_it_missed(c, mods):
    """The miss reason and explain() name the mock instead of guessing."""
    _, app = _primes(c, mods, "from PFX_sievelib import sieve as _sieve")
    app.count(1)
    with mock.patch.object(app, "_sieve", return_value=5):
        explanation = app.count.explain(1)
        app.count(1)
    assert not explanation.would_hit
    assert "_sieve is a MagicMock" in explanation.details["hint"]
    reasons = app.count.cache_info()["miss_reasons"]
    assert any("a helper is a mock" in str(r) for r in reasons), reasons


def test_a_mock_two_levels_down_runs_uncached(c, mods):
    lvl1, app = mods({
        "lvl2": """
            def h2(x):
                return x + 1
        """,
        "lvl1": """
            from PFX_lvl2 import h2 as _h2
            def h1(x):
                return _h2(x) * 10
        """,
        "app": """
            from PFX_lvl1 import h1
            def f(x):
                return h1(x)
        """,
    }, "lvl1,app")
    app.f = c.cache(app.f)
    assert _check(app.f, 1) == 20
    with mock.patch.object(lvl1, "_h2", return_value=-1):
        assert _check(app.f, 1) == -10


def test_a_cached_callee_patched_in_the_callers_module(c, mods):
    """`outer` calls cached `inner` by name; patching `app.inner` must reach outer's key."""
    (app,) = mods({"app": """
        def inner(x):
            return x + 1
        def outer(x):
            return inner(x) * 10
    """}, "app")
    app.inner = c.cache(app.inner)
    app.outer = c.cache(app.outer)
    assert _check(app.outer, 1) == 20
    with mock.patch.object(app, "inner", lambda x: -1):
        assert _check(app.outer, 1) == -10
    assert _check(app.outer, 1) == 20


def test_restored_helper_brings_its_own_subtree_back(c, mods):
    """Analysed while mocked, then restored: the REAL helper's global reads must
    reach the key, so the tree below a changed binding is re-analysed."""
    (app,) = mods({"app": """
        FACTOR = 10
        def _scale(x):
            return x * FACTOR
        def f(x):
            return _scale(x)
    """}, "app")
    app.f = c.cache(app.f)
    with mock.patch.object(app, "_scale", lambda x: -1):
        assert _check(app.f, 1) == -1
    assert _check(app.f, 1) == 10
    app.FACTOR = 100
    assert _check(app.f, 1) == 100


def test_unpatched_calls_hit_and_a_restore_hits_the_original_entry(c, mods):
    """Controls: nothing patched -> one execution; patch and restore -> the
    original entry is hit again rather than recomputed."""
    lib, app = mods({**LIB, "app": """
        from PFX_sievelib import sieve as _sieve
        CALLS = []
        def count(n):
            CALLS.append(n)
            return _sieve(n) * 10
    """}, "sievelib,app")
    app.count = c.cache(app.count)
    app.count(1)
    app.count(1)
    assert app.CALLS == [1], "an unpatched second call did not hit"
    with mock.patch.object(app, "_sieve", lambda n: -1):
        app.count(1)
    assert app.CALLS == [1, 1]
    app.count(1)
    assert app.CALLS == [1, 1], "restoring the real helper did not return to its entry"


def test_identically_written_functions_in_two_modules_keep_their_own_helpers(c, mods):
    """Found while fixing the above: the analyzer cached its report by the
    function's source TEXT, so a second module's identically written `count`
    got the first module's helper tree. Redefining the second module's own
    `_sieve` then changed nothing its key could see."""
    one, two = mods({
        "one": """
            def _sieve(n):
                return n + 1
            def count(n):
                return _sieve(n) * 10
        """,
        "two": """
            def _sieve(n):
                return n + 2
            def count(n):
                return _sieve(n) * 10
        """,
    }, "one,two")
    one.count = c.cache(one.count)
    two.count = c.cache(two.count)
    assert _check(one.count, 1) == 20
    assert _check(two.count, 1) == 30
    exec("def _sieve(n):\n    return n + 5\n", two.__dict__)    # an in-process redefinition
    assert _check(two.count, 1) == 60


def test_the_helper_home_rebound_is_not_what_runs(c, mods):
    """Patching the helper's HOME module after the caller imported it does not
    change what the caller runs -- and must not change its answer either."""
    lib, app = _primes(c, mods, "from PFX_sievelib import sieve as _sieve")
    assert _check(app.count, 1) == 20
    with mock.patch.object(lib, "sieve", lambda n: -1):
        assert _check(app.count, 1) == 20
