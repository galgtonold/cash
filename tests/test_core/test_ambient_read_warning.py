"""Reading the clock/environment/cwd inside a cached function is announced.

Round-15 gate finding. A decorated function whose body called
``datetime.now()`` cached the first call's timestamp and handed it back for
ever -- across processes, because the cache is on disk -- and said nothing.
``os.environ["TENANT"]`` did the same, which is the version that returns one
tenant's numbers to another tenant and exits 0.

The behaviour is unchanged and deliberate: freezing is what a cache does to any
input it cannot see. What was missing is the sentence saying so. These reads are
INPUTS, not side effects, so they get their own code (``KEY-AMBIENT-READ``) and
their own advice -- pass the value in as an argument, where it reaches the key
-- rather than the "audit these lines for writes" text of
``IMPURE-SIDE-EFFECTS``, which sends the reader looking for a write that is not
there.

Not routed through the unseeded-randomness detector, whose whole mechanism is a
seed ledger: no ``seed()`` makes ``datetime.now()`` reproducible.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime

import pytest

from cash import Cash
from cash.exceptions import CashImpureFunctionError, CashImpurityWarning
from cash.purity_analyzer import ISSUE_AMBIENT_READ, PurityAnalyzer


@pytest.fixture
def cash_instance(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _warnings_for(cash_instance, fn_factory):
    """Decorate and call once, returning every CashImpurityWarning raised."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn_factory(cash_instance)()
    return [w for w in rec if issubclass(w.category, CashImpurityWarning)]


def _kinds(fn) -> list[str]:
    return [i.kind for i in PurityAnalyzer().analyze(fn).issues]


# --- the reads themselves -------------------------------------------------

def _clock(c):
    @c.cache
    def stamp():
        return datetime.now().year
    return stamp


def _today(c):
    @c.cache
    def stamp():
        return date.today().year
    return stamp


def _env_subscript(c):
    import os

    @c.cache
    def which_tenant():
        return os.environ["PATH"][:1]
    return which_tenant


def _env_getenv(c):
    import os

    @c.cache
    def which_tenant():
        return os.getenv("PATH", "")[:1]
    return which_tenant


def _cwd(c):
    import os

    @c.cache
    def where():
        return len(os.getcwd())
    return where


def _fresh_uuid(c):
    import uuid

    @c.cache
    def ident():
        return str(uuid.uuid4())
    return ident


def _wall_clock(c):
    import time

    @c.cache
    def t():
        return int(time.time())
    return t


@pytest.mark.parametrize("factory", [
    _clock, _today, _env_subscript, _env_getenv, _cwd, _fresh_uuid, _wall_clock,
], ids=["datetime.now", "date.today", "os.environ[]", "os.getenv",
        "os.getcwd", "uuid4", "time.time"])
def test_each_ambient_read_is_announced(cash_instance, factory):
    got = _warnings_for(cash_instance, factory)
    assert got, "the ambient read was cached with no warning at all"
    text = "\n".join(str(w.message) for w in got)
    assert "KEY-AMBIENT-READ" in text, f"wrong diagnostic code:\n{text}"


def test_the_advice_is_about_the_key_not_about_side_effects(cash_instance):
    """The wrong text sends the reader hunting for a write that is not there."""
    text = "\n".join(str(w.message) for w in _warnings_for(cash_instance, _clock))
    assert "IMPURE-SIDE-EFFECTS" not in text
    assert "argument" in text, f"the fix must name passing it in:\n{text}"


def test_the_frozen_value_is_what_the_warning_is_about(cash_instance, monkeypatch):
    """The hazard itself, so the warning is pinned to a real failure.

    Without this the tests above pass on a warning that describes nothing.
    The environment is the right instrument here: a captured dict does NOT
    reproduce it -- a closure capture reaches the cache key, so changing one
    correctly recomputes. Only a value cash cannot see freezes, which is the
    whole point of the warning.
    """
    import os

    monkeypatch.setenv("CASH_TEST_TENANT", "acme")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        @cash_instance.cache
        def bill(rows):
            return f"{os.environ['CASH_TEST_TENANT']}:{rows}"

        first = bill(3)
        monkeypatch.setenv("CASH_TEST_TENANT", "globex")
        second = bill(3)

    assert first == "acme:3"
    assert second == first, "the frozen read is the documented hazard"


# --- controls -------------------------------------------------------------

def test_a_pure_body_says_nothing(cash_instance):
    def factory(c):
        @c.cache
        def add():
            return 1 + 1
        return add

    assert _warnings_for(cash_instance, factory) == []


def test_a_method_named_now_on_the_users_own_object_is_not_this(cash_instance):
    """Matched dotted, always: `self.clock.now()` is not `datetime.now()`."""
    class Clock:
        def now(self):
            return 7

    def factory(c):
        clock = Clock()

        @c.cache
        def read():
            return clock.now()
        return read

    text = "\n".join(str(w.message) for w in _warnings_for(cash_instance, factory))
    assert "KEY-AMBIENT-READ" not in text, f"false positive:\n{text}"


def test_an_argument_carrying_the_time_is_silent(cash_instance):
    """The fix the warning recommends must actually silence it."""
    def factory(c):
        @c.cache
        def report(as_of):
            return as_of.year
        return lambda: report(date.today())

    text = "\n".join(str(w.message) for w in _warnings_for(cash_instance, factory))
    assert "KEY-AMBIENT-READ" not in text, f"the recommended fix still warns:\n{text}"


def test_writing_the_environment_is_not_an_ambient_read():
    """`os.environ[k] = v` is a side effect, a different issue with a different
    fix; the Load-context guard is what keeps them apart."""
    import os

    def writes():
        os.environ["CASH_TEST_AMBIENT"] = "1"

    assert ISSUE_AMBIENT_READ not in _kinds(writes)


# --- waivers --------------------------------------------------------------

def test_assume_safe_flag_silences_it(cash_instance):
    def factory(c):
        @c.cache(assume_safe=True)
        def stamp():
            return datetime.now().year
        return stamp

    assert _warnings_for(cash_instance, factory) == []


def test_strict_mode_raises_on_it(cash_instance):
    with pytest.raises(CashImpureFunctionError) as exc:
        @cash_instance.cache(strict=True)
        def stamp():
            return datetime.now().year
        stamp()
    assert "ambient" in str(exc.value).lower()
