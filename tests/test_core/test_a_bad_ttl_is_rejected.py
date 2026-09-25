"""``ttl=`` is checked when the function is decorated.

The value is written into every entry's metadata and compared on every later
lookup. A ``ttl="300"`` read from an environment variable or a YAML file was
accepted, stored with the entry, and then raised ``TypeError`` out of every
lookup of that entry, in later runs too, even after the decorator was fixed.
"""

from __future__ import annotations

import datetime
import time

import pytest

from cash.core import Cash
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def clock(monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


@pytest.mark.parametrize("bad", ["300", [1], True, b"5"])
def test_a_ttl_that_is_not_a_number_raises_at_decoration(cash_instance, bad):
    with pytest.raises(TypeError, match="ttl must be a number of seconds"):

        @cash_instance.cache(ttl=bad)
        def f(x):
            return x


@pytest.mark.parametrize("bad", [-1, -0.5, float("nan"), float("inf")])
def test_a_negative_or_non_finite_ttl_raises_at_decoration(cash_instance, bad):
    with pytest.raises(ValueError, match="finite number of seconds >= 0"):

        @cash_instance.cache(ttl=bad)
        def f(x):
            return x


def test_a_str_ttl_names_the_fix(cash_instance):
    with pytest.raises(TypeError, match=r"int\(\.\.\.\)"):
        cash_instance.cache(ttl="300")


@pytest.mark.parametrize("good", [None, 0, 5, 0.5])
def test_numbers_and_none_are_accepted(cash_instance, good):
    @cash_instance.cache(ttl=good)
    def f(x):
        return x

    assert f(2) == 2


def test_a_timedelta_ttl_is_its_seconds(cash_instance, clock):
    runs = []

    @cash_instance.cache(ttl=datetime.timedelta(minutes=5))
    def fetch(x):
        runs.append(x)
        return x

    fetch(1)
    clock[0] += 299
    fetch(1)
    assert runs == [1], "inside five minutes the entry is served"
    clock[0] += 2
    fetch(1)
    assert runs == [1, 1], "after five minutes it is recomputed"


def _disk(c):
    return c.backend.backends[-1]


def rates(x):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)  # past the persistence floor: reaches disk
    return x * 2


def _entry_with_ttl(cache_dir, stored_ttl):
    """Run once with a good ttl, then rewrite the stored ttl as an older
    version (or a hand-edited cache) could have left it."""
    c = Cash(cache_dir=str(cache_dir), register_magic=False)
    c.cache(ttl=300)(rates)(1)
    backend = _disk(c)
    backend._writes.wait_all()
    [entry] = backend.list_entries()
    metadata, value = backend.get(entry["key"])
    backend.set(entry["key"], value, {**metadata, "ttl": stored_ttl})
    backend._writes.wait_all()
    c.backend.shutdown()


@pytest.mark.parametrize("stored", ["300", [1], float("nan")])
def test_an_entry_stored_with_a_bad_ttl_is_recomputed_not_raised(tmp_path, stored):
    _entry_with_ttl(tmp_path / "c", stored)
    c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
    cached = c.cache(ttl=300)(rates)

    assert cached(1) == 2  # raised TypeError before
    info = cached.cache_info()
    assert (info["hits"], info["misses"]) == (0, 1), "an entry whose ttl cannot be read is treated as expired"
    assert cached(1) == 2
    assert cached.cache_info()["hits"] == 1, "the rewritten entry carries a good ttl and is served"
    c.backend.shutdown()


def test_an_entry_with_a_good_ttl_is_served_by_the_next_run(tmp_path):
    """Control for the test above: the same steps with a readable ttl hit."""
    _entry_with_ttl(tmp_path / "c", 300)
    c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
    cached = c.cache(ttl=300)(rates)
    assert cached(1) == 2
    assert cached.cache_info()["hits"] == 1
    c.backend.shutdown()


def test_the_cli_reads_an_entry_with_a_bad_ttl_as_expired(tmp_path, monkeypatch, capsys):
    from cash.__main__ import _clear_expired, _scan_entries

    monkeypatch.delenv("CASH_TIER_0_TYPE", raising=False)
    _entry_with_ttl(tmp_path / "c", [1])
    [entry] = _scan_entries(tmp_path / "c")  # raised TypeError before
    assert entry.expires is not None
    _clear_expired(str(tmp_path / "c"))
    assert "Cleared 1 expired entry" in capsys.readouterr().out
    assert _scan_entries(tmp_path / "c") == []
