"""Round 20: advisories that fired on ordinary code, every run.

* A per-key accumulator in a LOCAL dict -- ``by_user[k].append(x)``,
  ``acc = out.get(k); acc[0] += 1``, ``for u, stamps in by_user.items():
  stamps.sort()`` -- was a "side effect" (r20s2, r20s5).
* ``row[3] = float(row[3])`` on a row ``csv.reader`` just produced (r20s2).
* A progress ``print(..., file=sys.stderr)`` or a ``logger.info`` was an
  IMPURE-SIDE-EFFECTS on every function that called the log helper (r20s1,
  r20s2, r20s5). A hit skipping a log line is what caching means.
* ``perf_counter()`` returned by a ``mark()`` helper and handed only to a
  ``done()`` that prints it was a KEY-AMBIENT-READ (r20s5).
* ``con.execute("SELECT ...")`` was a "write method" (r20s4).

Each has a control that must still warn.
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict

import pytest

from cash.purity_analyzer import PurityAnalyzer

pytestmark = pytest.mark.core

log = logging.getLogger(__name__)


def _issues(fn):
    return list(PurityAnalyzer().analyze(fn).issues)


def _effects(fn):
    return [i.description for i in _issues(fn) if i.kind in ("scope_mutation", "impure_call")]


def _ambient(fn):
    return [i.description for i in _issues(fn) if i.kind == "ambient_read"]


# -- local accumulators --------------------------------------------------------

def sessionize(events):
    by_user = defaultdict(list)
    for e in events:
        by_user[e["user"]].append(e["ts"])
    out = {}
    for user, stamps in by_user.items():
        stamps.sort()
        out[user] = len(stamps)
    return out


def station_stats(rows):
    out = {}
    for station, value in rows:
        acc = out.get(station)
        if acc is None:
            acc = [0, 0.0]
            out[station] = acc
        acc[0] += 1
        acc[1] += value
    return out


def latency_percentiles(rows):
    by_path = {}
    for path, ms in rows:
        by_path.setdefault(path, []).append(ms)
    for lats in by_path.values():
        lats.sort()
    return {p: lats[len(lats) // 2] for p, lats in by_path.items()}


def latency_by_path_sorted(log_rows):
    """r20s2's own shape: the items iterated through sorted()."""
    by_path = defaultdict(list)
    for r in log_rows:
        by_path[r[3]].append(r[5])
    out = {}
    for p, lats in sorted(by_path.items()):
        lats.sort()
        out[p] = lats[len(lats) // 2]
    return out


def parse_orders(path):
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            row[3] = float(row[3])
            out.append(row)
    return out


def parse_users(path):
    with open(path, newline="") as f:
        users = []
        for d in csv.DictReader(f):
            d["age"] = int(d["age"])
            users.append(d)
    return users


@pytest.mark.parametrize("fn", [sessionize, station_stats, latency_percentiles, latency_by_path_sorted,
                                parse_orders, parse_users])
def test_a_local_accumulator_is_not_a_side_effect(fn):
    assert _effects(fn) == [], _effects(fn)


# Controls: the same shapes reaching the CALLER's objects.

def element_holds_an_argument(rows):
    d = {"a": rows}
    d["a"].append(1)
    return d


def sorts_elements_of_an_argument(groups):
    for g in groups.values():
        g.sort()
    return groups


def accumulates_into_an_argument(rows, out):
    for k, v in rows:
        acc = out.get(k)
        acc[0] += v
    return out


def rows_of_a_list_argument(rows):
    for row in rows:
        row[3] = float(row[3])
    return rows


@pytest.mark.parametrize("fn", [element_holds_an_argument, sorts_elements_of_an_argument,
                                accumulates_into_an_argument, rows_of_a_list_argument])
def test_the_same_shapes_on_the_callers_objects_still_warn(fn):
    assert _effects(fn), f"{fn.__name__} mutates the caller's data and was not reported"


# -- log lines -------------------------------------------------------------------

def _log(msg):
    print(msg, file=sys.stderr)


def logs_progress(rows):
    _log(f"parsing {len(rows)} rows")
    print("still going", file=sys.stderr, flush=True)
    log.info("done with %d", len(rows))
    sys.stderr.write("ok\n")
    return len(rows)


def prints_its_output(rows):
    print(len(rows))
    return len(rows)


def test_a_log_line_is_not_a_side_effect():
    assert _effects(logs_progress) == [], _effects(logs_progress)


def test_a_print_to_stdout_is_still_reported():
    """Control: stdout may be the program's output, which a hit would drop."""
    assert _effects(prints_its_output), "print() to stdout was not reported"


# -- a timing helper -------------------------------------------------------------

def mark(name):
    print(f"@@RUN {name}", file=sys.stderr, flush=True)
    return time.perf_counter()


def done(name, t0):
    print(f"@@DONE {name} {time.perf_counter() - t0:.2f}s", file=sys.stderr, flush=True)


def timed_step(x):
    t0 = mark("step")
    y = x * 2
    done("step", t0)
    return y


def returns_the_marked_time(x):
    t0 = mark("step")
    return x, t0


def test_a_clock_read_that_only_reaches_a_timing_print_is_not_ambient():
    assert _ambient(timed_step) == [], _ambient(timed_step)


def test_a_clock_read_that_reaches_the_result_still_is():
    said = _ambient(returns_the_marked_time)
    assert said and "mark()" in said[0] and "perf_counter" in said[0], said


def test_the_clock_helper_is_still_part_of_the_key():
    """Judged at the call site, but still walked: an edit to it must invalidate."""
    report = PurityAnalyzer().analyze(timed_step)
    assert any(q.endswith("mark") for q in report.helper_source_hashes), report.helper_source_hashes


# -- a test's fake behind autospec ----------------------------------------------

def _lookup(x):
    return x + 1


def uses_lookup(x):
    return _lookup(x) * 2


def test_an_autospec_fakes_side_effect_is_not_analysed_as_production_code(tmp_path):
    """r20s1: `patch(..., autospec=True, side_effect=fake)` walked into the fake,
    and an `__import__` in it raised CashImpureFunctionError out of the test."""
    from unittest import mock

    from cash import Cash

    def fake(x):
        return int(__import__("os").environ.get("FAKE_VALUE", "5"))

    c = Cash(cache_dir=str(tmp_path / "c"))
    cached = c.cache(uses_lookup)
    with mock.patch(f"{__name__}._lookup", autospec=True, side_effect=fake):
        assert cached(1) == 10
    assert cached.cache_info()["miss_reasons"] == {"a helper is a mock": 1}


# -- where a pool's warning points ---------------------------------------------

def announces(x):
    print("computing", x)       # stdout: a static finding
    return x


def test_a_warning_from_a_pool_thread_names_the_users_file(tmp_path):
    """r20s1: first called in a ThreadPoolExecutor, the warning blamed
    concurrent/futures/thread.py."""
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    from cash import Cash

    c = Cash(cache_dir=str(tmp_path / "c"))
    cached = c.cache(announces)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(cached, [1]))
    found = [w for w in rec if "IMPURE-SIDE-EFFECTS" in str(w.message)]
    assert found, [str(w.message) for w in rec]
    assert os.path.normcase(found[0].filename) == os.path.normcase(__file__), found[0].filename


# -- once per version, not once per process -----------------------------------

def reports_to_stdout(x):
    print("row", x)
    return x


def also_reports_to_stdout(x):
    print("row", x)
    print("again", x)
    return x


def _impure_shown(c, fn, arg):
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)(arg)
    return [w for w in rec if "IMPURE-SIDE-EFFECTS" in str(w.message)]


def test_a_static_finding_is_shown_once_per_cache_not_every_run(tmp_path):
    """r20s1: 32 warning lines in a nightly job's log, every night."""
    from cash import Cash
    first = Cash(cache_dir=str(tmp_path / "c"))
    assert _impure_shown(first, reports_to_stdout, 1)
    first.shutdown()                                   # the end of that run
    later = Cash(cache_dir=str(tmp_path / "c"))        # the next run
    assert not _impure_shown(later, reports_to_stdout, 2)
    logged = later._func_warnings.get(f"{__name__}.reports_to_stdout") or []
    assert any(e.get("code") == "IMPURE-SIDE-EFFECTS" for e in logged), logged


def test_a_different_finding_is_shown_again(tmp_path):
    """Control: new findings are new text, and new text is shown."""
    from cash import Cash
    first = Cash(cache_dir=str(tmp_path / "c"))
    assert _impure_shown(first, reports_to_stdout, 1)
    first.shutdown()
    assert _impure_shown(Cash(cache_dir=str(tmp_path / "c")), also_reports_to_stdout, 1)


# -- a clock test double ---------------------------------------------------------
# freezegun is not a dependency; these stand-ins do the two things it does that
# reached cash: `time.perf_counter` replaced by a frozen function, and dates
# made under the freeze being instances of `freezegun.api.FakeDate`.

def slow_report(when):
    time.sleep(0.25)  # @cash:assume-safe
    return str(when)


def test_a_frozen_perf_counter_does_not_freeze_cashs_own_timer(tmp_path, monkeypatch):
    """r20s3: under freeze_time every body "ran 0.00s" and nothing was stored."""
    from cash import Cash

    c = Cash(cache_dir=str(tmp_path / "c"))
    cached = c.cache(slow_report)
    frozen = time.perf_counter()
    monkeypatch.setattr(time, "perf_counter", lambda: frozen)
    cached("2026-09-30")
    c.backend._writes.wait_all() if hasattr(c.backend, "_writes") else None
    for tier in getattr(c.backend, "backends", [c.backend]):
        if hasattr(tier, "_writes"):
            tier._writes.wait_all()
    assert list((tmp_path / "c").glob("*.entry")), "a 0.25 s body was judged too cheap to store"


def test_a_fake_date_is_keyed_as_the_date(tmp_path, monkeypatch):
    """r20s3: a `date` global made under freezegun re-keyed every function
    reading it, although the value was equal."""
    import datetime
    import types

    from cash import Cash

    api = types.ModuleType("freezegun.api")

    # Built with type(): no source of its own, like the library class.
    FakeDate = type("FakeDate", (datetime.date,), {"__module__": "freezegun.api"})
    api.FakeDate = FakeDate
    monkeypatch.setitem(sys.modules, "freezegun.api", api)

    c = Cash(cache_dir=str(tmp_path / "c"))
    cached = c.cache(slow_report)
    cached(FakeDate(2025, 10, 1))
    cached(datetime.date(2025, 10, 1))
    assert cached.cache_info()["hits"] == 1, cached.cache_info()


# -- sqlite ----------------------------------------------------------------------

def rate(con: sqlite3.Connection, key):
    return con.execute("SELECT v FROM rates WHERE k = ?", (key,)).fetchone()[0]


def rate_with_cte(con, key):
    return con.execute("""
        WITH r AS (SELECT k, v FROM rates) SELECT v FROM r WHERE k = ?""", (key,)).fetchall()


def record_rate(con, key, value):
    con.execute("INSERT INTO rates VALUES (?, ?)", (key, value))
    return value


def test_a_select_is_not_a_write():
    assert _effects(rate) == [], _effects(rate)
    assert _effects(rate_with_cte) == [], _effects(rate_with_cte)


def test_an_insert_still_is():
    assert _effects(record_rate), "an INSERT was not reported"
