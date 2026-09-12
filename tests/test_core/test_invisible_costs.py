"""What caching costs is shown next to what it saves (round 19).

A cached parser returned two million tuples to two cached consumers: warm runs
were 8.5-9x SLOWER than uncached while the summary reported time saved.
``frozen=True`` on the parser changed nothing -- its fast path was numpy-only,
with no way to see that. CACHE-NET-LOSS never fired in a command-line tool
that calls each function once per process, and when it did fire it said the
list took "about 0ms to hash".
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash
from cash.core import _ARG_COST
from cash.effectiveness import EffectivenessLedger

pytestmark = [pytest.mark.core]


def _rows(n):
    return [(i, i * 2, f"x{i % 100}") for i in range(n)]


def test_a_frozen_list_is_keyed_by_its_producer_not_its_contents(tmp_path, monkeypatch):
    import cash.core as core
    c = Cash(cache_dir=str(tmp_path / "cache"))
    calls = []

    @c.cache(frozen=True)
    def parse(n):
        return _rows(n)

    @c.cache
    def total(rows):
        calls.append(1)
        return sum(r[0] for r in rows)

    rows = parse(200_000)
    assert total(rows) == total(rows)
    assert len(calls) == 1
    # Counted, not timed (a timed bound flaked under a loaded -n 16 run): the
    # hit neither pickles the list nor walks it.
    dumped = []
    real_dumps = core.pickle.dumps
    monkeypatch.setattr(core.pickle, "dumps", lambda obj, *a, **k:
                        dumped.append(obj) or real_dumps(obj, *a, **k))
    total(rows)
    payloads = [x for x in dumped if isinstance(x, tuple) and len(x) == 2
                and isinstance(x[0], tuple)]
    assert payloads, "no key was hashed"
    assert not any(a is rows for x in payloads for a in x[0]),         "a hit still serialized the whole list"


def test_a_frozen_list_keys_the_same_in_the_next_process(tmp_path):
    """The identity is the producer's lineage, which is the same in every
    process that produces or restores that result."""
    job = tmp_path / "job.py"
    job.write_text(textwrap.dedent("""
        import sys, time
        import cash

        @cash.cache(frozen=True)
        def parse(n):
            time.sleep(0.15)
            return [(i, str(i)) for i in range(n)]

        @cash.cache
        def total(rows):
            print("[RUN] total", file=sys.stderr)
            time.sleep(0.15)
            return sum(r[0] for r in rows)

        print(total(parse(1000)))
    """), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    runs = [subprocess.run([sys.executable, str(job)], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=120) for _ in range(2)]
    assert [r.stdout.strip() for r in runs] == ["499500", "499500"], runs[0].stderr[-2000:]
    assert "[RUN] total" in runs[0].stderr
    assert "[RUN] total" not in runs[1].stderr, "the consumer missed in the next process"


def test_a_frozen_list_that_is_modified_is_noticed(tmp_path, monkeypatch):
    monkeypatch.setenv("CASH_DEBUG", "1")                  # audit on every use
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache(frozen=True)
    def parse(n):
        return [[i] for i in range(n)]

    @c.cache
    def total(rows):
        return sum(r[0] for r in rows)

    rows = parse(10)
    assert total(rows) == 45
    rows[0][0] = 100
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert total(rows) == 145, "the result for the unmodified list was served"
    assert any("KEY-FROZEN-MUTATED" in str(w.message) for w in rec)


def test_frozen_on_a_result_it_cannot_mark_says_so(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache(frozen=True)
    def labels(n):
        return {f"x{i}" for i in range(n)}

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        labels(3)
    assert any("KEY-FROZEN-NO-EFFECT" in str(w.message) and "set" in str(w.message)
               for w in rec)


def test_the_payload_cost_of_a_list_argument_is_charged_to_it(tmp_path):
    """The list went into the payload as is: the per-argument timer saw a
    lookup that returned at once and blamed 0ms."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    c._hash_arg_payload((_rows(100_000), 3), {})
    label, seconds, type_name, _producer, _old = _ARG_COST.last
    assert (label, type_name) == ("#0", "list")
    assert seconds > 0.005


def test_a_one_call_net_loss_is_reported_at_the_end_of_the_run():
    ledger = EffectivenessLedger(waste_threshold_seconds=2.0)
    culprit = ("rows", "list", 6.0, "app.parse", False)
    assert ledger.record("app.total", overhead_seconds=6.0, body_seconds=0.01,
                         was_hit=True, culprit=culprit) is None      # one call: record waits
    verdicts = ledger.final_verdicts()
    assert len(verdicts) == 1
    what, fix = verdicts[0]
    assert "app.total" in what and "'rows'" in what
    assert ledger.final_verdicts() == [], "said twice"


def test_a_worthwhile_single_call_is_not_reported():
    """Control: a hit that saved more than it cost is not a loss."""
    ledger = EffectivenessLedger(waste_threshold_seconds=2.0)
    ledger.record("app.slow", overhead_seconds=0.2, body_seconds=30.0, was_hit=True)
    assert ledger.final_verdicts() == []


def test_the_hit_line_and_the_summary_show_what_the_lookup_cost(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    line = c._describe_call({"func_name": "app.total", "cache_hit": True, "time_saved": 0.01,
                             "execution_time": 1.07, "cache_key": "app.total:s::a"})
    assert "the lookup took 1.07s; a net loss" in line
    quiet = c._describe_call({"func_name": "app.slow", "cache_hit": True, "time_saved": 9.0,
                              "execution_time": 0.002, "cache_key": "app.slow:s::a"})
    assert "lookup" not in quiet

    # The table, from a known account rather than a timed workload: a real
    # lookup's time depends on the machine, and got fast enough to pass under
    # the bar once hashing a big list stopped walking it.
    from collections import Counter
    c._function_stats["app.total"] = {
        "hits": 2, "misses": 0, "total_time_saved": 0.02, "lookup_seconds": 2.1,
        "miss_reasons": Counter(), "not_persisted": Counter(), "not_stored": Counter(),
        "bypassed": 0}
    summary = c.run_summary()
    assert "spent by cash on keys, lookups and stores, a net loss of 2.1s" in summary, summary
    assert "2.1s spent by cash" in summary, summary

    # Round 20 (r20s2): a function that never hit cost its keys and stores on
    # every miss, and the summary said nothing about it.
    c._function_stats["app.parse"] = {
        "hits": 0, "misses": 3, "total_time_saved": 0.0, "lookup_seconds": 0.0,
        "miss_overhead_seconds": 3.0, "miss_reasons": Counter({"no entry yet": 3}),
        "not_persisted": Counter(), "not_stored": Counter(), "bypassed": 0}
    summary = c.run_summary()
    assert "a net loss of 5.1s" in summary, summary
    assert "3.0s spent by cash" in summary, summary


# -- round 20: what the numbers were wrong about ------------------------------

def test_hits_on_several_threads_are_not_counted_as_serial_savings(tmp_path):
    """r20s1: sixteen 0.5 s calls on eight threads claimed 8.0 s saved; the
    warm run saved 1.0 s. A hit's saving is divided by the threads running
    cached calls with it."""
    from concurrent.futures import ThreadPoolExecutor

    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def slow(i):
        time.sleep(0.2)
        return i * i

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(slow, range(16)))
    before = slow.cache_info()["total_time_saved"]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(slow, range(16)))
    saved = slow.cache_info()["total_time_saved"] - before
    assert 0 < saved < 0.75 * 16 * 0.2, f"claimed {saved:.2f}s for 16 x 0.2 s on 8 threads"


def test_a_nested_cached_calls_overhead_is_not_the_outer_functions_saving(tmp_path):
    """r20s3: an inner cached call over 200k rows cost 12x its body to key, and
    that time was booked as the OUTER function's run -- its hits then claimed
    to save it, 4-9x what running it uncached costs."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    rows = [(i, {"k": i}) for i in range(200_000)]          # a dict per row: the slow path

    @c.cache
    def inner(rows, n):
        return len(rows) + n

    @c.cache
    def outer(n):
        return sum(inner(rows, i) for i in range(3))

    t0 = time.perf_counter()
    outer(1)
    first = time.perf_counter() - t0
    outer(1)                                                   # a hit
    saved = outer.cache_info()["total_time_saved"]
    assert saved < 0.3 * first, f"outer's hit claimed {saved:.2f}s of a {first:.2f}s first call"


def test_small_losses_across_functions_are_reported_together():
    """r20s2: ten functions each losing 0.4-0.9 s never crossed the 2 s a single
    warning waits for, and the run was 1.3x slower than no cache at all."""
    ledger = EffectivenessLedger(waste_threshold_seconds=2.0)
    for name in ("app.max_latency", "app.coupons", "app.parse_users"):
        ledger.record(name, overhead_seconds=0.9, body_seconds=0.01, was_hit=False)
    verdicts = ledger.final_verdicts()
    assert len(verdicts) == 1, verdicts
    what, fix = verdicts[0]
    assert "across 3 functions" in what and "app.coupons" in what, what
    assert "a net loss of about 2.7s" in what, what


def test_a_function_that_never_hit_is_not_said_to_be_slow_to_load():
    """r20s2 F8: 'it is loading the stored result' about a function whose
    calls were all misses -- nothing was loaded; keeping the result was the
    cost."""
    ledger = EffectivenessLedger(waste_threshold_seconds=2.0)
    culprit = ("path", "str", 0.0001, None, False)
    ledger.record("app.parse", overhead_seconds=2.5, body_seconds=0.6, was_hit=False, culprit=culprit)
    (what, _fix), = ledger.final_verdicts()
    assert "keeping the result" in what and "loading" not in what, what
