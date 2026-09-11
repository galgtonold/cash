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


def test_a_frozen_list_is_keyed_by_its_producer_not_its_contents(tmp_path):
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
    t0 = time.perf_counter()
    total(rows)
    assert time.perf_counter() - t0 < 0.05, "a hit still walked the whole list"


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

    @c.cache
    def total(rows):
        return sum(r[0] for r in rows)

    rows = _rows(200_000)
    total(rows)
    total(rows)
    summary = c.run_summary()
    assert "spent on the hits' lookups" in summary, summary
