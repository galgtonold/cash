"""``CASH_PHASE_TIMES`` splits each cached call's time into the phases cash
spent it in (`cash.phase_times`): the phases of a call add up to the call,
a hit runs no body, a call cash refuses shows what it paid before running
uncached, and ``CASH_PHASE_TIMES_FILE`` collects every process's times
for ``tools/phase_report.py``."""

from __future__ import annotations

import threading
import time
import warnings

import pytest

from cash import Cash, phase_times
from tools import phase_report

_NOT_PHASES = ("total", "cash")


@pytest.fixture
def timed(monkeypatch, tmp_path):
    monkeypatch.setattr(phase_times, "ON", True)
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _phases(info: dict, outcome: str) -> dict[str, dict]:
    prefix = outcome + "."
    return {k[len(prefix) :]: v for k, v in info["phase_times"].items() if k.startswith(prefix)}


def test_the_phases_of_a_call_add_up_to_the_time_it_took(timed):
    @timed.cache
    def nap(x):
        time.sleep(0.05)
        return x

    started = time.perf_counter()
    nap(1)
    took = time.perf_counter() - started

    miss = _phases(nap.cache_info(), "miss")
    total = miss["total"]["total"]
    parts = sum(v["total"] for k, v in miss.items() if k not in _NOT_PHASES)
    assert parts == pytest.approx(total, rel=1e-9, abs=1e-9)
    # The wrapper's clock starts on entry and stops before the counting.
    assert 0.9 * took <= total <= took
    # A 50 ms sleep can end a millisecond early on Windows (3.10 waits in
    # whole milliseconds of the system tick; perf_counter is finer).
    assert miss["body"]["total"] >= 0.05 * 0.95
    assert miss["cash"]["total"] == pytest.approx(total - miss["body"]["total"])
    for phase in ("setup", "key.code", "key.args", "read.fetch", "checks", "body_setup", "post", "store"):
        assert miss[phase]["count"] == 1, phase


def test_a_hit_has_no_body_time(timed):
    @timed.cache
    def double(x):
        return 2 * x

    for _ in range(5):
        double(21)

    info = double.cache_info()
    times = info["phase_times"]
    hits = sum(times[f"{o}.total"]["count"] for o in ("hit", "fast_hit") if f"{o}.total" in times)
    assert hits == info["hits"] == 4
    assert times["miss.total"]["count"] == 1
    assert "hit.body" not in times and "fast_hit.body" not in times
    for outcome in ("hit", "fast_hit"):
        if f"{outcome}.total" in times:
            assert times[f"{outcome}.cash"]["total"] == pytest.approx(times[f"{outcome}.total"]["total"])


def test_a_call_with_no_key_shows_what_it_paid_before_running(timed):
    @timed.cache
    def locked(lock, x):
        with lock:
            return x + 1

    with pytest.warns(Warning):
        assert locked(threading.Lock(), 1) == 2

    uncached = _phases(locked.cache_info(), "uncached")
    assert uncached["total"]["count"] == 1
    paid_before = sum(v["total"] for k, v in uncached.items() if k.startswith(("setup", "key.")))
    assert paid_before > 0
    assert uncached["body"]["count"] == 1
    assert "store" not in uncached and "read.fetch" not in uncached


def test_a_call_refused_after_its_body_is_counted_apart(timed):
    @timed.cache
    def grow(items):
        items.append(1)
        return {"held": items}

    with pytest.warns(Warning):
        grow([1, 2])

    times = grow.cache_info()["phase_times"]
    assert times["not_stored.total"]["count"] == 1
    assert times["not_stored.post"]["total"] > 0
    assert not any(k.startswith("miss.") for k in times)


class _Slotted:
    __slots__ = ("log",)

    def __init__(self):
        self.log = []


def test_a_call_run_by_its_refusal_verdict_counts_its_body_as_body(timed):
    """A call an earlier refusal runs uncached without a key
    (`refusal_verdicts`) still has its body timed as ``body``: counted with
    the wrapper, the body of every such call was reported as cash's own."""

    @timed.cache
    def fill(box):
        box.log.append(1)  # a hit cannot put this back: slots
        time.sleep(0.02)
        return len(box.log)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(5):
            fill(_Slotted())

    uncached = _phases(fill.cache_info(), "uncached")
    assert uncached["total"]["count"] >= 2
    assert uncached["body"]["count"] == uncached["total"]["count"]
    assert uncached["body"]["total"] >= 0.02 * uncached["total"]["count"]
    assert uncached["cash"]["total"] < uncached["body"]["total"]


def test_cache_clear_forgets_the_phase_times(timed):
    @timed.cache
    def one():
        return 1

    one()
    one.cache_clear()
    assert one.cache_info()["phase_times"] == {}


def test_off_nothing_is_timed(monkeypatch, tmp_path):
    monkeypatch.setattr(phase_times, "ON", False)
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache
    def one():
        return 1

    one()
    one()
    assert one.cache_info()["phase_times"] == {}


_SCRIPT = """
import cash

@cash.cache
def square(x):
    return x * x

for i in range(4):
    square(i % 2)
"""


def test_the_file_collects_every_process_and_the_report_adds_them_up(tmp_path):
    from tests._scripts import run_python

    (tmp_path / "work.py").write_text(_SCRIPT, encoding="utf-8")
    out = tmp_path / "times"
    out.mkdir()
    for _ in range(2):
        run_python("work.py", cwd=tmp_path, env={"CASH_PHASE_TIMES_FILE": str(out) + "/"})

    assert len(list(out.glob("phase-times-*.jsonl"))) == 2
    merged, processes = phase_report.load([str(out)])
    assert processes == 2
    stats = merged[("__main__", "square")]
    row = phase_report.row(stats)
    # First process: two misses, two hits; second: four hits.
    assert (row["calls"], row["misses"], row["hits"], row["refused"]) == (8, 2, 6, 0)
    assert row["body_s"] > 0 and row["overhead_s"] > 0
    assert len(row["top_phases"]) == 3
    table = phase_report.table(merged)
    assert "__main__.square" in table

    # One file, appended to by both processes.
    single = tmp_path / "all.jsonl"
    for _ in range(2):
        run_python("work.py", cwd=tmp_path, env={"CASH_PHASE_TIMES_FILE": str(single)})
    assert len(single.read_text(encoding="utf-8").splitlines()) == 2
    assert phase_report.main([str(single)]) == 0
