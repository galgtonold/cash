"""The cost model's constants have to agree with the measurements they came from.

`_COEFFS` exists to answer one question per value:

    persist if  execution_time - estimated_restore > min_cache_savings_pct * execution_time

so the thing worth pinning is not a coefficient but how often that answer is
wrong -- judged against the restore times in the matrix the constants were
fitted from, which is committed beside them.

Deterministic: it reads measured numbers off a CSV and compares decisions. It
times nothing.

It would have caught what round 26 found by hand. The constants shipped before
came from a matrix that timed each read while a write of the same key was still
in flight, through the backend that had just written the file, with a leaked
backend per cell still running its threads. The fitted intercept landed at
10.4 ms where a fresh-process read of a small entry measures ~5.6 ms, and 37 of
378 decisions came out wrong -- every one of them a value refused disk that had
earned its place, including a 100 MB frame from a 100 ms body whose restore
measures 70 ms.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cash.notebook import cost_model

#: config.min_cache_savings_pct
SAVINGS_PCT = 0.20
#: Five decades of body time, which is the range a decision is asked over.
BODY_SECONDS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

MATRIX = (Path(__file__).resolve().parents[2]
          / "benchmarks" / "results" / "ser_deser_matrix.frozen.csv")


def _persists(restore_seconds: float, body_seconds: float) -> bool:
    return body_seconds - restore_seconds > SAVINGS_PCT * body_seconds


def _measured_cells() -> list[tuple[str, float, float]]:
    if not MATRIX.exists():                      # a wheel install has no benchmarks/
        pytest.skip(f"measurement matrix not present at {MATRIX}")
    cells = []
    with open(MATRIX, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["error"] or row["backend_kind"] != "disk":
                continue
            cells.append((row["family"], float(row["actual_size_bytes"]),
                          float(row["deserialize_seconds"])))
    assert cells, "the matrix has no usable disk rows"
    return cells


def _score() -> tuple[int, int, int, list[str]]:
    """(total, kept_out, let_in, descriptions of the wrong ones)."""
    total = kept_out = let_in = 0
    wrong: list[str] = []
    for family, size, real in _measured_cells():
        type_name = next((t for t, f in cost_model._TYPE_TO_FAMILY.items()
                          if f == family), "")
        for body in BODY_SECONDS:
            total += 1
            truth = _persists(real, body)
            predicted = _persists(
                cost_model.estimated_restore_time(type_name, size, "disk"), body)
            if predicted == truth:
                continue
            if truth:
                kept_out += 1
            else:
                let_in += 1
            wrong.append(
                f"{family} {int(size):,}B, body {body * 1000:.0f}ms, measured "
                f"restore {real * 1000:.1f}ms: model says "
                f"{'persist' if predicted else 'skip'}, measurement says "
                f"{'persist' if truth else 'skip'}")
    return total, kept_out, let_in, wrong


def test_the_constants_agree_with_the_measurements():
    total, kept_out, let_in, wrong = _score()
    rate = (kept_out + let_in) / total
    assert rate <= 0.02, (
        f"{kept_out + let_in} of {total} promotion decisions ({rate:.1%}) "
        f"disagree with the measured restore times:" + "".join(
            "\n  " + w for w in wrong[:12]))


def test_the_constants_do_not_wave_through_a_slow_restore():
    """The costly direction: cash persists, and every later run pays a restore
    slower than recomputing -- plus the disk to keep it."""
    _total, _kept_out, let_in, wrong = _score()
    let_in_cases = [w for w in wrong if "model says persist" in w]
    assert let_in <= 1, (
        f"{let_in} values would be persisted although restoring them costs more "
        f"than recomputing:" + "".join("\n  " + w for w in let_in_cases[:8]))


def test_a_small_read_is_not_priced_like_a_large_one():
    """The intercept is what every cheap notebook statement is judged by.

    It has to be the real cost of opening and reading a small entry from a
    later session, not the residual the 100 MiB rows leave behind when the fit
    minimises seconds instead of ratios.
    """
    smallest = min(size for _f, size, _r in _measured_cells())
    measured = [r for _f, size, r in _measured_cells() if size == smallest][0]
    predicted = cost_model.estimated_restore_time("", int(smallest), "disk")
    assert predicted < measured * 2, (
        f"a {int(smallest):,}B entry measures {measured * 1000:.2f}ms to restore "
        f"but is priced at {predicted * 1000:.2f}ms")
