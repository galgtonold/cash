"""A floor under cache effectiveness after an edit.

The rest of the suite asserts that cached values are *correct*. Almost
nothing asserts that they are *reused* -- and reuse is the entire product.
A regression that quietly stopped restoring anything would leave every
correctness test green, because recomputing always produces the right
answer. That is how the reference notebooks came to spend 42-101% of their
compute again on a warm restart without a single test noticing.

This is the cheap end of that instrument. The offline sweep
(``benchmarks/bench_notebook_edit.py`` over the reference notebooks) is
where the real numbers come from; this keeps a small, fast version in CI so
the floor cannot silently drop between sweeps.

Design notes, both deliberate:

* **Counts, not seconds.** Every assertion here is on a statement *count*.
  Wall-clock thresholds under xdist are this repo's entire integration-flake
  class, and a benchmark-shaped test is the last place to add another.
  Statement statuses are deterministic; the seconds behind them are not.

* **Cells sized above the cost-model floor.** cash declines to cache any
  statement whose compute is under ~10ms -- storing it would cost more than
  recomputing. A first version of this test used a chain of microsecond
  statements and measured a restorable set of exactly zero, at which point
  the waste assertion passed while proving nothing. The arrays below are
  sized so each step clears that floor; keep them that way.

* **``session_mode='live'``.** The measured half reuses the priming run's
  ``Cash``, which is what re-running a cell in a live kernel gets you. It is
  the faithful shape for an edit-and-rerun test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.bench_notebook_edit import run_notebook_edit_benchmark

# Measured on this notebook at the time of writing. These are CEILINGS on
# current behaviour, not targets: the goal is zero, and lowering them as
# effectiveness bugs are fixed is the point of having them. Raising one
# means a regression -- find out why before you touch the number.
MAX_WASTED_STATEMENTS = 0

# Guard against a vacuous pass. If nothing restores, nothing can be wasted
# and every assertion below is trivially true -- which is exactly how a
# broken harness would look. The chain below produces five restorable
# statements; the margin allows for a machine fast enough to drop one of
# them under the cost-model floor without failing the build over it.
MIN_RESTORABLE_STATEMENTS = 4


def _write_chain_notebook(path: Path) -> None:
    """A short dependency chain: each cell consumes the previous cell's value.

    The *shape* of these statements is load-bearing, and two earlier
    versions of this notebook measured nothing because of it:

    * A chain of ``base * 2`` steps ran in 0.2-2ms, below cash's ~10ms
      caching floor. Nothing was stored, so nothing could be reused.
    * Raising the array size instead made each step a 32MB array. Those
      clear the time floor but the cost model correctly refuses them --
      writing 32MB costs more than recomputing 15ms of arithmetic.

    What caches is **expensive compute with a small result**, so each step
    below is several ufunc passes reduced to one float: ~30-60ms of work
    for a 24-byte value. Measured, not assumed. Keep that property if you
    change these cells, and re-measure if you do.
    """
    heavy = [
        "np.sin(base).sum() + np.cos(base).sum() "
        "+ np.sqrt(base).sum() + np.log1p(base).sum()",
        "np.tan(base).sum() + np.arctan(base).sum() "
        "+ np.exp(-base).sum() + np.cbrt(base).sum()",
        "np.tanh(base).sum() + np.square(base).sum() "
        "+ np.abs(base).sum() + np.expm1(-base).sum()",
        "np.arcsinh(base).sum() + np.log10(1 + base).sum() "
        "+ np.rint(base).sum() + np.negative(base).sum()",
        "np.cos(base).sum() + np.log2(1 + base).sum() "
        "+ np.sign(base).sum() + np.reciprocal(1 + base).sum()",
    ]
    cells = ["import numpy as np",
             "base = np.arange(2_000_000, dtype='float64')"]
    previous = "0.0"
    for i, expression in enumerate(heavy):
        cells.append(f"s{i} = float({expression}) + {previous}")
        previous = f"s{i}"
    cells.append(f"total = {previous}")
    cells.append("print(f'{total:.3f}')")
    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src}
            for src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3",
                                    "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


@pytest.fixture(scope="module")
def edit_report(tmp_path_factory):
    """Run the edit benchmark once and share it across the assertions.

    Module-scoped because the run is the expensive part; splitting it per
    assertion would multiply the cost for no extra coverage.
    """
    tmp_path = tmp_path_factory.mktemp("edit_floor")
    nb = tmp_path / "chain.ipynb"
    _write_chain_notebook(nb)
    return run_notebook_edit_benchmark(
        nb, tmp_path / "work",
        max_sites=2, session_mode="live", log=lambda *a, **k: None,
    )


def test_the_measurement_is_not_vacuous(edit_report):
    """Something must restore, or every other assertion here proves nothing."""
    assert edit_report["restorable_count"] >= MIN_RESTORABLE_STATEMENTS, (
        f"only {edit_report['restorable_count']} statement(s) restore with no "
        f"edit at all. Either cash stopped reusing anything on a plain "
        f"dependency chain, or this harness stopped measuring it. Both are "
        f"worth stopping for -- do not lower MIN_RESTORABLE_STATEMENTS to "
        f"make this pass."
    )


def test_the_positive_control_sees_a_real_dependency(edit_report):
    """The harness must be able to detect recomputation it should detect.

    ``linked`` injects a probe and a reader of that probe, then changes the
    probe's value. The reader has to recompute. If it does not, this file's
    zero-waste assertions are measuring nothing at all.
    """
    controls = [s for s in edit_report["scenarios"] if s["kind"] == "linked"]
    assert controls, "no positive control scenario was planned"
    for control in controls:
        assert control["control_sink_recomputed"] is True, (
            "the positive control did not recompute after its input changed. "
            "The harness cannot see a real dependency, so no result it "
            "produces about wasted work can be trusted."
        )


def test_a_null_edit_does_not_recompute_downstream_work(edit_report):
    """The floor itself.

    A comment, or an assignment to a name nothing reads, cannot change any
    downstream value. Anything downstream that recomputes anyway is work
    the user pays for and gets nothing back from.
    """
    offenders = []
    for scenario in edit_report["scenarios"]:
        if scenario["kind"] == "linked":
            continue
        if scenario["wasted_count"] > MAX_WASTED_STATEMENTS:
            offenders.append(
                f"{scenario['label']}: {scenario['wasted_count']} wasted "
                f"of {scenario['restorable_count']} restorable "
                f"(e.g. {[w['code'][:60] for w in scenario['wasted'][:3]]})"
            )
    assert not offenders, (
        "an edit that cannot affect any downstream value still forced "
        "downstream statements to recompute:\n  " + "\n  ".join(offenders)
    )
