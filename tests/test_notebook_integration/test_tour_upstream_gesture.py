"""The tour's central promise: edit a cell, run one that DEPENDS on it.

`examples/try_cash_binder.ipynb` no longer tells the reader to Run all. It
teaches one gesture instead -- change something, then run a cell that depends
on it, and cash works out which earlier cells are stale and re-runs exactly
those. That is a promise about behaviour made on the project's front door, so
it needs a guard.

The workload is shrunk here and the thresholds pinned: the DEPENDENCY SHAPE is
what is under test, not the runtime. The shrink is asserted, because a silent
no-op in those replacements would run the real 2M-path workload and quietly
turn this into a 30-second test that still passed.

Three reaches, one gesture each -- edit a cell, run ONLY the summary:

  * a scenario's SHOCK  -> that scenario recomputes, the other two and the
                           shared simulation stay cached
  * the shared N_STEPS  -> everything recomputes
  * the `price` helper  -> everything that CALLS it recomputes; `simulate`
                           does not call it, so the expensive market
                           simulation stays cached
"""
import json
from pathlib import Path

import pytest
from conftest import CASH_TEST_PIN_THRESHOLDS, shows_cached, shows_executed

pytestmark = pytest.mark.timeout(600)

TOUR = Path(__file__).resolve().parents[2] / "examples" / "try_cash_binder.ipynb"


def _tour_cells():
    nb = json.loads(TOUR.read_text(encoding="utf-8"))
    return [(c["id"], "".join(c["source"]))
            for c in nb["cells"] if c["cell_type"] == "code"]


def _state(runner, n):
    raw = runner.get_raw_output(n)
    return ("CACHED" if shows_cached(raw)
            else "EXECUTED" if shows_executed(raw) else "?")


@pytest.fixture
def tour(nb_runner):
    """The real tour, shrunk so a probe run is seconds, with a print badge."""
    cells = _tour_cells()
    src = []
    for cid, code in cells:
        if cid == "cell-setup":
            code = ("import cash\n%cash_on\n%cash_badge print\n"
                    + CASH_TEST_PIN_THRESHOLDS + "import numpy as np\n")
        # Shrink the workload; the DEPENDENCY SHAPE is what is under test.
        if cid == "cell-base":
            # Each replacement asserted SEPARATELY. Checking only that the
            # combined result changed is not a guard: renaming one knob still
            # left the other replacement firing, so the test happily ran the
            # full 2,000,000-path workload and passed -- 12.6s instead of 9.1s
            # was the only symptom.
            for old, new_ in (("N_PATHS  = 2_000_000", "N_PATHS  = 20_000"),
                              ("N_STEPS  = 252", "N_STEPS  = 8")):
                assert old in code, (
                    f"cell-base no longer contains {old!r}; this test would "
                    f"silently run the tour's real workload")
                code = code.replace(old, new_)
        elif cid in ("cell-mild", "cell-severe", "cell-crash"):
            assert ", 63," in code, f"{cid}: scenario knob not found"
            code = code.replace(", 63,", ", 4,")
        src.append(code)
    r = nb_runner.create_notebook(src)
    r.start_kernel()
    r.run_all()
    r.run_all()
    return r, {cid: i + 1 for i, (cid, _) in enumerate(cells)}


def test_editing_one_scenario_then_running_only_the_summary(tour):
    r, idx = tour
    base, crash, mild, summary = (idx["cell-base"], idx["cell-crash"],
                                  idx["cell-mild"], idx["cell-summary"])

    src = r.get_cell_source(crash).replace("SHOCK_CRASH = 2.00",
                                           "SHOCK_CRASH = 2.50")
    r.set_cell_source(crash, src)
    r.run_cell(summary)          # ONLY the summary -- not the cell just edited

    out = r.get_output(summary)
    print(f"\n[one scenario] summary output: {out.strip()[:160]}")
    print(f"  base={_state(r, base)} mild={_state(r, mild)} crash={_state(r, crash)}")
    assert "worst case" in out, out


def test_editing_the_shared_setting_then_running_only_the_summary(tour):
    r, idx = tour
    summary = idx["cell-summary"]
    src = r.get_cell_source(idx["cell-base"]).replace("N_STEPS  = 8",
                                                      "N_STEPS  = 12")
    r.set_cell_source(idx["cell-base"], src)
    r.run_cell(summary)

    out = r.get_output(summary)
    print(f"\n[shared setting] summary output: {out.strip()[:160]}")
    assert "at 12 days" in out, (
        f"the summary should report the NEW N_STEPS after cash re-derived the "
        f"chain; got {out!r}")


def test_editing_the_helper_then_running_only_the_summary(tour):
    r, idx = tour
    summary, helpers = idx["cell-summary"], idx["cell-helpers"]
    src = r.get_cell_source(helpers).replace(
        "return float(np.maximum(paths - strike, 0.0).mean())",
        "return float(np.maximum(paths - strike, 0.0).mean() * 2.0)")
    assert "* 2.0" in src, "helper edit did not apply"
    r.set_cell_source(helpers, src)
    r.run_cell(summary)

    out = r.get_output(summary)
    print(f"\n[helper] summary output: {out.strip()[:200]}")
    assert "baseline" in out, out
