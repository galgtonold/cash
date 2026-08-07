"""ADJUDICATION PROBE, round-14 P1 WRONG #1, FAITHFUL shape — run directly.

The minimal version (`zzprobe_r14p1_uncacheable_reads_stale.py`) did NOT
reproduce: an uncacheable save cell correctly saw the new upstream value. So
either P1 was wrong, or the minimal shape dropped something load-bearing.

Two candidates it dropped, both known to matter in this codebase:

  1. **In-place column assignment** (`clean["margin"] = ...`) -- a subscript
     store is an in-place mutation, which is skip-cached and handled by a
     different path than a rebinding assignment.
  2. **A DataFrame big enough that content hashing is SAMPLED** rather than
     full (>8 MiB). A sampled hash can miss a change that lands outside the
     sampled regions.

This probe restores both, and keeps the exact cell order P1 described:

    3  raw    = <build a big frame>
    4  clean  = raw[raw.qty > THRESH]          <- edited, then run by hand
    5  clean["margin"] = ...                   <- in-place, NOT re-run by hand
    6  agg    = clean.groupby("cat")["margin"].sum()
    7  fig, ax = plt.subplots(); ax.plot(...)
    8  ax.set_title(...); savefig; marker.write(agg.sum())

Move: edit cell 4's threshold, RUN cell 4, then run ONLY cell 8.

    python tests/test_notebook_integration/zzprobe_r14p1_faithful_chain.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_notebook_integration.conftest import NotebookTestRunner  # noqa: E402

N_ROWS = 2_000_000     # ~8 rows x 8 bytes -> comfortably past the 8 MiB sampling line

CELLS = [
    "import cash\n%cash_on",
    "import matplotlib\nmatplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\nimport numpy as np, pandas as pd",
    f"rng = np.random.default_rng(0)\n"
    f"raw = pd.DataFrame({{'qty': rng.integers(0, 100, {N_ROWS}),\n"
    f"                    'price': rng.random({N_ROWS}) * 10,\n"
    f"                    'cost': rng.random({N_ROWS}) * 5,\n"
    f"                    'cat': rng.integers(0, 4, {N_ROWS})}})\n"
    f"len(raw)",
    "clean = raw[raw.qty > 10]",
    "clean['margin'] = clean['price'] - clean['cost']",
    "agg = clean.groupby('cat')['margin'].sum()",
    "fig, ax = plt.subplots()\nax.plot(agg.index, agg.values)",
    "ax.set_title('margin')\nfig.savefig(r'{png}')\n"
    "open(r'{marker}', 'w').write(f'{{agg.sum():.4f}}')",
]


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="r14p1f_"))
    marker, png = work / "marker.txt", work / "out.png"
    cells = list(CELLS)
    cells[-1] = cells[-1].format(png=png, marker=marker)

    runner = NotebookTestRunner(str(work))
    runner.create_notebook(cells)
    runner.start_kernel()
    try:
        runner.run_all()
        before = marker.read_text(encoding="utf-8").strip()

        # Edit the CLEANING cell and run it. Cells 5-7 (margin, agg, build) are
        # deliberately not re-run -- that is the whole point.
        runner.set_cell_source(4, "clean = raw[raw.qty > 50]")
        runner.run_cell(4)
        try:
            runner.run_cell(8)
            err = ""
        except Exception as e:                       # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:90]}"
        after = marker.read_text(encoding="utf-8").strip()

        # Ground truth: what the chain gives with cash out of the way.
        runner.set_cell_source(9 if len(cells) >= 9 else 8, "")
        oracle = runner.peek(
            "float(raw[raw.qty > 50].assign(margin=lambda d: d['price'] - d['cost'])"
            ".groupby('cat')['margin'].sum().sum())")
    finally:
        runner.shutdown()
        shutil.rmtree(work, ignore_errors=True)

    print(f"  marker before edit : {before}")
    print(f"  marker after  edit : {after}")
    print(f"  cash-free oracle   : {oracle}")
    if err:
        print(f"  raised             : {err}")
    print()
    if after == before:
        print("REPRODUCED: the save cell wrote the pre-edit value after an "
              "upstream edit that was actually run.")
    else:
        try:
            print("NOT stale. Matches oracle:"
                  f" {abs(float(after) - float(oracle)) < 0.01}")
        except ValueError:
            print(f"Inconclusive -- could not compare {after!r} to {oracle!r}")


main()
