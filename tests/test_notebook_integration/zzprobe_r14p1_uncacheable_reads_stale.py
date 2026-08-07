"""ADJUDICATION PROBE for round-14 P1's WRONG #1 — run directly, not via pytest.

P1's claim, restated as a mechanism: a plot/save cell is uncacheable BY DESIGN
(``fig``/``ax`` are identity-coupled and refused), so it is force-executed on
every run. P1 says that forced execution reads its upstream variable out of the
live namespace **without re-checking whether that variable is stale**, while an
ordinary CACHEABLE read of the same variable does check and refuses.

If true, the staleness detector exists and simply is not consulted on the path
every plotting cell takes.

Shape (mirrors P1's cleaning -> margin -> agg -> build -> save chain):

    3  base = 10
    4  derived = base * 3        <- the "agg" analogue; NOT re-run by hand
    5  fig, ax = plt.subplots(); ax.plot(...)
    6  ax.set_title(...); fig.savefig(png); marker.write(derived)   <- uncacheable

Move under test: edit cell 3, RUN cell 3, then run ONLY cell 6 — skipping the
cell that rebuilds ``derived``. That is P1's "just regenerate the chart".

Correct outcome: the marker file holds the NEW value (cash reconstructs
``derived`` first), or the run refuses loudly. Wrong outcome: the marker holds
the OLD value, silently.

ARM B is the control P1's own diagnosis rests on: an ordinary cacheable read of
the same variable in the same state. If B refuses/updates while A serves stale,
the asymmetry is confirmed and P1's mechanism is right.

    python tests/test_notebook_integration/zzprobe_r14p1_uncacheable_reads_stale.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_notebook_integration.conftest import NotebookTestRunner  # noqa: E402

SETUP = "import cash\n%cash_on"
IMPORTS = ("import matplotlib\nmatplotlib.use('Agg')\n"
           "import matplotlib.pyplot as plt")


def build(work: Path, marker: Path, png: Path, save_cell: str):
    return [
        SETUP,
        IMPORTS,
        "base = 10",
        "derived = base * 3",
        "fig, ax = plt.subplots()\nax.plot([0, 1], [0, derived])",
        save_cell.format(marker=marker, png=png),
    ]


def run_arm(label: str, save_cell: str) -> str:
    work = Path(tempfile.mkdtemp(prefix="r14p1_"))
    marker, png = work / "marker.txt", work / "out.png"
    runner = NotebookTestRunner(str(work))
    runner.create_notebook(build(work, marker, png, save_cell))
    runner.start_kernel()
    try:
        runner.run_all()
        first = marker.read_text(encoding="utf-8").strip() if marker.exists() else "<none>"

        # Edit the ROOT and run it -- exactly P1's move. Cell 4 (`derived`) is
        # deliberately NOT re-run by hand; that is what cash must handle.
        runner.set_cell_source(3, "base = 100")     # 1-based
        runner.run_cell(3)
        try:
            runner.run_cell(6)
            err = ""
        except Exception as e:                       # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:80]}"
        second = marker.read_text(encoding="utf-8").strip() if marker.exists() else "<none>"
    finally:
        runner.shutdown()
        shutil.rmtree(work, ignore_errors=True)

    verdict = ("STALE" if second == first else
               "updated" if second == "300" else f"other({second})")
    print(f"  {label}")
    print(f"     before edit: {first}   after edit+rerun: {second}   -> {verdict}"
          f"{'   raised ' + err if err else ''}")
    return verdict


print("base 10 -> derived 30; after editing base to 100, derived must be 300.\n")

a = run_arm(
    "A  uncacheable save cell (fig/ax + savefig + file write)",
    "ax.set_title(f'V={{derived}}')\n"
    "fig.savefig(r'{png}')\n"
    "open(r'{marker}', 'w').write(str(derived))",
)
b = run_arm(
    "B  ordinary cacheable read (control)",
    "open(r'{marker}', 'w').write(str(derived))",
)

print()
if a == "STALE" and b != "STALE":
    print("CONFIRMED, and the ASYMMETRY holds: the uncacheable path serves stale")
    print("while the ordinary path does not. P1's mechanism is right.")
elif a == "STALE" and b == "STALE":
    print("Stale on BOTH paths -- real, but NOT specific to uncacheable cells;")
    print("P1's mechanism is wrong even though the finding stands.")
elif a != "STALE":
    print("NOT reproduced on the uncacheable path.")
