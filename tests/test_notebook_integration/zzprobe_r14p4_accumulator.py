"""ADJUDICATION PROBE for round-14 P4's BLOCKING and WRONG-2 — run directly.

P4 reported three findings and proposed ONE root cause for all of them: the new
per-statement/per-call caching skips a mutating, side-effecting statement once
its declared inputs look unchanged, without checking that the skip is safe.

Two of the three are testable in one shape, because they are the same cell:

    4  equity = []
       state  = {'cash': 10000.0}
       open(LOG, 'w').close()          <- the three RESET statements
       for d in days:
           equity.append(simulate_day(d, state))    <- appends a line to LOG

  BLOCKING: kernel restart, then run a DOWNSTREAM cell -> P4 saw `equity`
            come back EMPTY (len 0 -> ZeroDivisionError), with LOG not growing,
            i.e. the loop did not run at all rather than running differently.

  WRONG-2:  edit an upstream cell, jump to a downstream cell -> P4 saw LOG
            DOUBLE (348 -> 696) with the counter continuing from the previous
            run's end, i.e. the loop re-ran while the three reset statements
            above it were skipped.

Note those two point in OPPOSITE directions (nothing ran vs ran-without-reset),
which is itself worth knowing: a single root cause has to explain both.

Ground truth is the external LOG file, read outside the kernel, plus `peek`
for live kernel state (never a print -- a cache hit replays prints).

    python tests/test_notebook_integration/zzprobe_r14p4_accumulator.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_notebook_integration.conftest import NotebookTestRunner  # noqa: E402

N_DAYS = 40


def cells(log: Path) -> list[str]:
    return [
        "import cash\n%cash_on",
        "import time, os\n"
        f"LOG = r'{log}'\n"
        "def simulate_day(d, st):\n"
        "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n"
        "    time.sleep(0.02)\n"
        "    st['cash'] = st['cash'] * 1.001\n"
        "    return st['cash']",
        f"days = list(range({N_DAYS}))",
        "equity = []\n"
        "state = {'cash': 10000.0}\n"
        "open(LOG, 'w').close()\n"
        "for d in days:\n"
        "    equity.append(simulate_day(d, state))",
        "summary = (len(equity), round(state['cash'], 4))\nprint('SUM', summary)",
    ]


def n(log: Path) -> int:
    return len(log.read_bytes()) if log.exists() else 0


work = Path(tempfile.mkdtemp(prefix="r14p4_"))
log = work / "daily.log"
runner = NotebookTestRunner(str(work))
runner.create_notebook(cells(log))
runner.start_kernel()
try:
    runner.run_all()
    print(f"cold          : log={n(log):3d} (expect {N_DAYS})  "
          f"len(equity)={runner.peek('len(equity)')}")

    # --- P4 WRONG-2: edit an upstream cell, then jump to the loop cell -------
    runner.set_cell_source(3, f"days = list(range({N_DAYS + 5}))")
    runner.run_cell(3)
    before = n(log)
    runner.run_cell(4)
    after = n(log)
    print(f"after upstream edit + loop rerun: log {before} -> {after}  "
          f"len(equity)={runner.peek('len(equity)')}")
    doubled = after > N_DAYS + 5 + 2
    print(f"   WRONG-2 (log accumulated instead of reset): "
          f"{'REPRODUCED' if doubled else 'not reproduced'}"
          f"  [reset would give {N_DAYS + 5}]")

    # --- P4 BLOCKING: real restart, then run a DOWNSTREAM cell only ---------
    runner.restart()
    runner.run_cell(1)          # %cash_on, as the protocol requires
    pre = n(log)
    runner.run_cell(5)          # downstream only -- never re-run the loop
    try:
        length = runner.peek("len(equity)")
    except Exception as e:      # noqa: BLE001
        length = f"<raised {type(e).__name__}>"
    print(f"after restart + downstream-only: log {pre} -> {n(log)}  "
          f"len(equity)={length}")
    empty = str(length).strip() in ("0", "<raised", "") or "raised" in str(length)
    print(f"   BLOCKING (accumulator empty after restart): "
          f"{'REPRODUCED' if empty else 'not reproduced'}")
finally:
    runner.shutdown()
    shutil.rmtree(work, ignore_errors=True)
