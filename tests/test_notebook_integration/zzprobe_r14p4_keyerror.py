"""ADJUDICATION PROBE for round-14 P4's WRONG-1 — run directly.

P4's claim: a bare call to a mutating helper --

    clean_columns(df_work)      # does frame.drop(columns=[...], inplace=True)

-- succeeds on run 1, and on an identical warm re-run the callee body provably
executes again (external counter grows) yet the `frame.drop()` that "must raise
KeyError on an already-missing column" does not raise. P4 called that a
swallowed error, using as its oracle: `%cash_off`, then calling the helper a
SECOND time on the same live object -> KeyError, as vanilla pandas requires.

**The oracle is the thing to check.** Cash's idempotent-re-run contract says
re-running a self-modifying cell must run FROM THE CELL'S ENTRY STATE, not from
the mutated state -- so before the re-run `df` should have its dropped column
back, and the drop should legitimately succeed. If so, P4 compared two
different states: cash restores the object first, plain Python does not, and no
error was swallowed.

This probe reads the column list from the LIVE KERNEL (`peek`, never a print,
which a cache hit would replay) immediately before and after each re-run, so
the two hypotheses are directly distinguishable:

    restored -> 'tmp' is BACK before the re-run   => contract working, NOT a bug
    absent   -> 'tmp' still gone and no KeyError  => genuinely swallowed

    python tests/test_notebook_integration/zzprobe_r14p4_keyerror.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_notebook_integration.conftest import NotebookTestRunner  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="r14p4k_"))
counter = work / "clean_calls.log"

runner = NotebookTestRunner(str(work))
runner.create_notebook([
    "import cash\n%cash_on",
    "import os, time\nimport pandas as pd\n"
    f"CNT = r'{counter}'\n"
    "def clean_columns(frame):\n"
    "    fd = os.open(CNT, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
    "    os.write(fd, b'X')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.02)\n"
    "    frame.drop(columns=['tmp'], inplace=True)\n"
    "    frame['flag'] = 1",
    "df = pd.DataFrame({'a': [1, 2, 3], 'tmp': [4, 5, 6]})",
    "clean_columns(df)",
    "cols = list(df.columns)\nprint('COLS', cols)",
])
runner.start_kernel()
try:
    runner.run_all()
    n0 = len(counter.read_bytes())
    print(f"cold run              : cols={runner.peek('list(df.columns)')}  calls={n0}")

    for attempt in (1, 2):
        pre = runner.peek("list(df.columns)")
        try:
            runner.run_cell(4)
            err = "none"
        except Exception as e:                       # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:60]}"
        n1 = len(counter.read_bytes())
        post = runner.peek("list(df.columns)")
        restored = "tmp" in pre
        print(f"warm re-run #{attempt}         : cols BEFORE={pre}  "
              f"-> AFTER={post}  calls={n1}  raised={err}")
        print(f"   'tmp' present before the re-run: {restored}"
              f"   {'<- cash reset the object first' if restored else '<- object still mutated'}")
        n0 = n1
finally:
    runner.shutdown()
    shutil.rmtree(work, ignore_errors=True)

print()
print("If 'tmp' is BACK before each re-run, the drop legitimately succeeds and")
print("P4's oracle (calling twice on an unreset object) was not comparable.")
