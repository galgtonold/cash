"""ADJUDICATION PROBE for round-14 P3's WRONG — run directly.

P3's claim: a callee that WRITES A FILE gets its side effect silently dropped
when the looped-over list is REORDERED, for one specific array shape (a 2D
`standard_normal((n, k))` plus an axis reduction). Its own control -- the same
shape mutating a GLOBAL LIST instead of writing a file -- correctly re-executes
on the identical reorder.

Three arms here, so the claimed discriminator is actually isolated:

  A  file write,  2D array + axis reduction   (P3's failing shape)
  B  global-list mutation, same 2D shape      (P3's passing control)
  C  file write,  1D array                    (P3 said 1D never reproduced)

One caveat P3's setup carries: it wrote its counter with `builtins.open`, which
cash's FileAccessTracker patches into a tracked dependency -- the protocol asks
for `os.open`/`os.write` precisely to keep the instrument out of the cache key.
Arm A uses `open` (faithful to P3), arm D repeats it with `os.write` to show
whether the instrument itself is the variable.

    python tests/test_notebook_integration/zzprobe_r14p3_filewrite_reorder.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_notebook_integration.conftest import NotebookTestRunner  # noqa: E402

N_PATHS = 200_000      # keeps each call ~0.1s: above every floor, still quick

BODY_2D = ("    rng = np.random.default_rng(7)\n"
           "    z = rng.standard_normal(({n}, 10))\n"
           "    S = z.sum(axis=1) * vol\n"
           "    est = S.mean()\n")
BODY_1D = ("    rng = np.random.default_rng(7)\n"
           "    z = rng.standard_normal({n} * 10)\n"
           "    est = (z.sum() * vol) / {n}\n")

WRITE_OPEN = "    with open(LOG, 'a') as f:\n        f.write('X')\n"
WRITE_OS = ("    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
            "    os.write(fd, b'X')\n    os.close(fd)\n")
MUTATE_GLOBAL = "    SEEN.append(vol)\n"


def arm(label: str, body: str, effect: str, *, global_probe: bool) -> None:
    work = Path(tempfile.mkdtemp(prefix="r14p3_"))
    log = work / "sidefx.log"
    runner = NotebookTestRunner(str(work))
    runner.create_notebook([
        "import cash\n%cash_on",
        "import os\nimport numpy as np\n"
        f"LOG = r'{log}'\nSEEN = []\n"
        "def price(vol):\n"
        + body.format(n=N_PATHS) + effect +
        "    return float(est)",
        "vols = [0.10, 0.15, 0.20]",
        "out = []\nfor v in vols:\n    out.append(price(v))",
    ])
    runner.start_kernel()
    try:
        runner.run_all()
        cold_n = len(log.read_bytes()) if log.exists() else 0
        cold_seen = runner.peek("len(SEEN)")

        runner.set_cell_source(3, "vols = [0.15, 0.10, 0.20]")   # REORDER
        runner.run_cell(3)
        runner.run_cell(4)
        warm_n = len(log.read_bytes()) if log.exists() else 0
        warm_seen = runner.peek("len(SEEN)")
    finally:
        runner.shutdown()
        shutil.rmtree(work, ignore_errors=True)

    if global_probe:
        got, before = int(warm_seen) - int(cold_seen), cold_seen
        unit = "SEEN entries"
    else:
        got, before = warm_n - cold_n, cold_n
        unit = "log bytes"
    print(f"  {label}")
    print(f"     cold {unit}={before}   after reorder: +{got}   "
          f"{'*** SIDE EFFECT DROPPED ***' if got == 0 else 'side effect fired'}")


print("Reordering [0.10,0.15,0.20] -> [0.15,0.10,0.20]. The three values are all\n"
      "already cached, so a reorder may legitimately reuse them -- but a callee\n"
      "with a side effect must still run, per the docs P3 cites.\n")

arm("A  file write (builtins.open), 2D + axis reduction  [P3's shape]",
    BODY_2D, WRITE_OPEN, global_probe=False)
arm("B  global-list mutation,      2D + axis reduction  [P3's control]",
    BODY_2D, MUTATE_GLOBAL, global_probe=True)
arm("C  file write (builtins.open), 1D array",
    BODY_1D, WRITE_OPEN, global_probe=False)
arm("D  file write (os.write),      2D + axis reduction",
    BODY_2D, WRITE_OS, global_probe=False)
