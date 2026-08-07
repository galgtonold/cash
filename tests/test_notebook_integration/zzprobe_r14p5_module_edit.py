"""ADJUDICATION PROBE for round-14 P5's WRONG #2 — run directly, not via pytest.

P5's claim: edit a helper MODULE on disk, re-run only the caller, and cash
serves the pre-edit value while announcing "Module reloaded". The callee is
reached as a module attribute (`ea.fetch_entity(e)`), which matters: the call
unit's free name is `ea`, a MODULE, and modules are lineage-exempt.

Two arms, because that distinction is the whole hypothesis:
  A. `mod.f(e)`     -- attribute on an imported module   (P5's spelling)
  B. `from mod import f` then `f(e)`  -- bare name        (control)

THREE PHASES, and phase 2 is the one that makes this probe worth anything.
An earlier version of this file only checked the value after the edit, and
"correct value" there is equally consistent with NOTHING HAVING BEEN CACHED --
which would make the whole probe vacuous. Phase 2 proves caching was actually
in force before the edit is applied.

  1. cold run                -> counter == N          (work really happened)
  2. unchanged re-run        -> counter unchanged      (CACHING IS ACTIVE)
  3. edit module, re-run     -> values updated AND counter grew

Ground truth is an external call-counter file plus the VALUES, both read
outside the kernel. Run:

    python tests/test_notebook_integration/zzprobe_r14p5_module_edit.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_notebook_integration.conftest import NotebookTestRunner  # noqa: E402

N_ITEMS = 23          # P5's small-list arm was 23 items
# Body cost is a real discriminator, not a detail: below ~0.1s an entry stays
# in RAM, above it reaches disk. P5's mocked API slept 0.9s. Override from the
# command line to test that axis: `python <this> 0.9`.
BODY_S = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

MOD_TMPL = """
import time, os

def f(x):
    fd = os.open(r'{counter}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, b'X')
    os.close(fd)
    time.sleep({body})
    return x * 2 + {offset}
"""


def n_calls(counter: Path) -> int:
    return len(counter.read_bytes()) if counter.exists() else 0


def run_arm(label: str, import_line: str, call_expr: str) -> bool:
    work = Path(tempfile.mkdtemp(prefix="r14p5_"))
    mod, counter = work / "helper_mod.py", work / "calls.log"
    mod.write_text(
        MOD_TMPL.format(counter=counter, body=BODY_S, offset=0), encoding="utf-8")

    runner = NotebookTestRunner(str(work))
    runner.create_notebook([
        "import cash\n%cash_on",
        f"import sys\nsys.path.insert(0, r'{work}')\n{import_line}",
        f"items = list(range({N_ITEMS}))",
        f"out = []\nfor e in items:\n    out.append({call_expr})\nprint('OUT', out[:3])",
    ])
    runner.start_kernel()
    try:
        runner.run_all()
        cold = n_calls(counter)

        runner.run_cell(4)                       # unchanged re-run (1-based)
        warm_delta = n_calls(counter) - cold

        mod.write_text(                          # edit the module ON DISK
            MOD_TMPL.format(counter=counter, body=BODY_S, offset=1000),
            encoding="utf-8")
        before_edit = n_calls(counter)
        runner.run_cell(4)                       # re-run ONLY the caller
        after = runner.peek("out")
        edit_delta = n_calls(counter) - before_edit
    finally:
        runner.shutdown()
        shutil.rmtree(work, ignore_errors=True)

    expected = str([x * 2 + 1000 for x in range(N_ITEMS)])
    value_ok = after.strip() == expected
    # NOT `warm_delta == 0`. A decomposed loop re-runs a small measurement HEAD
    # on every warm pass -- that is how the split verdict gets its samples -- so
    # a couple of real calls out of N is caching working, not caching absent.
    # Reading +1/23 as "vacuous" sent the first run of this probe to the wrong
    # conclusion. Vacuity is "nothing was reused", i.e. the counter grew by ~N.
    caching_active = warm_delta <= max(2, N_ITEMS // 10)

    print(f"  {label}")
    print(f"     cold calls          : {cold} (expect {N_ITEMS})")
    print(f"     unchanged re-run    : +{warm_delta} "
          f"{'<- caching ACTIVE' if caching_active else '*** NOT CACHED: probe is VACUOUS ***'}")
    print(f"     after module edit   : +{edit_delta} calls, "
          f"value {'CORRECT' if value_ok else '*** STALE ***'}")
    if not value_ok:
        print(f"       got      {after}")
        print(f"       expected {expected}")
    return caching_active and value_ok


print(f"{N_ITEMS} items, {BODY_S}s body. After the module edit the values must "
      f"all gain +1000.\n")
a = run_arm("A  module attribute   helper_mod.f(e)", "import helper_mod", "helper_mod.f(e)")
b = run_arm("B  bare name          f(e)", "from helper_mod import f", "f(e)")

print()
if a and b:
    print("NOT REPRODUCED in either spelling, and caching was verified active "
          "in both -- so this is a real negative, not a vacuous pass.")
else:
    print("Look at the per-arm lines above: a VACUOUS arm proves nothing, a "
          "STALE arm reproduces P5's WRONG #2.")
