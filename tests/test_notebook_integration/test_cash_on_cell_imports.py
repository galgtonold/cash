"""After a restart, a name imported twice above the cell keys like the last import.

The cell that turns cash on runs before cash is on, so its imports bind names
with no lineage. The upstream simulation fills those in: simulating an import
gives each name it binds the import's lineage, if the runtime holds none --
and it held on to the FIRST. Round 23 (r23s3, 2026-09-15): the enabling cell
said ``import os, sys``, the next one ``import sys`` again. After a restart the
second import never ran (``sys`` was bound), and the simulation left ``sys``
with the first import's lineage while it keyed everything below with the
second's, as the runtime had before the restart. Every helper reading
``sys.__stderr__`` that ran again got a new lineage, ``cleaned`` missed, and a
235 s sweep recomputed.

Counted, not timed: a tee on ``CallUnit._record`` in the kernel records every
intercepted call and whether it hit.
"""

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

_TEE = """
import cash.notebook.call_unit as _cu
C = _cu.CallUnit
if not hasattr(C, "_test_orig"):
    C._test_orig = C._record
    def _tee(self, func_name, site, key, *, cache_hit, **k):
        C._test_calls.append((str(func_name), bool(cache_hit)))
        return C._test_orig(self, func_name, site, key, cache_hit=cache_hit, **k)
    C._record = _tee
C._test_calls = []
"""
_UNTEE = """
import cash.notebook.call_unit as _cu
C = _cu.CallUnit
if hasattr(C, "_test_orig"):
    C._record = C._test_orig
    del C._test_orig
"""
_CALLS = "__import__('cash.notebook.call_unit', fromlist=['_']).CallUnit._test_calls"

SLOW = "def slow(k):\n    print('RUN slow', k, file=sys.__stderr__)\n    time.sleep(0.3)\n    return k * 10"
VALUE = "v = slow(3)"
#: Replayed after a restart (nothing stores what a loop builds), so ``def slow``
#: runs again, and each ``slow(k)`` is looked up under a key that holds the
#: lineage ``slow`` got this time.
LOOP = "acc = []\nfor k in range(2):\n    acc.append(v + slow(k))"
REPORT = "print('A', acc)"
WANT = "A [30, 40]"


@pytest.fixture
def _teed(nb_runner):
    yield
    try:
        nb_runner.peek(f"exec({_UNTEE!r}, {{}})")
    except Exception:  # noqa: BLE001 - a kernel that never started has nothing to undo
        pass


def _restart_and_run(nb_runner, cell: int) -> list[tuple[str, bool]]:
    """``(function, hit)`` for every intercepted call the run of *cell* made."""
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    # The peek's own upstream check simulated the notebook and cached it
    # before the tee existed; %cash_on drops that cache.
    nb_runner.peek("get_ipython().run_line_magic('cash_on', '')")
    nb_runner.run_cell(cell)
    return ast.literal_eval(nb_runner.peek(_CALLS))


@pytest.mark.parametrize(
    "first, second",
    [
        ("import os, sys", "import sys\nimport time"),  # r23s3's two cells
        ("import sys", "import os, sys\nimport time"),
    ],
)
def test_a_restart_restores_what_a_helper_reading_a_module_computed(nb_runner, _teed, first, second):
    cells = ["import cash\n%cash_on\n" + first, second, SLOW, VALUE, LOOP, REPORT]
    nb_runner.create_notebook(cells)
    # The notebook's own first cell turns cash on, as in a kernel without the
    # autoload hook: start_kernel() would otherwise run %cash_on first.
    nb_runner.start_kernel(with_cash=False)
    nb_runner.assert_cash_active(False)
    nb_runner.run_all()
    assert WANT in nb_runner.get_output(6)

    calls = [hit for name, hit in _restart_and_run(nb_runner, 6) if name.endswith("slow")]

    assert WANT in nb_runner.get_output(6)
    # Non-vacuous: the loop replayed and called slow twice.
    assert len(calls) == 2, calls
    # Measured before the fix: both missed and ran again.
    assert all(calls), f"slow ran again after the restart: {calls}"
