"""PROBE: does a hidden-state argument keep a STABLE lineage?

The design question (CAS-243 review, 2026-07-29):

    The call unit keys on the LINEAGE of the call's free names and
    deliberately omits the iteration-context discriminator (that omission is
    the CAS-242 reorder fix). If an argument carries hidden state whose VALUE
    changes while its LINEAGE does not, every iteration computes the same key
    and iterations 2..N are served iteration 1's value -- wrong within a
    single run.

Arm A establishes the ORACLE: with cash off, what does the loop print?
Arm B is the same loop under cash as it ships today (decorator-keyed
interception via the directive), which keys on argument VALUES and should be
correct.

If A and B agree, today is correct and any divergence under lineage keying is
a REGRESSION introduced by the design change, not a pre-existing bug.
"""
import json


SETUP_ON = "import cash\n%cash_on\n"
SETUP_OFF = "import cash\n"

DEFS = """\
import time
CALLS = []
def compute(v):
    CALLS.append(v)
    time.sleep(0.3)
    return v * 10
"""

# The loop under test. `_` is the loop variable; the expensive call reads
# `it`, whose lineage should not move as the iterator is consumed.
LOOP = """\
CALLS.clear()
it = iter(range(3))
out = []
for _ in range(3):
    out.append(compute(next(it)))
print('OUT', out, 'CALLS', CALLS)
"""

LOOP_DIRECTIVE = """\
CALLS.clear()
it = iter(range(3))
out = []
for _ in range(3):
    # @cash:cache-calls
    out.append(compute(next(it)))
print('OUT', out, 'CALLS', CALLS)
"""


def test_probe_hidden_state_argument(nb_runner):
    """Report the oracle and the shipped-interception behaviour side by side."""
    results = {}

    # --- Arm A: no cash. The oracle. ---
    nb_runner.create_notebook([SETUP_OFF, DEFS, LOOP])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    results["A_oracle_no_cash"] = nb_runner.get_output(3)

    print("PROBE_RESULT " + json.dumps(results))


def test_probe_hidden_state_argument_under_cash(nb_runner):
    """Same loop with cash on and the cache-calls directive (shipped path)."""
    results = {}

    nb_runner.create_notebook([SETUP_ON, DEFS, LOOP_DIRECTIVE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    results["B_shipped_interception_run1"] = nb_runner.get_output(3)

    nb_runner.run_cells([3])
    results["B_shipped_interception_run2"] = nb_runner.get_output(3)

    print("PROBE_RESULT " + json.dumps(results))
