"""PROBE: does the STATEMENT path already freeze an impure callee?

The question under test, from the CAS-243/246 design discussion:

    Is sub-unit (call) analysis actually WEAKER than statement analysis, or is
    the statement path equally blind to a callee's body and simply protected,
    in the append case, by an unrelated refusal (the mutation)?

If ``x = next_seq()`` freezes across cell re-runs, statement analysis is
already blind to callee impurity and interception adds no new *class* of
defect -- only a wider surface.
"""
import json


SETUP = """\
import cash
%cash_on
"""

# The sleep is load-bearing: cash has a ~10ms cost floor, so a microsecond
# callee is not cached AT ALL and would confound "not cached" with "not frozen".
# Matches CAS-246's measured repro.
DEFS = """\
import time
N = [0]
def next_seq():
    N[0] += 1
    time.sleep(0.3)
    return N[0]
"""

# Arm A: plain assignment -- the ordinary statement cache path.
ARM_A = """\
a = next_seq()
print('A', a)
"""

# Arm B: in-place mutation -- skip-cached today (the shape CAS-243 exists for).
ARM_B = """\
seen = []
"""

ARM_B_RUN = """\
seen.append(next_seq())
print('B', seen)
"""


def test_probe_impure_callee_freeze(nb_runner):
    nb_runner.create_notebook([SETUP, DEFS, ARM_A, ARM_B, ARM_B_RUN])
    nb_runner.start_kernel()
    nb_runner.run_all()

    # get_output / run_cells are 1-BASED: ARM_A is cell 3, ARM_B_RUN is cell 5.
    out = {"A": [], "B": []}
    out["A"].append(nb_runner.get_output(3))
    out["B"].append(nb_runner.get_output(5))

    # Re-run ONLY the two measurement cells, twice more.
    for _ in range(2):
        nb_runner.run_cells([3, 5])
        out["A"].append(nb_runner.get_output(3))
        out["B"].append(nb_runner.get_output(5))

    print("PROBE_RESULT " + json.dumps(out))
    # No assertion -- this probe reports, it does not gate.
