"""MEASUREMENT: what does a call-unit store actually cost?

CAS-261 proposes letting the call unit store in aggregate (N sub-floor
calls whose total clears a floor). That trades ONE whole-loop store for N
individual stores. Whether it pays depends on a number nobody has measured:
the per-call store + hit cost.

n=124 at 5ms = 0.62s of body work, and n=124 is below the single-unit
threshold (125), so this is exactly CAS-261's band.

`_COST_FLOOR_S` is read as a module global at call time, so arm C flips it
IN-KERNEL -- no source edit, and the full real path (key build, _storable,
backend.set, hit lookup, replay) is measured, not a proxy.
"""
import json, time

SETUP_ON = "import cash\n%cash_on\n"
SETUP_OFF = "import cash\n"
FLOOR0 = "import cash.notebook.call_unit as cu\ncu._COST_FLOOR_S = 0.0\nprint('floor', cu._COST_FLOOR_S)"

def _defs(payload):
    # busy-wait, not sleep: Windows sleep granularity (~1-15ms) would swamp a 5ms body.
    return (
        "import time, numpy as np\n"
        "CALLS = []\n"
        "def busy(ms):\n"
        "    t = time.perf_counter() + ms/1000.0\n"
        "    while time.perf_counter() < t:\n"
        "        pass\n"
        "def slow(x):\n"
        "    CALLS.append(x)\n"
        "    busy(5)\n"
        f"    return {payload}\n"
    )

N = 124
LOOP = f"out = []\nfor e in range({N}):\n    out.append(slow(e))\n"
SINK = "print('NCALLS', len(CALLS), '| OUTLEN', len(out))"

def _run(nb, cells, loop_idx, sink_idx, with_cash=True):
    nb.create_notebook(cells)
    nb.start_kernel(with_cash=with_cash)
    t0 = time.perf_counter()
    nb.run_all()
    cold_wall = time.perf_counter() - t0
    cold = nb.get_output(sink_idx)
    t0 = time.perf_counter()
    nb.run_cells([loop_idx, sink_idx])          # unchanged warm rerun
    warm_wall = time.perf_counter() - t0
    return {"cold": cold, "cold_wall_ms": round(cold_wall*1000),
            "warm": nb.get_output(sink_idx), "warm_wall_ms": round(warm_wall*1000)}

def _arm(nb, payload, floor0, with_cash=True):
    setup = SETUP_ON if with_cash else SETUP_OFF
    if floor0:
        cells = [setup, FLOOR0, _defs(payload), LOOP, SINK]
        return _run(nb, cells, 4, 5, with_cash)
    cells = [setup, _defs(payload), LOOP, SINK]
    return _run(nb, cells, 3, 4, with_cash)

def test_a_oracle_int(nb_runner):
    print("M_A_ORACLE_INT " + json.dumps(_arm(nb_runner, "x * 2 + 1", False, with_cash=False)))

def test_b_default_floor_int(nb_runner):
    print("M_B_DEFAULT_INT " + json.dumps(_arm(nb_runner, "x * 2 + 1", False)))

def test_c_floor0_int(nb_runner):
    print("M_C_FLOOR0_INT " + json.dumps(_arm(nb_runner, "x * 2 + 1", True)))

def test_d_floor0_100kb(nb_runner):
    print("M_D_FLOOR0_100KB " + json.dumps(_arm(nb_runner, "np.zeros(12500)", True)))

def test_e_default_100kb(nb_runner):
    print("M_E_DEFAULT_100KB " + json.dumps(_arm(nb_runner, "np.zeros(12500)", False)))
