"""Steady state (run 3+), after the split verdict is learned AND the tail stored."""
import json, time
ON = "import cash\n%cash_on\n"
OFF = "import cash\n"
N = 124

def _defs(ms):
    return ("import time\nCALLS = []\n"
            "def busy(ms):\n    t = time.perf_counter() + ms/1000.0\n"
            "    while time.perf_counter() < t:\n        pass\n"
            "def slow(x):\n    CALLS.append(x)\n"
            f"    busy({ms})\n    return x * 10 + 1\n")

LOOP = f"out = []\nfor t in list(range(1, {N+1})):\n    out.append(slow(t))\n"
SINK = "print('CALLS', len(CALLS))"

def _arm(nb, ms, cash_on):
    nb.create_notebook([(ON if cash_on else OFF), _defs(ms), LOOP, SINK])
    nb.start_kernel(with_cash=cash_on); nb.run_all()
    nb.run_cells([3, 4])                       # run 2: learn + store tail
    base = nb.get_output(4)
    t0 = time.perf_counter(); nb.run_cells([3, 4]); warm = time.perf_counter() - t0
    after = nb.get_output(4)
    n_before = int(base.split()[1]); n_after = int(after.split()[1])
    return {"steady_ms": round(warm*1000), "real_calls": n_after - n_before}

def _pair(nb, ms):
    return {"cash_off": _arm(nb, ms, False), "cash_on": _arm(nb, ms, True)}

def test_0_1ms(nb_runner): print("ST_0_1 " + json.dumps(_pair(nb_runner, 0.1)))
def test_1ms(nb_runner):   print("ST_1 " + json.dumps(_pair(nb_runner, 1)))
def test_2_5ms(nb_runner): print("ST_2_5 " + json.dumps(_pair(nb_runner, 2.5)))
