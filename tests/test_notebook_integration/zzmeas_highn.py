"""High iteration count, sub-millisecond body, seconds of total work.

CAS-264's case is n=124 x 0.1ms = 12ms total, where cash's fixed overhead
dominates. The regime that actually matters is many iterations of a cheap
body summing to seconds: does cash's overhead stay CONSTANT (fine) or scale
with n (a problem)?
"""
import json, time
ON = "import cash\n%cash_on\n"
OFF = "import cash\n"

def _cells(n, ms, cash_on):
    defs = ("import time\n"
            "def busy(ms):\n    t = time.perf_counter() + ms/1000.0\n"
            "    while time.perf_counter() < t:\n        pass\n")
    loop = (f"total = 0\nfor i in range({n}):\n"
            + (f"    busy({ms})\n" if ms else "")
            + "    total += i * 2\n")
    return [(ON if cash_on else OFF), defs, loop, "print('TOTAL', total)"]

def _arm(nb, n, ms, cash_on):
    nb.create_notebook(_cells(n, ms, cash_on))
    nb.start_kernel(with_cash=cash_on)
    t0 = time.perf_counter(); nb.run_all(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); nb.run_cells([3, 4]); warm = time.perf_counter() - t0
    return {"cold_ms": round(cold*1000), "warm_ms": round(warm*1000),
            "out": nb.get_output(4)}

def test_20k_x_0_1ms_off(nb_runner):
    print("H_20K_OFF " + json.dumps(_arm(nb_runner, 20000, 0.1, False)))

def test_20k_x_0_1ms_on(nb_runner):
    print("H_20K_ON " + json.dumps(_arm(nb_runner, 20000, 0.1, True)))


def test_2m_trivial_off(nb_runner):
    print("H_2M_OFF " + json.dumps(_arm(nb_runner, 2000000, 0, False)))

def test_2m_trivial_on(nb_runner):
    print("H_2M_ON " + json.dumps(_arm(nb_runner, 2000000, 0, True)))
