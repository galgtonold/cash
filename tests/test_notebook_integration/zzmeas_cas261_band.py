"""How much does the UNCOVERED band actually cost on HEAD?

After step 1 the bands are:
  n >= 125            -> single unit (covered)
  n < 125, body >=3ms -> per-call caching (covered by step 1)
  n < 125, body < 3ms -> UNCOVERED; this measures it.

Warm rerun, cash-on vs cash-off, at n=124 (just under the single-unit
threshold of 125 for a 1-statement body).
"""
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

LOOP = f"out = []\nfor t in list(range(1, {N + 1})):\n    out.append(slow(t))\n"
SINK = "print('N', len(out), 'CALLS', len(CALLS))"

def _arm(nb, ms, cash_on):
    nb.create_notebook([(ON if cash_on else OFF), _defs(ms), LOOP, SINK])
    nb.start_kernel(with_cash=cash_on); nb.run_all()
    t0 = time.perf_counter(); nb.run_cells([3, 4]); warm = time.perf_counter() - t0
    return {"warm_ms": round(warm * 1000), "sink": nb.get_output(4)}

def _pair(nb, ms):
    return {"cash_off": _arm(nb, ms, False), "cash_on": _arm(nb, ms, True)}

def test_band_0_1ms(nb_runner):
    print("BAND_0_1 " + json.dumps(_pair(nb_runner, 0.1)))

def test_band_1ms(nb_runner):
    print("BAND_1 " + json.dumps(_pair(nb_runner, 1)))

def test_band_2_5ms(nb_runner):
    print("BAND_2_5 " + json.dumps(_pair(nb_runner, 2.5)))
