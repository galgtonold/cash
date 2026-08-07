"""MEASUREMENT: where is the break-even body size?

Round 1 measured per-call store ~0.7ms and hit ~0.8ms against a 5ms body
(caching wins ~6x). Lowering _COST_FLOOR_S is only safe down to the point
where the hit cost still beats re-running the body. Below that, caching a
call makes the notebook SLOWER -- which is what the 10ms floor exists to
prevent. This locates that point instead of guessing it.
"""
import json, time

ON = "import cash\n%cash_on\n"
OFF = "import cash\n"
FLOOR0 = "import cash.notebook.call_unit as cu\ncu._COST_FLOOR_S = 0.0\n"
N = 124

def _defs(ms):
    return (
        "import time\nCALLS = []\n"
        "def busy(ms):\n"
        "    t = time.perf_counter() + ms/1000.0\n"
        "    while time.perf_counter() < t:\n        pass\n"
        "def slow(x):\n    CALLS.append(x)\n"
        f"    busy({ms})\n    return x * 2 + 1\n"
    )

LOOP = f"out = []\nfor e in range({N}):\n    out.append(slow(e))\n"
SINK = "print('NCALLS', len(CALLS))"

def _arm(nb, ms, floor0, cash_on=True):
    cells = ([ON, FLOOR0, _defs(ms), LOOP, SINK] if floor0
             else [(ON if cash_on else OFF), _defs(ms), LOOP, SINK])
    li, si = (4, 5) if floor0 else (3, 4)
    nb.create_notebook(cells); nb.start_kernel(with_cash=cash_on)
    nb.run_all()
    t0 = time.perf_counter(); nb.run_cells([li, si]); warm = time.perf_counter() - t0
    return {"warm_ms": round(warm*1000), "calls": nb.get_output(si)}

def test_body_0_1ms(nb_runner):
    r = {"oracle": _arm(nb_runner, 0.1, False, cash_on=False),
         "floor0": _arm(nb_runner, 0.1, True)}
    print("F_0_1MS " + json.dumps(r))

def test_body_1ms(nb_runner):
    r = {"oracle": _arm(nb_runner, 1, False, cash_on=False),
         "floor0": _arm(nb_runner, 1, True)}
    print("F_1MS " + json.dumps(r))

def test_body_2ms(nb_runner):
    r = {"oracle": _arm(nb_runner, 2, False, cash_on=False),
         "floor0": _arm(nb_runner, 2, True)}
    print("F_2MS " + json.dumps(r))
