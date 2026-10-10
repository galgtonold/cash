"""A cheap cell over a big array is not cached when its RAM hit costs more than running it.

Measured: three cells over 80 MB arrays took 2.37 s per unchanged Run All
against 0.30 s plain. The fitted model priced each RAM hit at ~20 ms; the
copy measured 200-800 ms on the busy machine. The RAM tier now times its own
copies, and the store prices a hit at that speed (`memory_backend.hit_seconds`).

The copy speed is pinned in the kernel, after the cell that builds the array
(whose own store would otherwise time a real copy into it): slow, and the
cell re-runs; fast, the control, and the same cell is restored. Counted by a
tee on ``StatementProcessor.process_statement``, not timed.
"""

import ast

import pytest

pytest.importorskip("numpy")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

_TEE = """
import cash.notebook.statement.processor as _p
C = _p.StatementProcessor
if not hasattr(C, "_test_orig"):
    C._test_orig = C.process_statement
    def _tee(self, code, *a, **k):
        r = C._test_orig(self, code, *a, **k)
        try:
            if "range(1, 30)" in str(code):
                s = r.get("status")
                s = str(getattr(s, "value", s))
                C._test_n[s] = C._test_n.get(s, 0) + 1
        except Exception:
            pass
        return r
    C.process_statement = _tee
C._test_n = {}
"""
_COUNTS = "__import__('cash.notebook.statement.processor', fromlist=['_']).StatementProcessor._test_n"
_SPEED = "__import__('cash.backends.memory_backend', fromlist=['_'])._COPY_SPEED"

CELLS = [
    "import cash\n%cash_on",
    "import numpy as np\nx = np.ones(1_000_000)",
    "y = sum([x * k for k in range(1, 30)])\nprint('y', y[0])",
]


@pytest.mark.parametrize(
    ("bytes_per_second", "expected"),
    [(1e5, {"COMPUTED": 2}), (1e12, {"COMPUTED": 1, "RESTORED": 1})],
    ids=["slow_copy_reruns", "fast_copy_restores"],
)
def test_the_cell_is_kept_only_when_its_hit_is_cheaper(nb_runner, bytes_per_second, expected):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.peek(f"exec({_TEE!r}, {{}})")
    try:
        nb_runner.peek(f"{_SPEED}.__setitem__(slice(None), [{bytes_per_second!r}])")
        for _ in range(2):
            nb_runner.run_cell(3)
            assert "y 435.0" in nb_runner.get_output(3)
        counts = ast.literal_eval(nb_runner.peek(_COUNTS))
        assert counts == expected
    finally:
        nb_runner.peek(
            "(lambda C: (setattr(C, 'process_statement', C._test_orig), delattr(C, '_test_orig')) "
            "if hasattr(C, '_test_orig') else None)"
            "(__import__('cash.notebook.statement.processor', fromlist=['_']).StatementProcessor)"
        )
        # Forget the pinned speed: a reused kernel would price every later
        # test's RAM hits at it.
        nb_runner.peek(f"{_SPEED}.clear()")
