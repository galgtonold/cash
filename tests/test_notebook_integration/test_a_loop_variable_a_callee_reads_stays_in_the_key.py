"""A loop variable the callee reads as a global stays in a content-keyed call's key.

Round 25 (r25s3) let a content-keyed call drop the enclosing loop's variables
from its key when its arguments already carry them, so the chosen setting could
be served from the sweep. A loop variable the callee -- or a method of a class
it uses -- reads as a global is not in the arguments, and must stay.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(120)]

CELLS = [
    "import cash\n%cash_on",
    "import time\n"
    "class Scaler:\n"
    "    def apply(self, x):\n"
    "        return x * K\n"
    "def via_function(x):\n"
    "    time.sleep(0.02)\n"
    "    return x + K\n"
    "def via_method(x):\n"
    "    time.sleep(0.02)\n"
    "    return Scaler().apply(x)",
    "out = []\n"
    "for K in [2, 3]:\n"
    "    for i in range(3):\n"
    "        out.append((via_function(i), via_method(i)))\n"
    "print('OUT', out)",
]


def test_each_iteration_sees_its_own_global(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    want = "OUT [(2, 0), (3, 2), (4, 4), (3, 0), (4, 3), (5, 6)]"
    assert want in nb_runner.get_output(3), nb_runner.get_output(3)
