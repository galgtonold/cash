"""A classmethod reached as ``module.Class.method`` keys its class's constants.

``helpers.Model.run(x)`` with ``run`` a classmethod reading ``cls.factor``
resolves to a method bound to the class. The walk keyed only the method's
function, and the class it is bound to reached no channel, so editing
``factor = 2`` to ``5`` was served the old result. ``Model.run(x)`` after
``from helpers import Model`` was keyed, because the class is read by name.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = [pytest.mark.timeout(300)]

HELPERS = """
class Model:
    factor = {factor}

    @classmethod
    def run(cls, x):
        return x * cls.factor
"""

CALLERS = {
    "module_attribute": "import helpers\n\ndef call(x):\n    return helpers.Model.run(x)\n",
    "bound_alias": "from helpers import Model\nrun = Model.run\n\ndef call(x):\n    return run(x)\n",
}

JOB = """
import sys
import cash
from caller import call

@cash.cache
def f(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return call(x)

print(f(3))
"""


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30
    os.utime(path, (past, past))


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.mark.parametrize("caller", sorted(CALLERS))
def test_editing_the_class_constant_recomputes(tmp_path, caller):
    _write(tmp_path / "helpers.py", HELPERS.format(factor=2))
    _write(tmp_path / "caller.py", CALLERS[caller])
    _write(tmp_path / "job.py", JOB)
    assert _run(tmp_path) == ("6", True)
    assert _run(tmp_path) == ("6", False), "the unedited run did not hit"
    _write(tmp_path / "helpers.py", HELPERS.format(factor=5))
    assert _run(tmp_path) == ("15", True), "the class constant's edit was served the old result"
