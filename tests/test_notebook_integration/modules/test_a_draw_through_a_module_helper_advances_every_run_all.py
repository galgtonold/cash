"""A helper of a local module draws from a generator the module holds:
``t = draw()`` gives the next item on every Run All, as plain Python does,
also when cash stores even the cheapest statement.

With every statement stored, ``t = draw()`` was served from the first
run: ``t`` stayed 0 where plain Python gives 0, 1, 2, and the generator
never moved.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

MODULE = "def gen():\n    i = 0\n    while True:\n        yield i\n        i += 1\nG = gen()\ndef draw():\n    return next(G)\n"
CELLS = [
    "import sys, os\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\n"
    "from he_helper import draw",
    "t = draw()",
    "u = t + 1",
]


def test_each_run_all_draws_the_next_item(nb_runner, tmp_path):
    (tmp_path / "he_helper.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    for expected in (0, 1, 2):
        nb_runner.run_all()
        assert (nb_runner.peek("t"), nb_runner.peek("u")) == (str(expected), str(expected + 1))
