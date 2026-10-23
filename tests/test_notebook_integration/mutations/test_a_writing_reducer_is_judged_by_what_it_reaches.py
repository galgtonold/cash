"""What a reducer of the user's writes, when cash's pickle runs it, is
judged by the objects it reaches, not by the names it uses.

A ``__getstate__`` that hands a fresh copy of its state to a helper which
edits it (``_drop(self.__dict__.copy(), SKIP)``) changes nothing a
notebook variable holds: it is stored as before, and no variable moves. A
reducer in a module that writes the module's own list writes the
notebook's ``seen = rmod.SEEN`` under another name. And a writer pickled
again by a later statement (``ws = [w, 1]``) is not run again: the reader
below gives what plain Python gives, where no pickle runs it at all.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(240)]

SETUP = "import cash\n%cash_on\nimport time"
STATUSES = (
    "[str(s.get('status')) for s in "
    "get_ipython().magics_manager.magics['line']['cash_status'].__self__._last_cell_metrics['statements']]"
)


def test_a_getstate_handing_a_fresh_copy_to_a_helper_that_edits_it_stays_cached(nb_runner):
    holder = (
        "SKIP = ('cache',)\n"
        "def _drop(d, keys):\n"
        "    for k in keys:\n"
        "        d.pop(k, None)\n"
        "    return d\n"
        "class P:\n"
        "    def __init__(self):\n"
        "        self.rows = list(range(1000))\n"
        "        self.cache = {}\n"
        "    def __getstate__(self):\n"
        "        return _drop(self.__dict__.copy(), SKIP)\n"
        "p = (time.sleep(0.3), P())[1]"
    )
    nb_runner.create_notebook([SETUP, holder, "q = (time.sleep(0.3), len(SKIP))[1]"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert ast.literal_eval(nb_runner.peek(STATUSES)) == ["RESTORED"]
    nb_runner.run_cell(2)
    statuses = ast.literal_eval(nb_runner.peek(STATUSES))
    assert statuses[-1] == "RESTORED", statuses


def test_a_writer_pickled_again_below_is_not_run_again(nb_runner):
    cells = [
        SETUP,
        "cfg = {'k': 1}",
        "class W:\n    def __reduce__(self):\n        cfg['k'] += 1\n        return (W, ())",
        "w = W()",
        "r = (time.sleep(0.3), cfg['k'] * 10)[1]",
        "ws = (time.sleep(0.3), [w, 1])[1]",
        "r2 = (time.sleep(0.3), cfg['k'] * 10)[1]",
        "print('R', r, r2, cfg)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    got = []
    for _ in range(3):
        nb_runner.run_all()
        got.append(nb_runner.get_output(len(cells)).strip())
    assert got == ["R 10 10 {'k': 1}"] * 3, got


def test_a_module_reducer_writing_its_list_moves_the_notebooks_alias(nb_runner, tmp_path):
    (tmp_path / "rmod.py").write_text(
        "SEEN = []\nclass D:\n    def __reduce__(self):\n        SEEN.append(1)\n        return (D, ())\n",
        encoding="utf-8",
    )
    cells = [
        SETUP + f"\nimport sys\nsys.path.insert(0, {str(tmp_path)!r})\nimport rmod",
        "seen = rmod.SEEN",
        "d = (time.sleep(0.3), rmod.D())[1]",
        "k = (time.sleep(0.3), len(seen))[1]",
        "print('K', k, len(seen))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    got = []
    for _ in range(3):
        nb_runner.run_all()
        got.append(nb_runner.get_output(len(cells)).strip())
    # Plain Python never pickles ``d``, so ``SEEN`` stays empty.
    assert got == ["K 0 0"] * 3, got
