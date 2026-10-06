"""Module data is checked for outside changes only before a cell it can reach.

Once ``v = helpers.score(3)`` had read a 128 MB table, every later cell --
``x = 1`` among them -- hashed the whole table first, to see whether it was
changed outside the notebook: 350 ms a trivial cell. A cell that neither
reads the data nor derives an input from a statement that does cannot be
told anything by it; the next cell that can still sees the change.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = "import time\nK = {k}\ndef from_k(x):\n    time.sleep(0.3)\n    return x * K\n"

COUNT = (
    "import cash.notebook.recorded_reads as _rr\n"
    "if not hasattr(_rr, '_real_current'):\n"
    "    _rr._real_current = _rr._current\n"
    "    _rr._hashed = []\n"
    "    _rr._current = lambda kind, label: (_rr._hashed.append(kind), _rr._real_current(kind, label))[1]\n"
)


def test_a_cell_the_data_cannot_reach_does_not_hash_it(nb_runner, tmp_path):
    (tmp_path / "statelib.py").write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(
        ["import statelib", "b = statelib.from_k(10)", "y = 1", "z = y + 1", "u = b + 1\nprint('U', u)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cells([3, 4])
    assert nb_runner.peek("__import__('cash.notebook.recorded_reads').notebook.recorded_reads._hashed.count('mod')") == "0"

    # A change made outside still reaches the cell that derives from the reader.
    nb_runner.peek("setattr(__import__('statelib'), 'K', 5)")
    nb_runner.run_cell(5)
    assert nb_runner.peek("u") == "51", nb_runner.get_raw_output(5)
