"""Module data is hashed only around the cells and statements it can reach.

Once ``v = helpers.score(3)`` had read a 128 MB table, every later cell --
``x = 1`` among them -- hashed the whole table first, to see whether it was
changed outside the notebook (350 ms a trivial cell), and again before and
after each of its statements, to see whether the statement changed it. A cell
that neither reads the data nor derives an input from a statement that does
cannot be told anything by it, and a statement that neither reaches its module
nor names the value cannot change it; the next cell that can still sees an
outside change, and a statement that changes it is still the notebook's own.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = "import time\nTABLE = list(range({k}, {k} + 1000))\ndef from_k(x):\n    time.sleep(0.3)\n    return x * TABLE[0]\n"

COUNT = (
    "import cash.notebook.recorded_reads as rr\n"
    "if not hasattr(rr, 'real_digest'):\n"
    "    rr.real_digest = rr.module_data_digest\n"
    "    rr.hashed = []\n"
    "    def counting(label, value, rr=rr):\n"
    "        rr.hashed.append(label)\n"
    "        return rr.real_digest(label, value)\n"
    "    rr.module_data_digest = counting\n"
)
HASHED = "len(__import__('cash.notebook.recorded_reads').notebook.recorded_reads.hashed)"
UNCOUNT = (
    "import cash.notebook.recorded_reads as rr\n"
    "if hasattr(rr, 'real_digest'):\n"
    "    rr.module_data_digest = rr.real_digest\n"
    "    del rr.real_digest, rr.hashed\n"
)


@pytest.fixture
def counting(nb_runner):
    yield lambda: nb_runner.peek(f"exec({COUNT!r})")
    try:
        nb_runner.peek(f"exec({UNCOUNT!r})")
    except Exception:  # noqa: BLE001 - the kernel may be gone
        pass


def test_a_cell_the_data_cannot_reach_does_not_hash_it(nb_runner, tmp_path, counting):
    (tmp_path / "statelib.py").write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(
        ["import statelib", "b = statelib.from_k(10)", "y = 1", "z = y + 1", "u = b + 1\nprint('U', u)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    counting()
    nb_runner.run_cells([3, 4])
    assert nb_runner.peek(HASHED) == "0"

    # A change made outside still reaches the cell that derives from the reader.
    nb_runner.peek("__import__('statelib').TABLE.__setitem__(0, 5)")
    nb_runner.run_cell(5)
    assert nb_runner.peek("u") == "51", nb_runner.get_raw_output(5)


def test_a_statement_reaching_the_module_is_still_watched(nb_runner, tmp_path, counting):
    (tmp_path / "statelib.py").write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import statelib",
            "b = statelib.from_k(10)",
            "y = 1",
            "statelib.TABLE[0] = 5",
            "arr = statelib.TABLE",
            "arr[0] = 6",
            "c = statelib.from_k(1)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    counting()
    nb_runner.run_cell(3)
    # ``y = 1`` neither reaches the module nor names its data: none is hashed.
    assert nb_runner.peek(HASHED) == "0"
    # A statement that changes it, through the module or a name bound to it,
    # is watched before and after it runs, so the change is the notebook's own.
    for cell in (4, 6):
        before = int(nb_runner.peek(HASHED))
        nb_runner.run_cell(cell)
        assert int(nb_runner.peek(HASHED)) >= before + 2, cell
    nb_runner.run_cell(7)
    assert nb_runner.peek("c") == "6"


BIG_LIB = (
    "import time\nimport numpy as np\nTABLE = np.full(1_000_000, 2.0)\n"
    "def from_k(x):\n    time.sleep(0.3)\n    return float(x * TABLE[0])\n"
)


def test_a_cell_using_what_a_big_table_built_does_not_hash_it(nb_runner, tmp_path, counting):
    """``a = v + 1`` below ``v = helpers.score(3)`` hashed the whole table it
    read, 0.25 s a cell for 256 MB. The reader looks for an outside change;
    the cells using what it built do not hash the table again."""
    (tmp_path / "statelib.py").write_text(BIG_LIB, encoding="utf-8")
    nb_runner.create_notebook(
        ["import statelib", "b = statelib.from_k(10)", "u = b + 1", "w = u + b", "x = 1\nprint('W', w)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    counting()
    nb_runner.run_all()
    after_reader = None
    for cell in range(1, 6):
        nb_runner.run_cell(cell)
        if cell == 2:
            after_reader = int(nb_runner.peek(HASHED))
    assert after_reader is not None and after_reader >= 1, "the reader looks"
    assert int(nb_runner.peek(HASHED)) == after_reader, "a cell below the reader hashed the table"
    assert nb_runner.peek("w") == "41.0"
