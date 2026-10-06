"""A cell holding a magic runs its Python once, as in plain IPython.

IPython ran such a cell on its own, so cash tracked none of its Python, and
the upstream check of a cell below re-ran the Python it read in it: the
cell's file write happened twice. A name a magic bound kept the lineage of
the value before, so a cached statement below served the old result.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

WRITE = "os.write(os.open('log', os.O_WRONLY | os.O_APPEND | os.O_CREAT), b'r')"


@pytest.mark.parametrize(
    "cell",
    [
        f"%env FOO=1\nimport os\n{WRITE}\nx = 1",
        f"%%time\nimport os\n{WRITE}\nx = 1",
        f"%%prun -q\nimport os\n{WRITE}\nx = 1",
        f"%%capture cap\nimport os\n{WRITE}\nx = 1",
        f"import os\ndef f():\n    {WRITE}\n    return 1\nt = %timeit -o -n1 -r1 f()\nx = 1",
    ],
    ids=["line-magic", "time", "prun", "capture", "timeit"],
)
def test_the_body_runs_once(nb_runner, cell):
    nb_runner.create_notebook([cell, "y = x + 1", "z = y + 1"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(3)

    assert nb_runner.peek("z") == "3"
    assert (nb_runner.work_dir / "log").read_text() == "r"


@pytest.mark.parametrize("magic", ["%time x = 2", "x = %time 2"])
def test_a_name_a_magic_binds_is_not_keyed_to_its_old_value(nb_runner, magic):
    nb_runner.create_notebook(
        [
            "import time\ndef slow(v):\n    time.sleep(0.05)\n    return v * 10\nx = 1",
            magic,
            "y = slow(x)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 3])
    nb_runner.run_cells([2, 3])

    assert nb_runner.peek("y") == "20"


def test_a_timed_cell_prints_its_times_and_its_result(nb_runner):
    nb_runner.create_notebook(["%%time\nprint('hi')\nx = 41\nx + 1"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_output(1)
    assert out.count("hi") == 1
    assert out.count("Wall time") == 1
    assert "42" in out
