"""``with cash.assume_safe():`` in a real kernel.

At the top of a cell the block is one statement, and it waives as
``# @cash:assume-safe`` would for that statement: a call that writes a file
is cached, and a re-run does not repeat it. Runs are counted in a file only
the test reads, so the count is what really ran. The control is the same
cell without the block, which runs every time.
"""

import pytest

pytestmark = [pytest.mark.integration]


def _cells(counter, body: str) -> list[str]:
    return [
        "import cash, os, time\n"
        "def save(v):\n"
        "    time.sleep(0.25)\n"
        f"    fd = os.open(r'{counter}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "    with open(r'" + str(counter) + ".out', 'w') as fh:\n"
        "        fh.write(str(v))\n"
        "    return v",
        body,
    ]


def test_a_top_level_block_caches_a_statement_that_writes(nb_runner, tmp_path):
    counter = tmp_path / "runs"
    nb_runner.create_notebook(
        _cells(counter, "r = save(7)\nprint(f'r = {r}')")
        + ["with cash.assume_safe():\n    s = save(8)\nprint(f's = {s}')"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "r = 7" in nb_runner.get_output(2) and "s = 8" in nb_runner.get_output(3)
    nb_runner.run_cell(2)
    assert counter.read_bytes() == b"xxx", "control: without the block the write runs every time"
    nb_runner.run_cell(3)
    assert "s = 8" in nb_runner.get_output(3)
    assert counter.read_bytes() == b"xxx"


def test_a_cached_function_defined_in_a_cell_honours_the_block(nb_runner, tmp_path):
    # A file write is a finding cash reports (`impure_call`); the block
    # waives it, and the same write outside a block is reported.
    out = tmp_path / "written"
    nb_runner.create_notebook(
        [
            "import cash, warnings\nwarnings.simplefilter('always')",
            "@cash.cache\n"
            "def waived(n):\n"
            "    with cash.assume_safe():\n"
            f"        with open(r'{out}', 'w') as fh:\n"
            "            fh.write(str(n))\n"
            "    return n * 2\n"
            "print(waived(2))",
            "@cash.cache\n"
            "def plain(n):\n"
            f"    with open(r'{out}', 'w') as fh:\n"
            "        fh.write(str(n))\n"
            "    return n * 2\n"
            "print(plain(2))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    # The warning ends with the link to its section. Raw output: the filter
    # drops a cash warning's line, and by default the link is on that line.
    assert "#impure-side-effects" not in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)
    assert "#impure-side-effects" in nb_runner.get_raw_output(3), "control: the same code without the block warns"


_ENV_CELL = (
    "@cash.cache\n"
    "def {name}(v):\n"
    "    fd = os.open(r'{counter}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "{write}"
    "    return v * 2\n"
    "print({name}(3), {name}(3))"
)


@pytest.mark.parametrize(
    ("write", "stored"),
    [
        ("    with cash.assume_safe():\n        os.putenv('CASH_TEST_KERNEL_ENV', str(v))\n", True),
        ("    os.environ['CASH_TEST_KERNEL_ENV'] = str(v)  # @cash:assume-safe\n", True),
        ("    with cash.assume_safe():\n        pass\n    os.environ['CASH_TEST_KERNEL_ENV'] = str(v)\n", False),
    ],
    ids=["putenv-block", "environ-comment", "outside-the-block"],
)
def test_a_cached_function_s_waived_environment_write_is_stored(nb_runner, tmp_path, write, stored):
    """A ``@cash.cache`` function in a cell that changes the environment is
    stored when the write is waived; the same write outside the block keeps
    every call running."""
    counter = tmp_path / "runs"
    nb_runner.create_notebook(["import cash, os", _ENV_CELL.format(name="setenv", counter=counter, write=write)])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "6 6" in nb_runner.get_output(2)
    assert counter.read_bytes() == (b"x" if stored else b"xx")
    assert ("#impure-observed-effects" in nb_runner.get_raw_output(2)) is not stored
