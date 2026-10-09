"""A method a cell gives a library class is followed by the cells reading it.

The upstream check looks into each class the cells above read once per check
(``callee_reach.one_walk``): a pandas frame read by a hundred cells costs one
look at ``DataFrame``, not a hundred. A cell may give such a class a method of
its own, and a statement reading an instance then reaches that method and the
environment it reads. What one check learned about the class must not carry
into the next one, where the method exists: the reader below follows a change
of the setting as a plain kernel's Run All does.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.fresh_kernel, pytest.mark.timeout(240)]


def _cells(var, value):
    return [
        f"import os, time, fractions\nos.environ['{var}'] = '{value}'\nfr = fractions.Fraction(1, 3)\n"
        "x = [fr + i for i in range(3)]",
        "y = fr * 2",
        f"def where_mode(self):\n    time.sleep(0.3)\n    return os.environ['{var}']\n"
        "fractions.Fraction.where_mode = where_mode",
        "t = fr.where_mode() + '!'",
        "print('T', t, y)",
    ]


@pytest.fixture
def var(tmp_path):
    """A name of its own: a kernel keeps its environment."""
    return f"CASH_TEST_WHERE_{abs(hash(str(tmp_path))) % 10**8}"


def test_a_setting_read_through_the_method_reaches_the_reader(nb_runner, var):
    nb_runner.create_notebook(_cells(var, "a"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "T a! 2/3" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    # Unchanged: the same answer.
    nb_runner.run_all()
    assert nb_runner.peek("t") == "'a!'", nb_runner.get_raw_output(5)

    # The setting edited above: Run All gives what a plain kernel's does.
    nb_runner.set_cell_source(1, _cells(var, "b")[0])
    nb_runner.run_all()
    assert "T b! 2/3" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    assert nb_runner.peek("t") == "'b!'"

    # And when only the reader's cell runs after another edit.
    nb_runner.set_cell_source(1, _cells(var, "c")[0])
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)
    assert nb_runner.peek("t") == "'c!'", nb_runner.get_raw_output(5)


def test_a_change_from_outside_reaches_the_reader(nb_runner, var):
    """Set outside the notebook, as a launcher would: the reader of the
    method follows it, as a plain kernel's Run All does."""
    cells = _cells(var, "a")
    cells[0] = cells[0].replace(f"os.environ['{var}'] = 'a'", f"os.environ.setdefault('{var}', 'a')")
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("t") == "'a!'", nb_runner.get_raw_output(5)

    nb_runner.peek(f"__import__('os').environ.__setitem__('{var}', 'z')")
    nb_runner.run_cell(5)
    assert nb_runner.peek("t") == "'z!'", nb_runner.get_raw_output(5)
    nb_runner.run_all()
    assert nb_runner.peek("t") == "'z!'", nb_runner.get_raw_output(5)
