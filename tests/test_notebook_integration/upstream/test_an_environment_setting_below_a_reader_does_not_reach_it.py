"""An environment variable set in a cell below a reader does not reach the reader.

``os.environ['MODE'] = 'a'``, ``t = os.environ['MODE'] + slow()``,
``os.environ['MODE'] = 'b'``, ``u = t + '!'``: running the last cell keyed
``t`` on the value the environment holds after the whole notebook, ``'b'``,
and ran it again with it. ``u`` was ``'bNone!'`` where a plain kernel and a
top-to-bottom run give ``'aNone!'``. The values a statement read are now the
ones it saw when it ran, as for the data of a local module.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SLOW = "import os, time\ndef slow():\n    time.sleep(0.3)\n    return str(None)\n"


def _cells(var, reader):
    return [
        SLOW + f"os.environ['{var}'] = 'a'",
        reader,
        f"os.environ['{var}'] = 'b'",
        "u = t + '!'\nprint('U', u)",
    ]


@pytest.fixture
def var(tmp_path):
    """A name of its own: the warm kernel keeps its environment between tests."""
    return f"CASH_TEST_MODE_{abs(hash(str(tmp_path))) % 10**8}"


def test_a_setting_below_the_reader(nb_runner, var):
    nb_runner.create_notebook(_cells(var, f"t = os.environ['{var}'] + slow()"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "U aNone!" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "'aNone!'", nb_runner.get_raw_output(4)

    # An edit of the setting above still reaches the reader.
    nb_runner.set_cell_source(1, SLOW + f"os.environ['{var}'] = 'c'")
    nb_runner.run_all()
    assert nb_runner.peek("u") == "'cNone!'", nb_runner.get_raw_output(4)


def test_a_setting_below_a_module_function_reading_it(nb_runner, tmp_path, var):
    (tmp_path / "envlib.py").write_text(f"import os\ndef mode():\n    return os.environ['{var}']\n", encoding="utf-8")
    cells = _cells(var, "import envlib\nt = envlib.mode() + slow()")
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "U aNone!" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "'aNone!'", nb_runner.get_raw_output(4)


def test_after_a_restart(nb_runner, var):
    nb_runner.create_notebook(_cells(var, f"t = os.environ['{var}'] + slow()"))
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert nb_runner.peek("u") == "'aNone!'", nb_runner.get_raw_output(4)
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "'aNone!'", nb_runner.get_raw_output(4)


def test_a_change_from_outside_is_seen(nb_runner, var):
    """Set outside the notebook's cells, as a shell or a launcher would: the
    reader follows it, though a cell below sets the variable too."""
    nb_runner.create_notebook(_cells(var, f"t = os.environ['{var}'] + slow()"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "'aNone!'", nb_runner.get_raw_output(4)

    nb_runner.peek(f"__import__('os').environ.__setitem__('{var}', 'z')")
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "'zNone!'", nb_runner.get_raw_output(4)
