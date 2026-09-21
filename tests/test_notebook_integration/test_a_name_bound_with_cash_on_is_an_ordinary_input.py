"""A statement reading a name bound in the ``%cash_on`` cell is cached.

The quickstart's first cell is ``import cash`` then ``%cash_on``, and nearly
everyone puts their imports and paths there too. Cash was not listening when
that cell started, so nothing recorded what ``DATA = Path(...)`` produced: the
name had no lineage, and the first statement to read it was refused as
``NOT CACHED ... - Input variable missing lineage``. Round 27: r27s1's two
loads cached NOTHING until the tester split cell 0 in two; r27s4 got the same
reason on two statements and could not act on it.

Measured 2026-09-21 on the round's build: the first reader of ``DATA`` / ``N``
was refused; one cell further down, an upstream repair had re-run the
``%cash_on`` cell's statements under tracking and everything below cached.
So the gap was one cell wide -- and it was the cell that loads the data.

The simulation reads that cell out of the .ipynb and has a lineage for the
name like any other. The fix adopts it, ONCE, at the first check after
``%cash_on``: before anything is tracked there is no lineage the invalidator
could have dropped on purpose, so there is nothing for the adoption to undo.
Imports are not part of it -- their lineage was already propagated.

Only for a binding that cannot have read anything (constants, literals,
``Path(...)``). The last test is why: the first draft adopted
``RAW = DATA.read_text()`` too, nothing then knew RAW came from a file, and a
changed file left the cells below on the old text. It passes without the fix
and failed with that draft.
"""
import pytest

from conftest import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

# See test_loop_edit_rerun_matrix.py for the canonical explanation: only a
# failure whose text contains "re-ran" is retried. The staleness test is not
# marked -- a wrong value is never a load artifact.
LOAD_SENSITIVE = pytest.mark.flaky(reruns=2, reruns_delay=5, only_rerun=["re-ran"])

HEAD = ("import cash\n%cash_on\n%cash_persist on\n%cash_badge print\n"
        "from pathlib import Path")

#: Worth caching, or the cost floor refuses it for an unrelated reason.
WORK = "sum(i * i for i in range(2_000_000))"

# Concatenated, not %-formatted: the sources are full of literal `%`.
LOAD = "LOADED = DATA.read_text() * N + str(" + WORK + " % 7)"
SHOW = "print('R len=' + str(len(LOADED)))"


def _setup(path, n):
    return HEAD + "\nDATA = Path(r'" + str(path) + "')\nN = " + str(n)


def _data(tmp_path):
    path = tmp_path / "rows.txt"
    path.write_text("abcde", encoding="utf-8")
    return path


@LOAD_SENSITIVE
def test_the_first_reader_is_cached_and_restores(nb_runner, tmp_path):
    data = _data(tmp_path)
    nb_runner.create_notebook([_setup(data, 3), LOAD, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()

    first = nb_runner.get_raw_output(2)
    assert "missing lineage" not in first, (
        "the statement reading DATA and N was refused on the first run:\n"
        + first
    )

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    again = nb_runner.get_raw_output(2)
    assert shows_cached(again), (
        # "re-ran" is what makes LOAD_SENSITIVE retry; keep it.
        "the load re-ran after a restart instead of restoring:\n" + again
    )
    nb_runner.run_cell(3)
    assert "R len=16" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_editing_the_cash_on_cell_still_recomputes(nb_runner, tmp_path):
    """The safety control: adopting a lineage must not freeze the value."""
    data = _data(tmp_path)
    nb_runner.create_notebook([_setup(data, 3), LOAD, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R len=16" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.set_cell_source(1, _setup(data, 5))
    nb_runner.run_cell(1)
    nb_runner.run_cell(3)
    assert "R len=26" in nb_runner.get_output(3), (
        "N changed in the %cash_on cell and the load kept its old value:\n"
        + nb_runner.get_raw_output(3)
    )


def test_editing_the_file_still_recomputes(nb_runner, tmp_path):
    data = _data(tmp_path)
    nb_runner.create_notebook([_setup(data, 3), LOAD, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R len=16" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    data.write_text("abcdefgh", encoding="utf-8")
    nb_runner.run_cell(3)
    assert "R len=25" in nb_runner.get_output(3), (
        "the file DATA names changed and the load kept its old value:\n"
        + nb_runner.get_raw_output(3)
    )


def test_a_value_read_in_the_cash_on_cell_still_follows_its_file(
        nb_runner, tmp_path):
    """A lineage adopted for a value READ from a file must not pin it.

    The adoption gives ``RAW`` the simulation's lineage without anything
    under tracking having read the file. If that also meant nothing recorded
    the file, a changed file would leave every cell below serving the old
    text -- which Restart & Run All would not.
    """
    data = _data(tmp_path)
    setup = _setup(data, 3) + "\nRAW = DATA.read_text()"
    size = "SIZE = len(RAW) + " + WORK + " % 1"
    show = "print('R size=' + str(SIZE))"
    nb_runner.create_notebook([setup, size, show])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R size=5" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    data.write_text("abcdefgh", encoding="utf-8")
    nb_runner.run_cell(3)
    assert "R size=8" in nb_runner.get_output(3), (
        "the file RAW was read from changed and the cells below kept the "
        "old text:\n" + nb_runner.get_raw_output(3)
    )


#: A LOAD in the %cash_on cell: its binding could have read a file, so its
#: lineage is not adopted (above). Instead the first cell needing it re-runs
#: the binding under tracking, once, which records the file too.
LOAD_SETUP_EXTRA = "\nRAW = DATA.read_text()"
SIZE = "SIZE = len(RAW) + " + WORK + " % 1"
SHOW_SIZE = "print('R size=' + str(SIZE))"


@LOAD_SENSITIVE
def test_a_value_loaded_in_the_cash_on_cell_is_cached_below(nb_runner, tmp_path):
    data = _data(tmp_path)
    nb_runner.create_notebook([_setup(data, 3) + LOAD_SETUP_EXTRA, SIZE, SHOW_SIZE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R size=5" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.run_cell(2)
    again = nb_runner.get_raw_output(2)
    assert shows_cached(again) and "missing lineage" not in again, (
        "the statement reading RAW re-ran instead of restoring:\n" + again
    )

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    after = nb_runner.get_raw_output(2)
    assert shows_cached(after), (
        "after a restart the statement reading RAW re-ran instead of "
        "restoring:\n" + after
    )
    nb_runner.run_cell(3)
    assert "R size=5" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_a_value_loaded_in_the_cash_on_cell_follows_its_file_when_re_run_alone(
        nb_runner, tmp_path):
    """Only the reader's own cell is re-run -- no cell below to repair it."""
    data = _data(tmp_path)
    nb_runner.create_notebook([_setup(data, 3) + LOAD_SETUP_EXTRA, SIZE, SHOW_SIZE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    data.write_text("abcdefgh", encoding="utf-8")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "R size=8" in nb_runner.get_output(3), (
        "the file RAW was read from changed and SIZE kept the old text:\n"
        + nb_runner.get_raw_output(2) + nb_runner.get_raw_output(3)
    )
