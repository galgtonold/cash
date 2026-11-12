"""A cell IPython only refuses, or runs as a magic that touches no Python,
fingerprints none of the tracked values.

Such a cell did not parse for the upstream check, which stepped aside and had
IPython run it as written (``running_as_written``): a fingerprint of every
tracked value before and after the cell. With a list of 87,192 frames in the
namespace that was over a minute for a typo plain Python reports in 10 ms,
and the same for an automagic ``pip install x`` line.

The fingerprints are counted in the kernel, through the name the statement
processor calls (``mutation_fingerprint``), not timed. The error and the
magic's output are what plain IPython shows.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(240)]

_DATA = "data = [[i] for i in range(20_000)]"

# Wraps the name the statement processor calls, counting each call.
_COUNT = """\
import cash.notebook.statement.processor as fp_site
def counting(value, *args, _real=fp_site.mutation_fingerprint, **kwargs):
    counting.calls += 1
    return _real(value, *args, **kwargs)
counting.calls = 0
fp_site.mutation_fingerprint = counting"""
_REPORT = "print('fingerprints', counting.calls)  # {when}"
_RESTORE = "fp_site.mutation_fingerprint = counting.__kwdefaults__['_real']"


def _calls(text: str) -> int:
    line = next(line for line in text.splitlines() if line.startswith("fingerprints "))
    return int(line.split()[1])


def _error(nb_runner, cell: int) -> tuple[str, str]:
    with pytest.raises(CellExecutionError) as excinfo:
        nb_runner.run_cell(cell)
    return excinfo.value.ename, excinfo.value.evalue


def _run_counted(nb_runner, cells: list[str], probe: str):
    """Run *cells* under cash, then *probe* between two counts: what *probe*
    raised or printed, and the fingerprints taken while it ran."""
    nb_runner.create_notebook(
        [_COUNT, *cells, _REPORT.format(when="before"), probe, _REPORT.format(when="after"), _RESTORE]
    )
    nb_runner.start_kernel()
    n = len(cells)
    try:
        nb_runner.run_cells(list(range(1, n + 3)))
        before = _calls(nb_runner.get_output(n + 2))
        try:
            nb_runner.run_cell(n + 3)
            outcome = ("ok", nb_runner.get_output(n + 3))
        except CellExecutionError as e:
            outcome = (e.ename, e.evalue)
        nb_runner.run_cell(n + 4)
        after = _calls(nb_runner.get_output(n + 4))
    finally:
        nb_runner.run_cell(n + 5)
    return outcome, after - before


def _plain(nb_runner, cells: list[str], probe: str):
    nb_runner.create_notebook([*cells, probe])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_cells(list(range(1, len(cells) + 1)))
    try:
        nb_runner.run_cell(len(cells) + 1)
        outcome = ("ok", nb_runner.get_output(len(cells) + 1))
    except CellExecutionError as e:
        outcome = (e.ename, e.evalue)
    nb_runner.shutdown()
    return outcome


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param("y = data[0] 2", id="syntax-error-reading-a-tracked-name"),
        pytest.param("for i in range(3):\nprint(i)", id="indentation-error"),
    ],
)
def test_a_syntax_error_cell_takes_no_fingerprints(nb_runner, probe):
    """The cell fails with the error plain IPython shows, and no tracked
    value is fingerprinted for it."""
    plain = _plain(nb_runner, [_DATA], probe)
    assert plain[0] in {"SyntaxError", "IndentationError"}, plain

    outcome, taken = _run_counted(nb_runner, [_DATA], probe)

    assert outcome == plain
    assert taken == 0, f"{taken} fingerprints for a cell that cannot run"


def test_an_automagic_line_takes_no_fingerprints(nb_runner):
    """``env NAME=value`` written without its ``%`` (automagic) does what
    plain IPython does, and fingerprints nothing: it runs no Python."""
    probe = "env CASH_UNPARSABLE_PROBE=set"
    check = "import os\nprint(os.environ.get('CASH_UNPARSABLE_PROBE'))"
    plain = _plain(nb_runner, [_DATA], probe)
    assert plain[0] == "ok", plain

    outcome, taken = _run_counted(nb_runner, [_DATA], probe)
    assert outcome == plain
    assert taken == 0, f"{taken} fingerprints for a magic that runs no Python"

    nb_runner.add_cell(check)
    nb_runner.run_cell(len(nb_runner.nb.cells))
    assert "set" in nb_runner.get_output(len(nb_runner.nb.cells))


def test_an_automagic_line_that_runs_python_keeps_its_fingerprints(nb_runner, tmp_path):
    """``run script.py`` without its ``%`` runs the user's Python, which can
    change a tracked value in place: cash still fingerprints around it, and a
    statement reading the value sees the change."""
    script = tmp_path / "grow.py"
    script.write_text("data.append([-1])\n", encoding="utf-8")
    probe = f'run -i "{script}"'

    outcome, taken = _run_counted(nb_runner, [_DATA], probe)
    assert outcome[0] == "ok", outcome
    assert taken > 0

    nb_runner.add_cell("print('rows', len(data))")
    nb_runner.run_cell(len(nb_runner.nb.cells))
    assert "rows 20001" in nb_runner.get_output(len(nb_runner.nb.cells))
