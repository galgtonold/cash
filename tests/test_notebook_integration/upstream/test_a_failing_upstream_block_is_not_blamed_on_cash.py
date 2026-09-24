"""When re-running a user's block fails, say whose failure it is.

Found by replaying a real session without openpyxl installed: a cell doing

    with pd.ExcelWriter(path) as xw:
        summary.to_excel(xw, sheet_name="stores")

raised ModuleNotFoundError, and the NEXT cell then warned

    [NOTEBOOK-BAILOUT] cash hit an internal error and stepped aside:
    ModuleNotFoundError: No module named 'openpyxl' ... Nothing in your code
    caused this and re-running is safe ... Please report it

A user reads that and files a bug against cash. What actually happened: the
next cell needed upstream state, cash replayed the ``with`` block to rebuild it,
and the block failed again for the same reason as the first time -- a package
the environment does not have.

A plain upstream statement already reports this properly: the processor's error
becomes an ``UpstreamStateError`` naming the statement. The control-structure
branch re-raises the original exception instead, and the handler below it listed
the types user code was expected to raise (RuntimeError, NameError, KeyError,
TypeError, ValueError). ModuleNotFoundError is not among them, so it escaped
into the catch-all that exists for cash's own bugs.

Here the block reads a file that is deleted before the replay -- a
FileNotFoundError, equally outside that list, and it needs no uninstalled
package. The first run succeeds, so the notebook gets into the state that
matters: a block cash believes it can re-run, which then cannot run.
"""

PIN = "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n"
SETUP = "import cash\n%load_ext cash\n%cash_badge print\n" + PIN + "%cash_on"

CELLS = [
    SETUP,
    "from pathlib import Path\nPath('side.txt').write_text('hello there')\nbase = sum(i*i for i in range(1_500_000))",
    "with open('side.txt') as fh:\n    contents = fh.read()\nloaded = len(contents)",
    "answer = base + loaded\nprint('ANSWER', answer)",
]

# The edit that forces the rebuild AND takes the file away, so replaying the
# block above raises where it succeeded before.
REMOVES_THE_FILE = (
    "from pathlib import Path\nimport os\nos.remove('side.txt')\nbase = sum(i*i for i in range(1_500_001))"
)


def _rebuild(nb_runner) -> str:
    """Run the notebook, take the file away, ask for the last cell.

    Returns whatever cash ends up telling the user -- the cell's output if it
    survives, the raised error's text if it does not. Stopping the cell IS the
    right answer here (the alternative is running it against upstream state cash
    could not rebuild); what matters is what the message says.
    """
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, REMOVES_THE_FILE)
    try:
        nb_runner.run_cells([4])
    except Exception as exc:  # the message is the point
        return f"{type(exc).__name__}: {exc}"
    return nb_runner.get_output(4)


def test_a_block_that_cannot_be_replayed_is_not_called_a_cash_bug(nb_runner):
    told = _rebuild(nb_runner)
    assert "NOTEBOOK-BAILOUT" not in told, f"cash blamed itself for a block it could not replay:{chr(10)}{told}"
    assert "Nothing in your code caused this" not in told, told


def test_the_block_that_failed_is_named(nb_runner):
    """Whatever cash says, it has to point at the block."""
    told = _rebuild(nb_runner)
    assert "side.txt" in told, f"the failure names neither the file nor the block:{chr(10)}{told}"
    assert "FileNotFoundError" in told, told


def test_a_quoted_name_in_the_message_does_not_break_the_cell(nb_runner):
    """The message is re-raised inside the user's cell as generated code.

    It used to be interpolated into a triple-quoted literal, and Python quotes
    names in its own messages -- "No such file or directory: 'side.txt'" ends
    with a quote, which closed the literal early. The cell then died with
    `SyntaxError: unterminated string literal` from code cash wrote, with the
    real failure nowhere in sight.
    """
    told = _rebuild(nb_runner)
    assert "SyntaxError" not in told, told
    assert "unterminated" not in told, told
