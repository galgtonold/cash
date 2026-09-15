"""An input the kernel already holds, current, is used as it is.

Before a cell runs, cash simulates the cells above it and leaves alone every
input whose value is what that simulation expects. Two checks overrode this
and rebuilt a value that was already current -- restored it from the cache,
or re-ran what produced it:

* A cell that writes an input it also reads (``df['a'] = df['a'] + 1``,
  ``df = df.rename(...)``) is checked for holding its own earlier output, so a
  re-run does not apply it twice. The check compared the value with what the
  *last statement that wrote it* had read -- the cell's starting state only if
  that statement is in this cell. When it is in a cell above, a plain first
  run looked stale: round 23's r23s4 restored its 5,030-article frame this way
  on every Run All.
* A cell that adds a column to a frame from above (``docs['topic'] = ...``)
  leaves that frame ahead of the simulation. Re-running the cell read this as
  an unsaved edit upstream and rebuilt everything derived from the frame
  (r23s4: the model and vocabulary behind its topic table).

Observed with a mark set on the live value from outside the notebook: a value
rebuilt by cash is a different object and does not carry it.
"""
import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

ON = "import cash\n%cash_on"


# Through ``globals()``: a peek runs under cash too, and one that names the
# variable is an execution reading it -- cash brings it up to date first.
def _mark(nb_runner, name):
    nb_runner.peek(f"globals()[{name!r}].attrs.__setitem__('live', 1)")


def _marked(nb_runner, name):
    return nb_runner.peek(f"globals()[{name!r}].attrs.get('live')") == "1"


def test_a_cell_writing_into_an_input_uses_the_live_value(nb_runner):
    nb_runner.create_notebook([
        ON,
        "import pandas as pd\ndf = pd.DataFrame({'a': range(1000)})\ndf['b'] = df['a'] * 2",
        "df['a'] = df['a'] + 1\nprint('SUM', int(df['a'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    _mark(nb_runner, "df")

    nb_runner.run_cell(3)

    assert "SUM 500500" in nb_runner.get_output(3)
    assert _marked(nb_runner, "df"), "df was rebuilt though cell 2 had just made it"


def test_a_cell_reassigning_an_input_uses_the_live_value(nb_runner):
    nb_runner.create_notebook([
        ON,
        "import pandas as pd\ndf = pd.DataFrame({'a': range(1000)})\ndf = df[df['a'] % 2 == 0]",
        "df = df.rename(columns={'a': 'x'})\nprint('N', len(df))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    _mark(nb_runner, "df")

    nb_runner.run_cell(3)

    assert "N 500" in nb_runner.get_output(3)
    # rename carries a frame's attrs to its result
    assert _marked(nb_runner, "df"), "df was rebuilt though cell 2 had just made it"


def test_rerunning_a_cell_that_adds_a_column_keeps_what_came_from_the_frame(nb_runner):
    nb_runner.create_notebook([
        ON,
        "import pandas as pd\ndocs = pd.DataFrame({'a': range(1000)})",
        "feat = docs['a'] * 2",
        "docs['topic'] = feat % 3\nprint('TOPICS', int(docs['topic'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    _mark(nb_runner, "feat")

    nb_runner.run_cell(4)

    assert nb_runner.get_output(4).count("TOPICS 999") == 1
    assert _marked(nb_runner, "feat"), "feat was rebuilt though nothing above it changed"


# What the first check is for: a re-run does not apply the cell twice.

def test_rerunning_a_cell_writing_into_an_input_does_not_apply_it_twice(nb_runner):
    nb_runner.create_notebook([
        ON,
        "import pandas as pd\ndf = pd.DataFrame({'a': range(1000)})\ndf['b'] = df['a'] * 2",
        "df['a'] = df['a'] + 1\nprint('SUM', int(df['a'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(3)

    assert "SUM 500500" in nb_runner.get_output(3)
    assert nb_runner.peek("int(df['a'].sum())") == "500500"


def test_rerunning_a_cell_reassigning_an_input_does_not_apply_it_twice(nb_runner):
    nb_runner.create_notebook([
        ON,
        "import pandas as pd\ndf = pd.DataFrame({'a': range(1000)})\ndf = df[df['a'] % 2 == 0]",
        "df = df.iloc[1:]\nprint('N', len(df))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(3)

    assert "N 499" in nb_runner.get_output(3)
    assert nb_runner.peek("len(df)") == "499"


def test_rerunning_a_cell_after_a_later_cell_wrote_its_input(nb_runner):
    """Re-running lands where a clean run to this cell lands: ``df`` is put
    back to its state at the cell's start, not left with the later cell's
    write. (Read through ``globals()``: a peek naming ``df`` is a cell at the
    notebook's end and would bring ``df`` up to date through cell 4.)"""
    nb_runner.create_notebook([
        ON,
        "import pandas as pd\ndf = pd.DataFrame({'a': range(1000)})\ndf['b'] = df['a'] * 2",
        "df['a'] = df['a'] + 1\nprint('SUM', int(df['a'].sum()))",
        "df['a'] = df['a'] * 10",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(3)

    assert "SUM 500500" in nb_runner.get_output(3)
    assert nb_runner.peek("int(globals()['df']['a'].sum())") == "500500"
