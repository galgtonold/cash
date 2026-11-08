"""A write into a table cash restored reaches the arrays taken from it, as
in a plain run.

With pandas copy-on-write, ``y = df['t']``, ``df = df.copy()``,
``v = y.to_numpy()``, ``y.iloc[0] = 100`` leaves ``v[0] == 100.0``: no other
table shares ``y``'s column, so pandas writes it in place, and ``v`` reads
that memory. A table restored from the cache shared its memory with the
cache's copy, so pandas copied the column before writing it and ``v`` kept
``0.0`` -- after a restart too, where the table came from disk.

After a restart the table read from disk is the notebook's own, and every
case prints what a plain run prints. Within one kernel a served table still
shares the cache's memory; the write then warns NOTEBOOK-SERVED-VIEW
instead of passing silently.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\nimport time\nimport numpy as np\nimport pandas as pd"
# Slow enough to be stored, so the second Run All and the restart serve it.
FRAME = "df = (time.sleep(0.3), pd.DataFrame({'t': [0.0, 1.0, 2.0]}))[1]"

CASES = {
    "to_numpy": [FRAME, "y = df['t']", "df = df.copy()", "v = y.to_numpy()", "y.iloc[0] = 100", "print(v[0])"],
    "values": [FRAME, "y = df['t']", "df = df.copy()", "v = y.values", "y.iloc[0] = 100", "print(v[0])"],
    "frame_to_numpy": [FRAME, "v = df.to_numpy()", "df.iloc[0, 0] = 100", "print(v[0, 0])"],
    "array_in_its_own_stored_cell": [
        FRAME,
        "y = df['t']",
        "df = df.copy()",
        "v = (time.sleep(0.3), y.to_numpy())[1]",
        "y.iloc[0] = 100",
        "print(v[0])",
    ],
    "stored_series": [
        FRAME,
        "y = (time.sleep(0.3), df['t'])[1]",
        "df = df.copy()",
        "v = y.to_numpy()",
        "y.iloc[0] = 100",
        "print(v[0])",
    ],
    # A fitted model keeping a read-only view of its labels (``model._y``),
    # with numpy in place of the model: the labels are their own Series,
    # built in the slow cell that takes the view. cash runs that cell again
    # rather than restore a holder sharing memory with ``y``; this pins that
    # the restart still matches a plain run.
    "labels_held_by_an_object": [
        "y = pd.Series([0.0, 1.0, 2.0])\nheld = (time.sleep(0.3), {'_y': y.to_numpy()})[1]",
        "y.iloc[0] = 100",
        "print(held['_y'][:1])",
    ],
}


def _plain(cells: list[str]) -> str:
    """What the last cell prints when the cells run as plain Python."""
    import contextlib
    import io

    namespace: dict = {}
    exec(SETUP.replace("import cash\n%cash_on\n", ""), namespace)
    out = io.StringIO()
    for cell in cells:
        with contextlib.redirect_stdout(out):
            exec(cell, namespace)
    return out.getvalue().strip().splitlines()[-1]


def _error(nb_runner, cell: int) -> str:
    """The error cell *cell* raised, as shown: its name, message and notes."""
    shown = [o for o in nb_runner.get_cell(cell).get("outputs", []) if o.output_type == "error"]
    return "\n".join(f"{o.ename}: {o.evalue}\n" + "\n".join(o.traceback) for o in shown)


def _write_cell(cells: list[str]) -> int:
    """The 1-based notebook cell (after SETUP) that writes the table."""
    return next(i for i, cell in enumerate(cells) if "= 100" in cell) + 2


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", list(CASES))
def test_after_a_restart_the_write_reaches_what_it_reaches_in_a_plain_run(nb_runner, case):
    cells = CASES[case]
    expected = _plain(cells)
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(len(cells) + 1) == expected, "control: the first run is plain"

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.run_all()

    assert nb_runner.get_output(len(cells) + 1) == expected, nb_runner.get_raw_output(len(cells) + 1)
    assert "NOTEBOOK-SERVED-VIEW" not in nb_runner.get_raw_output(_write_cell(cells))


@pytest.mark.parametrize("case", [case for case in CASES if case != "labels_held_by_an_object"])
def test_a_second_run_all_warns_that_the_array_kept_the_old_values(nb_runner, case):
    """Within one kernel the frame is served from the RAM tier, whose memory
    it shares: the write cannot reach the array, so it warns."""
    cells = CASES[case]
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    write = _write_cell(cells)
    assert "NOTEBOOK-SERVED-VIEW" not in nb_runner.get_raw_output(write), "control: nothing served yet"

    nb_runner.run_all()

    printed = nb_runner.get_output(len(cells) + 1)
    if printed != _plain(cells):
        assert "NOTEBOOK-SERVED-VIEW" in nb_runner.get_raw_output(write), nb_runner.get_raw_output(write)
        assert "'v'" in nb_runner.get_raw_output(write)


def test_a_write_into_a_served_table_with_no_array_taken_does_not_warn(nb_runner):
    """Control: the same write, with nothing reading the old memory."""
    cells = [FRAME, "y = df['t']", "df = df.copy()", "y.iloc[0] = 100", "print(y.iloc[0])"]
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(6) == "100.0"
    assert "NOTEBOOK-SERVED-VIEW" not in nb_runner.get_raw_output(5), nb_runner.get_raw_output(5)


@pytest.mark.fresh_kernel
def test_a_write_past_pandas_into_a_served_table_says_why_it_is_refused(nb_runner):
    """``df['t'].array[0] = 5.0`` writes the column in place in a plain run.
    A table served from the RAM tier is read-only memory the cache shares,
    so the write raises; the error says so. After a restart the table read
    from disk is the notebook's own, and the write works. The write is
    edited before each run, so it runs again rather than being served."""
    cells = [SETUP, FRAME, "df['t'].array[0] = 5.0", "print(df['t'].iloc[0])"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.client.allow_errors = True
    nb_runner.run_all()
    assert nb_runner.get_output(4) == "5.0", "control: the first run is plain"

    # Edited, so the write runs again, into the frame the RAM tier serves.
    nb_runner.set_cell_source(3, "df['t'].array[0] = 6.0")
    nb_runner.run_all()
    raised = _error(nb_runner, 3)
    assert "read-only" in raised and "served from the cache" in raised, raised

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.set_cell_source(3, "df['t'].array[0] = 7.0")
    nb_runner.run_all()
    assert not _error(nb_runner, 3), _error(nb_runner, 3)
    assert nb_runner.get_output(4) == "7.0"
