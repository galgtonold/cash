"""IPython keeps the traceback of a cell's error in its history
(``history_manager.exceptions``); under ``%cash_on`` it is the one the user
was shown.

Cash ends a failed cell by re-raising the error through IPython's
``run_cell``, which formats it once more for that record. The exception
still carried cash's own frames (the cell executor, the statement steps), so
the record held a traceback through cash's internals, and formatting it took
a syntax-highlighting lexer per frame: about 40 ms on every failed cell, ten
times plain's. The record is now the shown traceback: the cell's line and
the user's frames below it.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(120)]

_CASH_FRAMES = ("cell_executor.py", "magics.py", "processor.py", "error_display.py", "_drive")


def _last_recorded_traceback(nb_runner) -> str | None:
    """The history's record of the last failed cell, as plain text; None
    when this IPython keeps none."""
    text = nb_runner.peek(
        "(lambda h: None if not getattr(h, 'exceptions', None) else "
        "__import__('re').sub(r'\\x1b\\[[0-9;]*m', '', "
        "'\\n'.join(h.exceptions[max(h.exceptions)]['traceback'])))"
        "(get_ipython().history_manager)"
    )
    return None if text == "None" else text


def test_the_record_of_a_failed_cell_holds_its_own_frames(nb_runner):
    nb_runner.create_notebook(
        [
            "def helper(v):\n    return v + undefined_name",
            "x = 1\ny = helper(x)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(2)

    record = _last_recorded_traceback(nb_runner)
    if record is None:
        pytest.skip("this IPython keeps no tracebacks in its history")
    # The user's frames, as shown: the failing line and the helper's.
    assert "y = helper(x)" in record, record
    assert "return v + undefined_name" in record, record
    assert "NameError" in record, record
    leaked = [frame for frame in _CASH_FRAMES if frame in record]
    assert not leaked, f"cash's frames {leaked} in the history's record:\n{record[:2000]}"
