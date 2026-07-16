"""CAS-156: a user error under %cash_on must show a CLEAN traceback.

When a cell has a plain ``SyntaxError`` (or an upstream cell fails at runtime),
cash used to inject its OWN internal frames (``analysis.py``,
``virtual_lineage.py``, ``cell_executor.py``, ``checker.py``) into the traceback
the user saw, plus a spurious "During handling of the above exception, another
exception occurred" chain — making a typo look like cash crashed. Cash-off shows
a clean ~4-line error.

Root cause: ``CellExecutor._resolve_upstream_state`` dispatched IPython's real
``run_cell`` (to surface the user-facing error) from *inside* its own broad
``except`` block, so Python implicitly chained the fresh exception onto cash's
internal one via ``__context__`` and IPython's ultratb rendered the whole chain.
The fix moves the dispatch OUTSIDE the except (into
``_handle_upstream_resolution_failure``), where ``sys.exc_info()`` is already
clear so no accidental chaining occurs.

These tests assert on the rendered traceback text. The harness's
``get_output`` / ``get_raw_output`` do NOT capture error outputs, so we drive the
failing cell under ``pytest.raises(CellExecutionError)`` and inspect
``str(excinfo.value)`` — which contains the full rendered traceback nbclient
propagates from the kernel.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.core, pytest.mark.timeout(120)]

# Cash's own internal frames / the accidental-chain banner. None of these may
# appear when the failure is a plain user error.
_CASH_INTERNAL_MARKERS = [
    "During handling of the above exception",
    "virtual_lineage.py",
    "analyze_code_block",
    "cell_executor.py",
    "checker.py",
]


def _assert_clean(text: str) -> None:
    for marker in _CASH_INTERNAL_MARKERS:
        assert marker not in text, (
            f"traceback leaked cash internals ({marker!r}); "
            f"expected a clean user-facing error. Got:\n{text[:1200]}"
        )


def test_single_cell_syntax_error_is_clean(nb_runner):
    """A plain typo in a single cell shows SyntaxError with no cash frames."""
    nb_runner.create_notebook([
        "x = 1 2",  # complete-but-invalid -> plain SyntaxError
    ])
    nb_runner.start_kernel()

    with pytest.raises(CellExecutionError) as excinfo:
        nb_runner.run_cell(1)
    text = str(excinfo.value)

    assert "SyntaxError" in text, (
        f"expected a SyntaxError to reach the user, got:\n{text[:1200]}"
    )
    _assert_clean(text)


def test_syntax_error_after_upstream_resolution_is_clean(nb_runner):
    """Cell 2 references an upstream var AND has a typo.

    Referencing ``a`` forces cash to run upstream resolution *before* it hits the
    parse failure, exercising the lineage/simulation path — exactly where the old
    code leaked ``virtual_lineage.py`` / ``checker.py`` frames into the chain.
    """
    nb_runner.create_notebook([
        "a = 1",
        "b = a + 2 2",  # references upstream `a`, then a plain SyntaxError
    ])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)

    with pytest.raises(CellExecutionError) as excinfo:
        nb_runner.run_cell(2)
    text = str(excinfo.value)

    assert "SyntaxError" in text, (
        f"expected a SyntaxError to reach the user, got:\n{text[:1200]}"
    )
    _assert_clean(text)


def test_upstream_failure_keeps_intentional_cause_but_not_accidental_chain(nb_runner):
    """Regression for the deliberate CAS-87 / CAS-153 path.

    An upstream edit that makes a downstream cell fail must STILL surface both
    ``UpstreamStateError`` and the real underlying cause (the undefined name) —
    that intentional cause-chain is the whole point of failing the cell loudly.
    But the *accidental* context-chain ("During handling of the above
    exception") that used to expose cash's frames must be gone.
    """
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "print('y=' + str(y))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y=20" in nb_runner.get_output(3)

    # Edit the producer so it now raises a NameError (a typo'd name).
    nb_runner.set_cell_source(1, "x = undefined_name_xyz * 2")

    with pytest.raises(CellExecutionError) as excinfo:
        nb_runner.run_cell(3)
    text = str(excinfo.value)

    # The intentional failure IS surfaced...
    assert "UpstreamStateError" in text, (
        f"expected the upstream failure to surface in the cell, got:\n{text[:1200]}"
    )
    assert "undefined_name_xyz" in text, (
        f"error should name the real underlying cause, got:\n{text[:1200]}"
    )
    # ...but the accidental context-chain that leaked cash's frames is gone.
    assert "During handling of the above exception" not in text, (
        f"the accidental __context__ chain must be suppressed, got:\n{text[:1200]}"
    )
