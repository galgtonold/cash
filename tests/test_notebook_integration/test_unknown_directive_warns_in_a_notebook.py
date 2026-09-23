"""A removed ``# @cash:`` spelling is reported in the cell that uses it.

The unit tests in ``tests/test_notebook/test_unknown_directive_warns.py`` pin
the message and the once-per-name ledger; this checks the warning actually
reaches the notebook's output, where the person who wrote the comment looks.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration]


def test_an_old_nocache_spelling_warns_in_the_cell(nb_runner):
    nb_runner.create_notebook(
        [
            "import time",
            textwrap.dedent("""\
                # @cash:nocache
                stamp = time.perf_counter()
                print("stamp", stamp)
            """),
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(2)
    assert "[ANNOT-UNKNOWN-DIRECTIVE]" in out, out
    # Blamed on the cell's line, not on ipykernel's run_cell frame.
    assert "<cash>:1: CashCacheIneffectiveWarning" in out, out
    assert "ipkernel" not in out, out
    assert "did you mean `# @cash:no-cache`?" in out, out

    nb_runner.run_cell(2)
    assert "ANNOT-UNKNOWN-DIRECTIVE" not in nb_runner.get_raw_output(2), "warned twice for one name"
