"""Batch 201 – Assertion and debugging print interaction tests.

Tests editing assert statements, debug prints, and
conditional debugging output.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestAssertEdits:
    """Editing assert statements."""

    def test_assert_pass_then_fail(self, nb_runner):
        """Assert passes, then edit to make it fail."""
        nb_runner.create_notebook([
            "x = 10  # assert source",
            "assert x > 5, f'Expected x > 5, got {x}'\nprint(f'x = {x} (ok)')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 10 (ok)" in nb_runner.get_output(2)

        # Change x to make assert fail
        nb_runner.set_cell_source(1, "x = 3  # assert source fail")
        with pytest.raises(CellExecutionError):
            nb_runner.run_all()

    def test_edit_assert_condition(self, nb_runner):
        """Edit the assert condition."""
        nb_runner.create_notebook([
            "val = 42  # assert cond source",
            "assert val == 42\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Tighten assertion
        nb_runner.set_cell_source(
            2, "assert val > 0 and val < 100\nprint(f'val = {val} (in range)')"
        )
        nb_runner.run_all()
        assert "val = 42 (in range)" in nb_runner.get_output(2)


class TestDebugPrintEdits:
    """Editing debug print patterns."""

    def test_edit_debug_format(self, nb_runner):
        """Edit the debug print format."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]  # debug print source",
            "print(f'len={len(data)} sum={sum(data)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3 sum=6" in nb_runner.get_output(2)

        # Change to more detailed format
        nb_runner.set_cell_source(
            2, "print(f'data={data} len={len(data)} min={min(data)} max={max(data)}')"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "data=[1, 2, 3]" in out
        assert "min=1" in out
        assert "max=3" in out

    def test_add_remove_debug_prints(self, nb_runner):
        """Edit output content between runs."""
        nb_runner.create_notebook([
            "a = 5\nb = 10  # debug prints source",
            "c = a + b",
            "print(f'a={a} b={b} c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "a=5 b=10 c=15" in out

        # Change print format
        nb_runner.set_cell_source(3, "print(f'sum={c}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "sum=15" in out2
