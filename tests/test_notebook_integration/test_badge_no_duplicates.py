"""Integration tests verifying that badge rendering does not produce duplicate outputs.

The badge system uses ``display(HTML(...), display_id=..., update=True)`` to
update a single output area in-place.  A previous bug caused a background
timer thread to call ``publish_display_data`` from a non-main thread, which
JupyterLab treated as **new** output areas instead of updates — resulting in
dozens of duplicate progress badges during long-running statements.

The fix disables the background badge timer entirely.  These tests verify
that each cell execution produces exactly **one** HTML ``display_data``
output (the badge), regardless of how many statements the cell contains.

In nbclient the ``update_display_data`` kernel message correctly updates
existing outputs in-place (it doesn't append), so we can assert the output
count.  If someone accidentally re-introduces a thread or extra
``display()`` calls, the count will exceed 1 and these tests will fail.
"""
import pytest

pytestmark = pytest.mark.badges


def _count_html_display_outputs(cell) -> int:
    """Count the number of display_data outputs containing text/html in a cell."""
    count = 0
    for output in cell.get('outputs', []):
        if output.output_type == 'display_data':
            data = output.get('data', {})
            if 'text/html' in data:
                count += 1
    return count


class TestBadgeNoDuplicates:
    """Verify that badge rendering produces exactly one HTML output per cell."""

    def test_single_statement_cell_one_badge(self, nb_runner):
        """A cell with a single statement should have exactly 1 badge output."""
        nb_runner.create_notebook([
            "x = 42",
            "print(f'x={x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Cell 1 (x = 42) and Cell 2 (print) should each have at most 1 HTML badge
        for cell_num in [1, 2]:
            cell = nb_runner.get_cell(cell_num)
            html_count = _count_html_display_outputs(cell)
            assert html_count <= 1, (
                f"Cell {cell_num} has {html_count} HTML display_data outputs, expected <= 1. "
                f"This may indicate duplicate badge rendering."
            )

    def test_multi_statement_cell_one_badge(self, nb_runner):
        """A cell with multiple statements should still have exactly 1 badge output.

        The badge is created once and updated in-place for each statement.
        """
        nb_runner.create_notebook([
            "import time",
            (
                "a = 1\n"
                "b = a + 1\n"
                "c = b + 1\n"
                "d = c + 1\n"
                "e = d + 1\n"
                "print(f'e={e}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert 'e=5' in output

        cell = nb_runner.get_cell(2)
        html_count = _count_html_display_outputs(cell)
        assert html_count <= 1, (
            f"Multi-statement cell has {html_count} HTML display_data outputs, expected <= 1. "
            f"Badge updates should use update_display_data (in-place), not new display_data."
        )

    def test_slow_statement_no_duplicate_badges(self, nb_runner):
        """A cell with a slow statement should not produce timer-thread badge duplicates.

        This is the core regression test: the old bug was that a background
        timer thread would fire publish_display_data every ~2 seconds, creating
        new output areas.  With the timer disabled, only main-thread badge
        renders should appear.
        """
        nb_runner.create_notebook([
            "import time",
            (
                "# Slow statement that previously triggered timer badge duplicates\n"
                "time.sleep(3)\n"
                "result = 'done'"
            ),
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert 'result=done' in output

        # The slow cell should still only have 1 badge
        cell = nb_runner.get_cell(2)
        html_count = _count_html_display_outputs(cell)
        assert html_count <= 1, (
            f"Slow cell has {html_count} HTML display_data outputs, expected <= 1. "
            f"Timer thread may be creating duplicate badge outputs."
        )

    def test_slow_interior_statement_no_duplicate_badges(self, nb_runner):
        """The slow statement is not the cell's FIRST one.

        Every other test in this file walks one of two shapes: a lone slow
        statement, or a run of fast ones.  Neither reaches this defect, and
        the real tour notebook -- where a cold ``import matplotlib.pyplot``
        and a sleeping ``@cash.cache`` call both sit mid-cell -- stored two
        badges per cell while every test here stayed green.

        The mechanism needs both halves of the shape.  A statement executes
        inside ``IPython.utils.capture.capture_output(display=True)``, which
        swaps ``shell.display_pub`` for a ``CapturingDisplayPublisher``
        PROCESS-wide (not per-thread).  The progress timer fires on a
        background thread while that swap is in place, so its RUNNING badge
        is swallowed into the STATEMENT's captured outputs instead of going
        to the frontend -- and cash then replays those with
        ``publish_display_data(data, metadata)``, which drops ``transient``
        and ``update``.  The badge lands as a second, display_id-less output
        that no later update can ever reach: a frozen RUNNING badge stored
        for good beside the cell's real DONE badge.
        """
        nb_runner.create_notebook([
            (
                "import time\n"
                "ready = 1\n"
                "time.sleep(1.5)\n"
                "print(f'ready={ready}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert 'ready=1' in nb_runner.get_output(1)

        cell = nb_runner.get_cell(1)
        html_count = _count_html_display_outputs(cell)
        assert html_count <= 1, (
            f"A cell whose slow statement is not its first has {html_count} HTML "
            f"display_data outputs, expected <= 1. A progress badge published "
            f"from the timer thread was captured as the statement's own output "
            f"and replayed without its display_id."
        )

    def test_second_run_cached_one_badge(self, nb_runner):
        """On second run (cached), cells should still have exactly 1 badge."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a * 2",
            "print(f'b={b}')",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        assert 'b=20' in nb_runner.get_output(3)

        # Second run — should use cache
        nb_runner.run_all()
        assert 'b=20' in nb_runner.get_output(3)

        # Check badge count for each cell on second run
        for cell_num in [1, 2, 3]:
            cell = nb_runner.get_cell(cell_num)
            html_count = _count_html_display_outputs(cell)
            assert html_count <= 1, (
                f"Cell {cell_num} (cached run) has {html_count} HTML display_data outputs, "
                f"expected <= 1."
            )

    def test_self_assignment_cell_one_badge(self, nb_runner):
        """Self-assignment cells (df = df.something()) should have 1 badge."""
        nb_runner.create_notebook([
            "data = list(range(100))",
            (
                "data = sorted(data, reverse=True)\n"
                "total = sum(data)\n"
                "print(f'total={total}')"
            ),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert 'total=4950' in nb_runner.get_output(2)

        cell = nb_runner.get_cell(2)
        html_count = _count_html_display_outputs(cell)
        assert html_count <= 1, (
            f"Self-assignment cell has {html_count} HTML display_data outputs, expected <= 1."
        )
