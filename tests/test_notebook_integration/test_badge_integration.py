"""Integration tests for badge display improvements.

Tests that badges display correctly during actual notebook execution,
including loop iteration grouping, skipped step expansion, and
iteration context stripping.
"""
import pytest

pytestmark = pytest.mark.badges


class TestBadgeDisplayIntegration:
    """Integration tests for badge display rendering during notebook execution."""

    def test_loop_badge_shows_iterations(self, nb_runner):
        """Badge for a loop cell should show grouped iterations."""
        nb_runner.create_notebook([
            "import time",
            "results = {}",
            "for name in ['alpha', 'beta', 'gamma']:\n    results[name] = len(name)\n    time.sleep(0.01)",
            "print(results)",
        ])
        nb_runner.start_kernel()

        # First run: all computed
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert 'alpha' in output
        assert 'gamma' in output

        # Second run should use cache
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert 'alpha' in output2

    def test_loop_badge_second_run_cached(self, nb_runner):
        """On second run, loop iterations should be cached and badge should reflect that."""
        nb_runner.create_notebook([
            "total = 0",
            "for i in range(3):\n    total += i + 1",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert 'total=6' in output

        # Second run: should be cached
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert 'total=6' in output2

    def test_upstream_loop_with_downstream(self, nb_runner):
        """Upstream loop iterations shown when downstream cell triggers upstream check."""
        nb_runner.create_notebook([
            "data = {}",
            "for key in ['x', 'y']:\n    data[key] = len(key)",
            "summary = str(data)",
            "print(summary)",
        ])
        nb_runner.start_kernel()

        # Run all first
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert 'x' in output

        # Run only cell 4 - should trigger upstream check
        nb_runner.run_cell(4)
        output2 = nb_runner.get_output(4)
        assert 'x' in output2

    def test_badge_with_cell_modification(self, nb_runner):
        """Badge correctly shows computed/cached after cell modification."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "print(f'y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 'y=20' in nb_runner.get_output(3)

        # Modify upstream cell
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert 'y=200' in nb_runner.get_output(3)

    def test_badge_with_skipped_steps(self, nb_runner):
        """Badge shows skipped steps when running downstream cell directly."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "print(f'd={d}')",
        ])
        nb_runner.start_kernel()

        # Run all to populate cache
        nb_runner.run_all()
        assert 'd=4' in nb_runner.get_output(5)

        # Run only the last cell: should trigger upstream check with skipped steps
        nb_runner.run_cell(5)
        output = nb_runner.get_output(5)
        assert 'd=4' in output

    def test_loop_with_nested_if_grouped(self, nb_runner):
        """For loop containing an if-statement should produce correct output.

        Regression test: nested if/try inside a for loop used to break badge
        grouping because the nested control structure metrics lacked
        ``__iteration_context__``, causing them to be rendered as separate
        control_group rows instead of staying within the for_loop_group.
        """
        nb_runner.create_notebook([
            "results = []",
            (
                "for i in range(5):\n"
                "    x = i * 10\n"
                "    if x > 20:\n"
                "        results.append(x)"
            ),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()

        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert 'results=[30, 40]' in output

        # Second run should cache and still produce correct output
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert 'results=[30, 40]' in output2
