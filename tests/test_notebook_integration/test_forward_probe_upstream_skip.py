"""Integration tests for forward-probe upstream skip optimization.

When the current cell contains statements that would be cache hits from DISK,
and those cache hits would restore broken upstream variables, the system
should skip unnecessary upstream re-execution.
"""
import pytest

pytestmark = pytest.mark.upstream


class TestForwardProbeUpstreamSkip:
    """Test that upstream auto-execution is skipped when current cell cache hits resolve broken vars."""

    @pytest.mark.timeout(60)
    def test_skip_upstream_when_current_cell_has_disk_hit(self, nb_runner):
        """After first run, a slow current cell statement is on disk.
        On second run (simulated restart), upstream should be skipped because
        the current cell cache hit will restore the needed variables."""
        nb_runner.create_notebook([
            "%cash_on\n%cash_debug on",
            "x = 10",
            # This statement uses x and produces y. It's slow enough to be
            # promoted to disk (>1s threshold).
            "import time\ntime.sleep(1.5)\ny = x * 2\nprint(f'y={y}')",
        ])
        nb_runner.start_kernel()

        # First run: compute everything
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert 'y=20' in output

        # Simulate kernel restart: clear in-memory tracking state
        nb_runner.reset_cash_state()

        # Second run: x is "broken" (not in memory), but cell 3
        # (which needs x and produces y) should be a disk cache hit.
        # The forward probe should detect this and skip upstream re-execution.
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert 'y=20' in output2

    @pytest.mark.timeout(60)
    def test_upstream_still_runs_when_no_disk_cache(self, nb_runner):
        """If current cell statements are NOT on disk, upstream should still run."""
        nb_runner.create_notebook([
            "%cash_on",
            "x = 10",
            # Fast statement — won't be on disk, only RAM
            "y = x * 2\nprint(f'y={y}')",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert 'y=20' in output

        # Simulate kernel restart
        nb_runner.reset_cash_state()

        # Second run: x is broken, current cell is NOT on disk (too fast),
        # so upstream MUST be re-executed to produce correct result
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert 'y=20' in output2

    @pytest.mark.timeout(60)
    def test_no_nameerror_when_probe_resolves_var(self, nb_runner):
        """Regression: forward probe must inject placeholder + lineage so
        statement processor's _check_input_lineage_skip doesn't bail out,
        preventing NameError when the broken var is used as input."""
        nb_runner.create_notebook([
            "%cash_on",
            "import pandas as pd\ndf = pd.DataFrame({'a': range(100)})",
            # Slow mutation of df — will be on disk (>1s).
            # Uses df as input AND produces df as output.
            "import time\ntime.sleep(1.5)\ndf['b'] = df['a'] * 2\nprint(f'cols={list(df.columns)}')",
        ])
        nb_runner.start_kernel()

        # First run: populate cache
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert 'b' in output

        # Simulate kernel restart
        nb_runner.reset_cash_state()

        # Second run: df is broken, but cell 3's disk-cached statement
        # should restore it. Must NOT raise NameError.
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert 'b' in output2
