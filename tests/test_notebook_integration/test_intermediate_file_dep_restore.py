"""
Integration tests for intermediate statement file-dependency restore.

After a kernel restart, intermediate transformation statements (e.g.
df.sort_values()) that inherited file dependencies from upstream read_csv
should be restorable from disk cache.  Previously, executed_file_deps was
not propagated on upstream virtual restore, causing the cost model to
produce metadata-only entries for fast-but-large DataFrames -- permanently
preventing disk restore of those intermediate steps.
"""
import pytest
import time

pytestmark = [pytest.mark.files, pytest.mark.upstream]


def _simulate_kernel_restart(nb_runner):
    """Clear both cash state AND user namespace to simulate a real kernel restart."""
    nb_runner.reset_cash_state()
    # Also clear user variables to match a real restart
    clear_code = """
import sys as _sys
_keep = {'get_ipython', 'exit', 'quit', 'In', 'Out', '_', '__', '___',
         '_ih', '_oh', '_dh', '_i', '_ii', '_iii', '_i1', '_i2', '_i3',
         '__builtins__', '__builtin__', 'ip', '_sys'}
_to_del = [k for k in list(get_ipython().user_ns.keys()) if k not in _keep and not k.startswith('_')]
for _k in _to_del:
    try:
        del get_ipython().user_ns[_k]
    except KeyError:
        pass
del _to_del, _keep, _k, _sys
"""
    nb_runner._run_async(
        nb_runner.client.kc._async_execute_interactive(clear_code, store_history=False)
    )


class TestIntermediateFileDependencyRestore:
    """Verify intermediate transformations are properly cached and restorable."""

    def test_sort_values_restored_after_restart(self, nb_runner, tmp_path):
        """
        Pipeline: read_csv -> sort_values -> downstream sum.

        After kernel restart, running only the downstream cell should restore
        the sorted df from disk, not re-execute the sort.
        """
        csv = tmp_path / "data.csv"
        csv_path = str(csv).replace("\\", "/")
        csv.write_text("val\n30\n10\n20\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path}')",
            "df = df.sort_values('val')\nprint('sorted:', df['val'].tolist())",
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)

        # Simulate kernel restart
        _simulate_kernel_restart(nb_runner)

        # Run only downstream: upstream stmts should be restored, not executed
        nb_runner.run_cell(3)
        assert "total = 60" in nb_runner.get_output(3)

    def test_chained_transformations_restored(self, nb_runner, tmp_path):
        """
        Pipeline: read_csv -> datetime conversion -> sort -> downstream compute.

        This mirrors the real-world financial_demo scenario.  All intermediate
        transformations should produce full cache entries (not metadata-only)
        because they inherit file dependencies from read_csv.
        """
        csv = tmp_path / "data.csv"
        csv_path = str(csv).replace("\\", "/")
        csv.write_text("date,val\n2024-01-03,30\n2024-01-01,10\n2024-01-02,20\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path}')",
            "df['date'] = pd.to_datetime(df['date'])",
            "df = df.sort_values('date')\nprint('sorted:', df['val'].tolist())",
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(4)

        # Simulate kernel restart
        _simulate_kernel_restart(nb_runner)

        # Run only downstream cell
        nb_runner.run_cell(4)
        assert "total = 60" in nb_runner.get_output(4)

    def test_intermediate_stmts_invalidate_on_file_change(self, nb_runner, tmp_path):
        """
        After file content changes, even inherited-file-dep cache entries
        should be invalidated and recomputed.
        """
        csv = tmp_path / "data.csv"
        csv_path = str(csv).replace("\\", "/")
        csv.write_text("val\n10\n20\n30\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path}')",
            "df = df.sort_values('val')\ntotal = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Modify file
        time.sleep(0.2)
        csv.write_text("val\n100\n200\n300\n")

        # Re-run: cash should detect file change and recompute
        nb_runner.run_all()
        assert "total = 600" in nb_runner.get_output(2)

    def test_two_restarts_produce_correct_results(self, nb_runner, tmp_path):
        """
        After two restarts, downstream cells should still produce the same
        correct output -- verifying the executed_file_deps propagation works
        across multiple restart cycles.
        """
        csv = tmp_path / "data.csv"
        csv_path = str(csv).replace("\\", "/")
        csv.write_text("val\n5\n15\n25\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path}')",
            "df = df.sort_values('val')",
            "total = df['val'].sum()\nprint('total =', total)",
        ])
        nb_runner.start_kernel()

        # Initial run
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(3)

        # After first restart
        _simulate_kernel_restart(nb_runner)
        nb_runner.run_cell(3)
        assert "total = 45" in nb_runner.get_output(3)

        # After second restart -- should also work
        _simulate_kernel_restart(nb_runner)
        nb_runner.run_cell(3)
        assert "total = 45" in nb_runner.get_output(3)
