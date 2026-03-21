"""
Batch 105 â€” File dependency + cell edit interactions.

Tests the interaction between:
- Editing cells that read files
- Changing file contents between runs
- Editing both the file AND the cell code
- File changes + kernel restart + cell edits
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.files]


class TestFileChangeWithCellEdit:
    """Edit the code that reads a file, AND change the file."""

    def test_edit_cell_after_file_change(self, nb_runner, tmp_path):
        """Change file contents, then also edit the code reading it."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        csv_path.write_text("val\n10\n20\n30\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
            "result = df['val'].sum()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        # Change file AND edit cell to use mean instead
        csv_path.write_text("val\n100\n200\n300\n")
        import time; time.sleep(0.1)
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert "result = 200" in output

    def test_file_change_triggers_recomputation(self, nb_runner, tmp_path):
        """Change file between runs, cell code unchanged â†’ detects stale file."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        csv_path.write_text("x\n1\n2\n3\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
            "total = df['x'].sum()\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Change file only
        csv_path.write_text("x\n10\n20\n30\n")
        import time; time.sleep(0.1)
        nb_runner.run_cell(2)
        assert "total = 60" in nb_runner.get_output(2)

    def test_file_change_with_upstream_code_edit(self, nb_runner, tmp_path):
        """Upstream cell code changes AND file changes simultaneously."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        csv_path.write_text("a\n5\n10\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
            "multiplier = 2",
            "result = df['a'].sum() * multiplier\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Change BOTH file AND multiplier
        csv_path.write_text("a\n50\n100\n")
        import time; time.sleep(0.1)
        nb_runner.set_cell_source(2, "multiplier = 10")
        nb_runner.run_cell(3)
        assert "result = 1500" in nb_runner.get_output(3)


class TestFileChangeWithRestart:
    """File changes combined with kernel restarts."""

    def test_file_change_after_restart(self, nb_runner, tmp_path):
        """Change file after restart â€” should detect stale cache."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        csv_path.write_text("val\n1\n2\n3\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
            "s = df['val'].sum()\nprint(f's = {s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 6" in nb_runner.get_output(2)

        # Restart + change file
        nb_runner.shutdown()
        csv_path.write_text("val\n10\n20\n30\n")
        import time; time.sleep(0.1)
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 60" in nb_runner.get_output(2)

    def test_file_change_and_code_edit_after_restart(self, nb_runner, tmp_path):
        """File changes + code edit + kernel restart â€” triple stress."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        csv_path.write_text("x\n5\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
            "result = df['x'].iloc[0] * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.shutdown()
        csv_path.write_text("x\n100\n")
        import time; time.sleep(0.1)
        nb_runner.set_cell_source(2, "result = df['x'].iloc[0] * 3\nprint(f'result = {result}')")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(2)


class TestMultipleFileReads:
    """Multiple cells reading different files."""

    def test_two_files_change_one(self, nb_runner, tmp_path):
        """Two files read in different cells, only one changes."""
        f1 = tmp_path / "f1.csv"
        f2 = tmp_path / "f2.csv"
        f1_str = str(f1).replace('\\', '/')
        f2_str = str(f2).replace('\\', '/')
        f1.write_text("a\n10\n")
        f2.write_text("b\n20\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf1 = pd.read_csv('{f1_str}')\ndf2 = pd.read_csv('{f2_str}')",
            "result = df1['a'].iloc[0] + df2['b'].iloc[0]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Only change f1
        f1.write_text("a\n100\n")
        import time; time.sleep(0.1)
        nb_runner.run_cell(2)
        assert "result = 120" in nb_runner.get_output(2)

    def test_file_deleted_then_recreated(self, nb_runner, tmp_path):
        """Delete a file, then recreate it with new data."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace('\\', '/')
        csv_path.write_text("x\n5\n")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
            "val = df['x'].iloc[0]\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 5" in nb_runner.get_output(2)

        # Delete and recreate
        csv_path.unlink()
        csv_path.write_text("x\n999\n")
        import time; time.sleep(0.1)
        nb_runner.run_all()
        assert "val = 999" in nb_runner.get_output(2)
