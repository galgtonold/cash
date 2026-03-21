"""Batch 179 – File dependency combined with cell edit tests.

Tests where both file content AND code change, verifying
that cache correctly invalidates on either trigger.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.files, pytest.mark.timeout(90)]


class TestFilePlusCellEdits:
    """File changes combined with cell edits."""

    def test_edit_code_with_same_file(self, nb_runner, tmp_path):
        """Same file content, different code → recompute."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("hello")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{fpath}') as f:\n    content = f.read()",
            "result = content.upper()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO" in nb_runner.get_output(2)

        # Edit code (file stays same)
        nb_runner.set_cell_source(
            2, "result = content.lower()\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = hello" in nb_runner.get_output(2)

    def test_edit_file_with_same_code(self, nb_runner, tmp_path):
        """Same code, different file content → recompute."""
        data_file = tmp_path / "numbers.txt"
        data_file.write_text("1,2,3")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{fpath}') as f:\n    raw = f.read()",
            "nums = [int(x) for x in raw.split(',')]\ntotal = sum(nums)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Edit file
        data_file.write_text("10,20,30")
        # Restart to ensure file dep is re-checked
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_edit_both_file_and_code(self, nb_runner, tmp_path):
        """Change both file content AND code → recompute."""
        data_file = tmp_path / "config.txt"
        data_file.write_text("scale=2")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{fpath}') as f:\n    line = f.read()",
            "key, val = line.split('=')\nresult = int(val) * 10\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Edit file and code
        data_file.write_text("scale=5")
        nb_runner.set_cell_source(
            2,
            "key, val = line.split('=')\nresult = int(val) * 100\nprint(f'result = {result}')",
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)


class TestCSVFileEdits:
    """CSV file changes with code edits."""

    def test_csv_data_change(self, nb_runner, tmp_path):
        """Change CSV data, verify cache invalidation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        fpath = str(csv_file).replace("\\", "/")

        nb_runner.create_notebook([
            "import csv",
            f"with open('{fpath}') as f:\n    reader = csv.DictReader(f)\n    rows = list(reader)",
            "total_a = sum(int(r['a']) for r in rows)\nprint(f'total_a = {total_a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_a = 4" in nb_runner.get_output(3)

        # Edit CSV
        csv_file.write_text("a,b\n10,20\n30,40\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_a = 40" in nb_runner.get_output(3)
