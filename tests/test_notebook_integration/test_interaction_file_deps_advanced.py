"""Batch 133 – File dependency + cell edit advanced interaction tests.

Tests that exercise file reads (CSV, JSON, text) combined with
cell edits, verifying that cache invalidation works correctly
when both code and files change.
"""

import pytest

pytestmark = [pytest.mark.files, pytest.mark.stress, pytest.mark.timeout(30)]


class TestCSVFileWithEdits:
    """CSV file reads + cell code edits."""

    def test_read_csv_then_edit_processing(self, nb_runner, tmp_path):
        """Read CSV, edit the processing code."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "result = df['a'].sum()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(2)

        # Edit processing to use column b
        nb_runner.set_cell_source(
            2, "result = df['b'].sum()\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_edit_csv_then_rerun(self, nb_runner, tmp_path):
        """Read CSV, modify the CSV file, rerun."""
        csv_path = tmp_path / "vals.csv"
        csv_path.write_text("x\n10\n20\n30\n")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "total = df['x'].sum()\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Modify the CSV file
        csv_path.write_text("x\n100\n200\n300\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 600" in nb_runner.get_output(2)

    def test_edit_csv_and_code_together(self, nb_runner, tmp_path):
        """Change both the file and the code."""
        csv_path = tmp_path / "items.csv"
        csv_path.write_text("val\n1\n2\n3\n")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook([
            f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
            "result = df['val'].sum()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        # Change both file and code
        csv_path.write_text("val\n10\n20\n30\n")
        nb_runner.set_cell_source(
            2, "result = df['val'].mean()\nprint(f'result = {result}')"
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20.0" in nb_runner.get_output(2)


class TestJSONFileWithEdits:
    """JSON file reads + cell edits."""

    def test_read_json_edit_processing(self, nb_runner, tmp_path):
        """Read JSON file, edit the processing."""
        json_path = tmp_path / "config.json"
        json_path.write_text('{"scale": 2, "offset": 10}')
        json_str = str(json_path).replace("\\", "/")

        nb_runner.create_notebook([
            f"import json\nwith open('{json_str}') as f:\n    config = json.load(f)",
            "result = config['scale'] * 5 + config['offset']\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Edit processing
        nb_runner.set_cell_source(
            2, "result = config['scale'] ** 3 + config['offset']\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(2)


class TestTextFileWithEdits:
    """Plain text file reads + cell edits."""

    def test_read_text_file_edit_processing(self, nb_runner, tmp_path):
        """Read text file, edit the processing code."""
        txt_path = tmp_path / "data.txt"
        txt_path.write_text("hello\nworld\nfoo\nbar\n")
        txt_str = str(txt_path).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{txt_str}') as f:\n    lines = f.read().strip().split('\\n')",
            "count = len(lines)\nprint(f'count = {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(2)

        # Edit processing
        nb_runner.set_cell_source(
            2, "total_chars = sum(len(l) for l in lines)\nprint(f'chars = {total_chars}')"
        )
        nb_runner.run_all()
        assert "chars = 16" in nb_runner.get_output(2)

    def test_edit_text_file_rerun(self, nb_runner, tmp_path):
        """Modify text file content, rerun."""
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("alpha\nbeta\n")
        txt_str = str(txt_path).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{txt_str}') as f:\n    content = f.read().strip()",
            "word_count = len(content.split())\nprint(f'words = {word_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words = 2" in nb_runner.get_output(2)

        # Add more content
        txt_path.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words = 5" in nb_runner.get_output(2)
