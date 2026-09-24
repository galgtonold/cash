"""Data files and the cells that read them, changed together or one at a time."""

import textwrap
import time

import pytest


@pytest.mark.stress
@pytest.mark.files
class TestFileChangeWithCellEdit:
    """Edit the code that reads a file, AND change the file."""

    def test_edit_cell_after_file_change(self, nb_runner, tmp_path):
        """Change file contents, then also edit the code reading it."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "result = df['val'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        # Change file AND edit cell to use mean instead
        csv_path.write_text("val\n100\n200\n300\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.run_cell(2)
        output = nb_runner.get_output(2)
        assert "result = 200" in output

    def test_file_change_triggers_recomputation(self, nb_runner, tmp_path):
        """Change file between runs, cell code unchanged → detects stale file."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("x\n1\n2\n3\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "total = df['x'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Change file only
        csv_path.write_text("x\n10\n20\n30\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.run_cell(2)
        assert "total = 60" in nb_runner.get_output(2)

    def test_file_change_with_upstream_code_edit(self, nb_runner, tmp_path):
        """Upstream cell code changes AND file changes simultaneously."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("a\n5\n10\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "multiplier = 2",
                "result = df['a'].sum() * multiplier\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Change BOTH file AND multiplier
        csv_path.write_text("a\n50\n100\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.set_cell_source(2, "multiplier = 10")
        nb_runner.run_cell(3)
        assert "result = 1500" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.files
@pytest.mark.timeout(90)
class TestFilePlusCellEdits:
    """File changes combined with cell edits."""

    def test_edit_code_with_same_file(self, nb_runner, tmp_path):
        """Same file content, different code → recompute."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("hello", encoding="utf-8")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{fpath}') as f:\n    content = f.read()",
                "result = content.upper()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO" in nb_runner.get_output(2)

        # Edit code (file stays same)
        nb_runner.set_cell_source(2, "result = content.lower()\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = hello" in nb_runner.get_output(2)

    def test_edit_file_with_same_code(self, nb_runner, tmp_path):
        """Same code, different file content → recompute."""
        data_file = tmp_path / "numbers.txt"
        data_file.write_text("1,2,3", encoding="utf-8")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{fpath}') as f:\n    raw = f.read()",
                "nums = [int(x) for x in raw.split(',')]\ntotal = sum(nums)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Edit file
        data_file.write_text("10,20,30", encoding="utf-8")
        # Restart to ensure file dep is re-checked
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_edit_both_file_and_code(self, nb_runner, tmp_path):
        """Change both file content AND code → recompute."""
        data_file = tmp_path / "config.txt"
        data_file.write_text("scale=2", encoding="utf-8")
        fpath = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{fpath}') as f:\n    line = f.read()",
                "key, val = line.split('=')\nresult = int(val) * 10\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Edit file and code
        data_file.write_text("scale=5", encoding="utf-8")
        nb_runner.set_cell_source(
            2,
            "key, val = line.split('=')\nresult = int(val) * 100\nprint(f'result = {result}')",
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)


@pytest.mark.files
class TestFileDependencyWithUpstreamChanges:
    """Test file deps combined with code changes."""

    def test_file_dep_plus_code_change(self, nb_runner, tmp_path):
        """
        Read a CSV, transform it, modify the transformation code.
        Both file dep and code change should be detected.
        """
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace("\\", "/")
        pd.DataFrame({"val": [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['val'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "result = 60" in nb_runner.get_output(2)

        # Change the code to compute mean instead
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.run_cell(2)

        out = nb_runner.get_output(2)
        assert "result = 20.0" in out, f"Expected 20.0 after code change, got: {out}"


@pytest.mark.files
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestCSVFileWithEdits:
    """CSV file reads + cell code edits."""

    def test_read_csv_then_edit_processing(self, nb_runner, tmp_path):
        """Read CSV, edit the processing code."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['a'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(2)

        # Edit processing to use column b
        nb_runner.set_cell_source(2, "result = df['b'].sum()\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_edit_csv_then_rerun(self, nb_runner, tmp_path):
        """Read CSV, modify the CSV file, rerun."""
        csv_path = tmp_path / "vals.csv"
        csv_path.write_text("x\n10\n20\n30\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "total = df['x'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        # Modify the CSV file
        csv_path.write_text("x\n100\n200\n300\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 600" in nb_runner.get_output(2)

    def test_edit_csv_and_code_together(self, nb_runner, tmp_path):
        """Change both the file and the code."""
        csv_path = tmp_path / "items.csv"
        csv_path.write_text("val\n1\n2\n3\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "result = df['val'].sum()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(2)

        # Change both file and code
        csv_path.write_text("val\n10\n20\n30\n", encoding="utf-8")
        nb_runner.set_cell_source(2, "result = df['val'].mean()\nprint(f'result = {result}')")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20.0" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.files
@pytest.mark.timeout(90)
class TestCSVFileEdits:
    """CSV file changes with code edits."""

    def test_csv_data_change(self, nb_runner, tmp_path):
        """Change CSV data, verify cache invalidation."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        fpath = str(csv_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import csv",
                f"with open('{fpath}') as f:\n    reader = csv.DictReader(f)\n    rows = list(reader)",
                "total_a = sum(int(r['a']) for r in rows)\nprint(f'total_a = {total_a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_a = 4" in nb_runner.get_output(3)

        # Edit CSV
        csv_file.write_text("a,b\n10,20\n30,40\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_a = 40" in nb_runner.get_output(3)


@pytest.mark.files
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestJSONFileWithEdits:
    """JSON file reads + cell edits."""

    def test_read_json_edit_processing(self, nb_runner, tmp_path):
        """Read JSON file, edit the processing."""
        json_path = tmp_path / "config.json"
        json_path.write_text('{"scale": 2, "offset": 10}', encoding="utf-8")
        json_str = str(json_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import json\nwith open('{json_str}') as f:\n    config = json.load(f)",
                "result = config['scale'] * 5 + config['offset']\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Edit processing
        nb_runner.set_cell_source(2, "result = config['scale'] ** 3 + config['offset']\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(2)


@pytest.mark.files
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestTextFileWithEdits:
    """Plain text file reads + cell edits."""

    def test_read_text_file_edit_processing(self, nb_runner, tmp_path):
        """Read text file, edit the processing code."""
        txt_path = tmp_path / "data.txt"
        txt_path.write_text("hello\nworld\nfoo\nbar\n", encoding="utf-8")
        txt_str = str(txt_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{txt_str}') as f:\n    lines = f.read().strip().split('\\n')",
                "count = len(lines)\nprint(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(2)

        # Edit processing
        nb_runner.set_cell_source(2, "total_chars = sum(len(l) for l in lines)\nprint(f'chars = {total_chars}')")
        nb_runner.run_all()
        assert "chars = 16" in nb_runner.get_output(2)

    def test_edit_text_file_rerun(self, nb_runner, tmp_path):
        """Modify text file content, rerun."""
        txt_path = tmp_path / "notes.txt"
        txt_path.write_text("alpha\nbeta\n", encoding="utf-8")
        txt_str = str(txt_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{txt_str}') as f:\n    content = f.read().strip()",
                "word_count = len(content.split())\nprint(f'words = {word_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words = 2" in nb_runner.get_output(2)

        # Add more content
        txt_path.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words = 5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.files
class TestMultipleFileReads:
    """Multiple cells reading different files."""

    def test_two_files_change_one(self, nb_runner, tmp_path):
        """Two files read in different cells, only one changes."""
        f1 = tmp_path / "f1.csv"
        f2 = tmp_path / "f2.csv"
        f1_str = str(f1).replace("\\", "/")
        f2_str = str(f2).replace("\\", "/")
        f1.write_text("a\n10\n", encoding="utf-8")
        f2.write_text("b\n20\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf1 = pd.read_csv('{f1_str}')\ndf2 = pd.read_csv('{f2_str}')",
                "result = df1['a'].iloc[0] + df2['b'].iloc[0]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Only change f1
        f1.write_text("a\n100\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.run_cell(2)
        assert "result = 120" in nb_runner.get_output(2)

    def test_file_deleted_then_recreated(self, nb_runner, tmp_path):
        """Delete a file, then recreate it with new data."""
        csv_path = tmp_path / "data.csv"
        csv_path_str = str(csv_path).replace("\\", "/")
        csv_path.write_text("x\n5\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_path_str}')",
                "val = df['x'].iloc[0]\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 5" in nb_runner.get_output(2)

        # Delete and recreate
        csv_path.unlink()
        csv_path.write_text("x\n999\n", encoding="utf-8")
        import time

        time.sleep(0.1)
        nb_runner.run_all()
        assert "val = 999" in nb_runner.get_output(2)


@pytest.mark.files
class TestMultiFileDependencies:
    """Test caching with multiple file dependencies."""

    def test_two_csv_files_merged(self, nb_runner, tmp_path):
        """Read two CSV files and merge them - both should be tracked."""
        import pandas as pd

        csv1 = tmp_path / "users.csv"
        csv2 = tmp_path / "orders.csv"
        csv1_str = str(csv1).replace("\\", "/")
        csv2_str = str(csv2).replace("\\", "/")

        pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]}).to_csv(csv1, index=False)
        pd.DataFrame({"user_id": [1, 2], "amount": [100, 200]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\nusers = pd.read_csv('{csv1_str}')",
                f"orders = pd.read_csv('{csv2_str}')",
                "merged = users.merge(orders, on='user_id')\nprint(merged.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "Alice" in out, f"Expected Alice in output, got: {out}"
        assert "100" in out

        # Modify the second file
        pd.DataFrame({"user_id": [1, 2], "amount": [999, 888]}).to_csv(csv2, index=False)
        time.sleep(0.5)  # Ensure mtime changes

        # Re-run - should detect file change
        nb_runner.run_cells([2, 3])

        out2 = nb_runner.get_output(3)
        assert "999" in out2, f"Expected 999 after file change, got: {out2}"

    def test_json_file_dependency(self, nb_runner, tmp_path):
        """Test that JSON file reads are tracked as dependencies."""
        import json

        config_file = tmp_path / "config.json"
        config_str = str(config_file).replace("\\", "/")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"multiplier": 5, "offset": 10}, f)

        nb_runner.create_notebook(
            [
                f"""import json
with open('{config_str}') as f:
    config = json.load(f)""",
                "result = 100 * config['multiplier'] + config['offset']\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "result = 510" in out, f"Got: {out}"

        # Modify config
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"multiplier": 2, "offset": 0}, f)
        time.sleep(0.5)

        nb_runner.run_cells([1, 2])

        out2 = nb_runner.get_output(2)
        assert "result = 200" in out2, f"Expected 200 after config change, got: {out2}"

    def test_file_creation_in_one_cell_read_in_another(self, nb_runner, tmp_path):
        """Create a file in one cell, read in another - should track dependency."""
        fpath = tmp_path / "generated.txt"
        fpath_str = str(fpath).replace("\\", "/")

        nb_runner.create_notebook(
            [
                # Cell 1: Write a file
                f"""with open('{fpath_str}', 'w') as f:
    f.write('hello world')""",
                # Cell 2: Read it back
                f"""with open('{fpath_str}') as f:
    content = f.read()
print(f'content = {{content}}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "content = hello world" in out, f"Got: {out}"

    def test_csv_with_pathlib(self, nb_runner, tmp_path):
        """Test that pathlib-based file reads are tracked."""
        import pandas as pd

        data_file = tmp_path / "data.csv"
        data_str = str(data_file).replace("\\", "/")
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(data_file, index=False)

        nb_runner.create_notebook(
            [
                f"""import pandas as pd
from pathlib import Path
df = pd.read_csv(Path('{data_str}'))""",
                "total = df['x'].sum() + df['y'].sum()\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "total = 21" in out, f"Got: {out}"


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestFileTrackingEdgeCases:
    """Tests for file tracking with various file operations."""

    @pytest.mark.files
    def test_csv_create_read_modify_cycle(self, nb_runner, tmp_path):
        """Create CSV externally, read it in notebook, modify CSV, re-read should detect change."""
        import pandas as pd

        csv_path = str(tmp_path / "data.csv").replace("\\", "/")
        # Create initial CSV externally
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path}')",
                "total = df['a'].sum()",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=3" in output

        # Modify CSV file externally
        import time

        time.sleep(0.1)  # Ensure mtime changes
        pd.DataFrame({"a": [10, 20], "b": [30, 40]}).to_csv(csv_path, index=False)

        # Re-run: should detect file change and recompute
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "total=30" in output2

    @pytest.mark.files
    def test_json_file_dependency(self, nb_runner, tmp_path):
        """JSON file read should be tracked as dependency."""
        json_path = str(tmp_path / "config.json").replace("\\", "/")

        # Create JSON file
        import json

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"value": 42, "name": "test"}, f)

        nb_runner.create_notebook(
            [
                "import json",
                f"with open('{json_path}') as f:\n    config = json.load(f)",
                "result = config['value'] * 2",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=84" in output

    @pytest.mark.files
    def test_text_file_line_by_line(self, nb_runner, tmp_path):
        """Read text file line by line."""
        txt_path = str(tmp_path / "data.txt").replace("\\", "/")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("line1\nline2\nline3\n")

        nb_runner.create_notebook(
            [
                f"with open('{txt_path}') as f:\n    lines = f.readlines()",
                "count = len(lines)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "count=3" in output

    @pytest.mark.files
    def test_multiple_csv_files(self, nb_runner, tmp_path):
        """Reading multiple CSV files should track all dependencies."""
        import pandas as pd

        csv1 = str(tmp_path / "sales.csv").replace("\\", "/")
        csv2 = str(tmp_path / "costs.csv").replace("\\", "/")

        pd.DataFrame({"amount": [100, 200, 300]}).to_csv(csv1, index=False)
        pd.DataFrame({"cost": [50, 100, 150]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"sales = pd.read_csv('{csv1}')",
                f"costs = pd.read_csv('{csv2}')",
                "profit = sales['amount'].sum() - costs['cost'].sum()",
                "print(f'profit={profit}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "profit=300" in output

        # Modify only costs file
        pd.DataFrame({"cost": [10, 20, 30]}).to_csv(csv2, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "profit=540" in output2


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.files
class TestFileSystemOps:
    """File system read/write with caching."""

    def test_file_write_read_chain(self, nb_runner, tmp_path):
        """Write files, read back, process."""
        work = tmp_path / "rw_test"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                for i in range(5):
                    (base / f'log_{{i}}.txt').write_text(f'Entry {{i}}: value={{i*10}}')
            """),
                textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                contents = []
                for f in sorted(base.glob('log_*.txt')):
                    contents.append(f.read_text())
                total_files = len(contents)
            """),
                "print(f'files={total_files}')\nprint(f'first={contents[0]}')\nprint(f'last={contents[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "files=5" in out
        assert "Entry 0" in out
        assert "Entry 4" in out

    def test_temp_file_processing(self, nb_runner, tmp_path):
        """Process data through temp files."""
        work = tmp_path / "temp_proc"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                from pathlib import Path
                import json
                data = {{'items': [1, 2, 3, 4, 5], 'meta': 'test'}}
                path = Path('{work_str}') / 'data.json'
                path.write_text(json.dumps(data))
            """),
                textwrap.dedent(f"""\
                from pathlib import Path
                import json
                path = Path('{work_str}') / 'data.json'
                loaded = json.loads(path.read_text())
                total = sum(loaded['items'])
                meta = loaded['meta']
            """),
                "print(f'total={total} meta={meta}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=15" in out
        assert "meta=test" in out
