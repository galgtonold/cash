"""Data files read across cells, changed together with the code upstream of them."""

import textwrap
import time

import pytest


# Multi-file dependencies, disk restore after restart,
# complex module patterns, nested loops, dynamic imports, and advanced caching.
#
# These tests push the boundaries of the caching system with intricate patterns.
@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiFileDependencies:
    """Test scenarios where multiple files interact with the caching system."""

    @pytest.mark.files
    def test_two_csv_files_independent(self, nb_runner, tmp_path):
        """Two CSV files read in different cells — changing one shouldn't invalidate the other."""
        import pandas as pd

        csv1 = tmp_path / "sales.csv"
        csv2 = tmp_path / "products.csv"
        csv1_str = str(csv1).replace("\\", "/")
        csv2_str = str(csv2).replace("\\", "/")
        pd.DataFrame({"amount": [10, 20]}).to_csv(csv1, index=False)
        pd.DataFrame({"name": ["A", "B"]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf_sales = pd.read_csv('{csv1_str}')",
                f"df_products = pd.read_csv('{csv2_str}')",
                "total = df_sales['amount'].sum()",
                "names = df_products['name'].tolist()",
                "print(f'Total: {total}, Names: {names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(5)
        assert "Total: 30" in out1
        assert "Names: ['A', 'B']" in out1

        # Change only products CSV
        pd.DataFrame({"name": ["X", "Y", "Z"]}).to_csv(csv2, index=False)
        time.sleep(0.1)  # Ensure mtime changes
        nb_runner.run_all()
        out2 = nb_runner.get_output(5)
        assert "Total: 30" in out2  # Sales unchanged
        assert "Names: ['X', 'Y', 'Z']" in out2  # Products updated

    @pytest.mark.files
    def test_csv_read_then_write_new_csv(self, nb_runner, tmp_path):
        """Read CSV, transform, write to new CSV, then read the new CSV."""
        import pandas as pd

        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        input_str = str(input_csv).replace("\\", "/")
        output_str = str(output_csv).replace("\\", "/")
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(input_csv, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{input_str}')",
                f"df['x2'] = df['x'] * 2\ndf.to_csv('{output_str}', index=False)",
                f"df_out = pd.read_csv('{output_str}')",
                "print(df_out.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "2" in out and "4" in out and "6" in out

    @pytest.mark.files
    def test_json_file_dependency(self, nb_runner, tmp_path):
        """JSON file read and used in computation."""
        import json

        config_path = tmp_path / "config.json"
        config_str = str(config_path).replace("\\", "/")
        json.dump({"threshold": 50, "name": "test"}, config_path.open("w"))

        nb_runner.create_notebook(
            [
                f"import json\nwith open('{config_str}') as f:\n    config = json.load(f)",
                "result = config['threshold'] * 2",
                "print(f\"Name: {config['name']}, Result: {result}\")",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(3)
        assert "Name: test, Result: 100" in out1

        # Change the JSON
        json.dump({"threshold": 75, "name": "updated"}, config_path.open("w"))
        time.sleep(0.1)
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "Name: updated, Result: 150" in out2

    @pytest.mark.files
    def test_text_file_line_count(self, nb_runner, tmp_path):
        """Read a text file and count lines."""
        txt_path = tmp_path / "data.txt"
        txt_str = str(txt_path).replace("\\", "/")
        txt_path.write_text("line1\nline2\nline3\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"with open('{txt_str}') as f:\n    lines = f.readlines()",
                "count = len(lines)\nprint(f'Lines: {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Lines: 3" in out1

        # Add more lines
        txt_path.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        time.sleep(0.1)
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Lines: 5" in out2


# Combined complex patterns stressing multiple subsystems.
#
# Tests focusing on:
# 1. File deps + upstream simulation combined
# 2. Module reload + function tracking + from-import combined
# 3. Kernel restart + file modification + upstream invalidation
# 4. Multi-output cells + dependency chain + cache invalidation
# 5. Complex real-world simulation: data loading → processing → analysis → visualization data
# 6. Nested function definitions with closures across cells
# 7. Exception handling interleaved with caching
# 8. Type conversion chains (str→int→float→list→dict)
@pytest.mark.integration
@pytest.mark.timeout(30)
class TestFileDepsWithUpstream:
    """Combined file dependency and upstream simulation patterns."""

    @pytest.mark.files
    @pytest.mark.upstream
    def test_file_dep_change_propagates_through_chain(self, nb_runner, tmp_path):
        """File change should propagate through dependent cells."""
        import pandas as pd

        csv_path = str(tmp_path / "input.csv").replace("\\", "/")
        pd.DataFrame({"val": [1, 2, 3]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path}')",
                "total = df['val'].sum()",
                "doubled = total * 2",
                "print(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "doubled=12" in output

        # Modify file
        time.sleep(0.1)
        pd.DataFrame({"val": [10, 20, 30]}).to_csv(csv_path, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "doubled=120" in output2

    @pytest.mark.files
    @pytest.mark.upstream
    def test_multiple_file_deps_one_changes(self, nb_runner, tmp_path):
        """Multiple file deps, only one changes - verify partial invalidation."""
        import pandas as pd

        file_a = str(tmp_path / "a.csv").replace("\\", "/")
        file_b = str(tmp_path / "b.csv").replace("\\", "/")

        pd.DataFrame({"x": [1]}).to_csv(file_a, index=False)
        pd.DataFrame({"y": [100]}).to_csv(file_b, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"a = pd.read_csv('{file_a}')['x'].iloc[0]",
                f"b = pd.read_csv('{file_b}')['y'].iloc[0]",
                "result = a + b",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=101" in output

        # Change only file_a
        time.sleep(0.1)
        pd.DataFrame({"x": [50]}).to_csv(file_a, index=False)

        nb_runner.run_all()
        output2 = nb_runner.get_output(5)
        assert "result=150" in output2


# Complex combined interaction tests.
#
# Tests that combine multiple features: functions + files, imports + edits + restart,
# loops + mutations + edits, etc. These simulate real-world notebook workflows.
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestFunctionPlusFileEdit:
    """Function that reads file, both function and file edited."""

    def test_function_reads_file_then_file_changes(self, nb_runner, tmp_path):
        """Function reads a file; file content changes."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("100", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"def load_value():\n    with open('{path_str}') as f:\n        return int(f.read().strip())",
                "val = load_value()\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 100" in nb_runner.get_output(2)

        # Change file
        data_file.write_text("999", encoding="utf-8")
        nb_runner.run_all()
        assert "val = 999" in nb_runner.get_output(2)

    def test_function_and_file_both_change(self, nb_runner, tmp_path):
        """Both the function definition and the file change."""
        data_file = tmp_path / "vals.txt"
        data_file.write_text("10\n20\n30", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"def process():\n    with open('{path_str}') as f:\n        return sum(int(x) for x in f)",
                "result = process()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

        # Change file content and function
        data_file.write_text("1\n2\n3", encoding="utf-8")
        nb_runner.set_cell_source(
            1,
            f"def process():\n    with open('{path_str}') as f:\n        return max(int(x) for x in f)",
        )
        nb_runner.run_all()
        assert "result = 3" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestRestartWithFileDeps:
    """Kernel restart combined with file dependency changes."""

    @pytest.mark.restore
    @pytest.mark.files
    def test_restart_after_file_modification(self, nb_runner, tmp_path):
        """Restart kernel after external file mod should recompute."""
        import pandas as pd

        csv_path = str(tmp_path / "restart_test.csv").replace("\\", "/")
        pd.DataFrame({"val": [5, 10, 15]}).to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{csv_path}')",
                "result = df['val'].mean()",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=10.0" in output

        # Modify file and restart
        time.sleep(0.1)
        pd.DataFrame({"val": [100, 200, 300]}).to_csv(csv_path, index=False)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=200.0" in output2


@pytest.mark.files
class TestFileOperationPatterns:
    """Test various file operation patterns and their caching behavior."""

    def test_write_then_read_same_cell(self, nb_runner, tmp_path):
        """Write and read a file in the same cell."""
        nb_runner.create_notebook(
            [
                f"import json\ndata = {{'key': 'value'}}\nwith open(r'{(tmp_path / 'test.json').as_posix()}', 'w') as f:\n    json.dump(data, f)",
                f"import json\nwith open(r'{(tmp_path / 'test.json').as_posix()}') as f:\n    loaded = json.load(f)\nprint(loaded['key'])",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "value" in nb_runner.get_output(2)

    def test_csv_with_different_separators(self, nb_runner, tmp_path):
        """Read CSV with semicolons."""
        csv_path = tmp_path / "semi.csv"
        csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv(r'{csv_path.as_posix()}', sep=';')",
                "print(df.sum().to_string())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "5" in out, f"Expected sum of column a=5, got: {out}"


@pytest.mark.integration
@pytest.mark.stress
class TestClassWithFileDependency:
    """Test class definitions that interact with file operations."""

    def test_class_reads_config_file(self, nb_runner, tmp_path):
        """Class method reads from a config file."""
        config_file = tmp_path / "app_config.json"
        config_file.write_text('{"version": "1.0", "debug": false}', encoding="utf-8")
        path_str = str(config_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import json
                class AppConfig:
                    def __init__(self):
                        with open('{path_str}') as f:
                            self._data = json.load(f)
                    @property
                    def version(self):
                        return self._data['version']
            """),
                textwrap.dedent("""\
                cfg = AppConfig()
                print(cfg.version)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "1.0" in nb_runner.get_output(2)

    def test_data_processor_with_csv(self, nb_runner, tmp_path):
        """Data processing class that reads CSV files."""
        csv_path = tmp_path / "processor_data.csv"
        csv_path.write_text("metric,value\nCPU,75\nMEM,60\nDISK,45\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                textwrap.dedent("""\
                class MetricAnalyzer:
                    def __init__(self, path):
                        self.df = pd.read_csv(path)
                    def summary(self):
                        return self.df['value'].describe()
                    def max_metric(self):
                        idx = self.df['value'].idxmax()
                        return self.df.loc[idx, 'metric']
            """),
                textwrap.dedent(f"""\
                analyzer = MetricAnalyzer('{path_str}')
                print(analyzer.max_metric())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "CPU" in nb_runner.get_output(3)
