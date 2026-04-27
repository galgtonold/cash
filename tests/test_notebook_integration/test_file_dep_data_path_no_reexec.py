"""Integration test: data_path should NOT be re-executed when CSV changes.

Scenario: Cell 2 has `data_path = 'file.csv'` and `df = pd.read_csv(data_path)`.
When the CSV file changes, only df-producing statements should re-execute.
The pure `data_path` assignment has no file dependency and should be skipped.
"""
import pytest
import time

pytestmark = pytest.mark.files


class TestDataPathNotReexecuted:

    @pytest.mark.timeout(60)
    def test_data_path_not_reexecuted_on_file_change(self, nb_runner, tmp_path):
        """data_path assignment should NOT be auto-executed when CSV changes."""
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        nb_runner.create_notebook([
            "%cash_on\n%cash_debug on",
            f"import pandas as pd\ndata_path = '{str(csv_file).replace(chr(92), '/')}'\ndf = pd.read_csv(data_path)\nprint(f'loaded {{len(df)}} rows')",
            "result = df.sum()\nprint('sum_a=' + str(result['a']))",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert 'loaded 2 rows' in output2
        output3 = nb_runner.get_output(3)
        assert 'sum_a=4' in output3

        # Modify the CSV file
        time.sleep(0.1)
        csv_file.write_text("a,b\n10,20\n30,40\n50,60\n")

        # Run cell 3 again — should trigger upstream re-execution of
        # df = pd.read_csv(...) but NOT data_path assignment
        nb_runner.run_cell(3)
        raw_output = nb_runner.get_raw_output(3)

        # Check that data_path is NOT in the scheduled-for-execution list.
        # Debug output from simulation mentioning data_path is expected and ok.
        # The actual re-execution scheduling uses "[UPSTREAM] Scheduled for execution:"
        scheduled_lines = [
            line for line in raw_output.split('\n')
            if 'Scheduled for execution' in line
        ]
        data_path_scheduled = any('data_path' in line for line in scheduled_lines)
        assert not data_path_scheduled, (
            f"data_path was scheduled for re-execution!\n"
            f"Scheduled lines: {scheduled_lines}"
        )

        # Result should reflect new data
        output3_new = nb_runner.get_output(3)
        assert 'sum_a=90' in output3_new, (
            f"Expected sum_a=90 with new data. Got: {output3_new}"
        )

    @pytest.mark.timeout(60)
    def test_data_path_separate_cell_not_reexecuted(self, nb_runner, tmp_path):
        """data_path in a separate cell from read_csv should not be re-executed."""
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        nb_runner.create_notebook([
            "%cash_on\n%cash_debug on",
            "import pandas as pd",
            f"data_path = '{str(csv_file).replace(chr(92), '/')}'",
            f"df = pd.read_csv(data_path)\nprint(f'loaded {{len(df)}} rows')",
            "result = df.sum()\nprint('sum_a=' + str(result['a']))",
        ])
        nb_runner.start_kernel()

        # First run
        nb_runner.run_all()
        output4 = nb_runner.get_output(4)
        assert 'loaded 2 rows' in output4
        output5 = nb_runner.get_output(5)
        assert 'sum_a=4' in output5

        # Modify the CSV file
        time.sleep(0.1)
        csv_file.write_text("a,b\n10,20\n30,40\n50,60\n")

        # Run cell 5 again — upstream should re-execute df=pd.read_csv(data_path)
        # but NOT data_path assignment (it's a constant, no file dependency)
        nb_runner.run_cell(5)
        raw_output = nb_runner.get_raw_output(5)

        scheduled_lines = [
            line for line in raw_output.split('\n')
            if 'Scheduled for execution' in line
        ]
        data_path_scheduled = any('data_path' in line for line in scheduled_lines)
        assert not data_path_scheduled, (
            f"data_path was scheduled for re-execution!\n"
            f"Scheduled lines: {scheduled_lines}"
        )

        # Result should reflect new data
        output5_new = nb_runner.get_output(5)
        assert 'sum_a=90' in output5_new, (
            f"Expected sum_a=90 with new data. Got: {output5_new}"
        )

    @pytest.mark.timeout(60)
    def test_data_path_multi_cell_like_financial_demo(self, nb_runner, tmp_path):
        """Reproduce the financial_analysis_demo layout with exact cell content.

        Uses run_all() first, then modifies CSV (same columns), then re-runs compute cell.
        data_path should NOT be scheduled for re-execution.
        """
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("Ticker,Date,Close,Volume\nAAPL,2024-01-01,100,1000\nAAPL,2024-01-02,101,1100\n")

        csv_path = str(csv_file).replace('\\', '/')
        nb_runner.create_notebook([
            # Cell 1: imports
            "import pandas as pd\nimport numpy as np\nimport time\nimport os",
            # Cell 2: cash on
            "%cash_on\n%cash_debug on",
            # Cell 3: data_path + read_csv + Date conversion (like the real notebook)
            (f"print(os.getcwd())\n"
             f"# Ensure the data exists\n"
             f"data_path = '{csv_path}'\n"
             f"print('Loading data...')\n"
             f"# This read_csv call will be cached\n"
             f"df = pd.read_csv(data_path)\n"
             f"df['Date'] = pd.to_datetime(df['Date'])\n"
             f"print(df.head())"),
            # Cell 4: sort (self-assignment exactly like cell 4)
            ('print("Sorting data...")\n'
             't0 = time.time()\n'
             "df = df.sort_values(by=['Ticker', 'Date'])\n"
             "print(f'Sorted in {time.time() - t0:.2f}s')"),
            # Cell 5: compute cell
            "total = df['Close'].sum()\nprint('total=' + str(total))",
        ])
        nb_runner.start_kernel()

        # First run: execute all cells
        nb_runner.run_all()
        assert 'total=201' in nb_runner.get_output(5)

        # Modify the CSV file (same columns, different values)
        time.sleep(0.2)
        csv_file.write_text("Ticker,Date,Close,Volume\nAAPL,2024-01-01,200,1000\nAAPL,2024-01-02,201,1100\nGOOGL,2024-01-01,300,2000\n")

        # Re-execute cell 5 — should trigger upstream re-execution
        nb_runner.run_cell(5)
        raw_output = nb_runner.get_raw_output(5)
        print("=== RAW OUTPUT ===")
        print(raw_output)
        print("=== END ===")

        # Check scheduled statements: data_path should NOT be scheduled
        scheduled_lines = [
            line for line in raw_output.split('\n')
            if 'Scheduled for execution' in line
        ]
        data_path_scheduled = any('data_path' in line for line in scheduled_lines)
        assert not data_path_scheduled, (
            f"data_path was scheduled for re-execution!\n"
            f"Scheduled lines: {scheduled_lines}"
        )

        # Result should reflect new data
        assert 'total=701' in nb_runner.get_output(5), (
            f"Expected total=701 with new data. Got: {nb_runner.get_output(5)}"
        )
