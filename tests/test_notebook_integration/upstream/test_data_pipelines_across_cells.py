"""Data pipelines and chains of type conversions spread over several cells, edited at one stage."""

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexDataPipelines:
    """Real-world data pipeline patterns."""

    @pytest.mark.core
    def test_groupby_aggregate(self, nb_runner, tmp_path):
        """GroupBy + aggregate pattern."""
        import pandas as pd

        csv = tmp_path / "orders.csv"
        csv_str = str(csv).replace("\\", "/")
        pd.DataFrame(
            {
                "category": ["A", "B", "A", "B", "A"],
                "value": [10, 20, 30, 40, 50],
            }
        ).to_csv(csv, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')",
                "summary = df.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()",
                "print(summary.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "A" in out and "B" in out

    @pytest.mark.core
    def test_merge_two_dataframes(self, nb_runner, tmp_path):
        """Merge two DataFrames from different cells."""
        import pandas as pd

        csv1 = tmp_path / "users.csv"
        csv2 = tmp_path / "orders.csv"
        csv1_str = str(csv1).replace("\\", "/")
        csv2_str = str(csv2).replace("\\", "/")
        pd.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}).to_csv(csv1, index=False)
        pd.DataFrame({"user_id": [1, 2, 1, 3], "amount": [100, 200, 150, 300]}).to_csv(csv2, index=False)

        nb_runner.create_notebook(
            [
                f"import pandas as pd\nusers = pd.read_csv('{csv1_str}')",
                f"orders = pd.read_csv('{csv2_str}')",
                "merged = pd.merge(users, orders, on='user_id')",
                "total_by_user = merged.groupby('name')['amount'].sum().reset_index()",
                "print(total_by_user.to_string(index=False))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "Alice" in out and "Bob" in out and "Charlie" in out


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestRealWorldSimulation:
    """Simulates a realistic data analysis workflow."""

    @pytest.mark.stress
    @pytest.mark.files
    def test_full_data_pipeline(self, nb_runner, tmp_path):
        """Complete data pipeline: load → clean → transform → analyze → report."""
        import numpy as np
        import pandas as pd

        # Create test data
        csv_path = str(tmp_path / "sales.csv").replace("\\", "/")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100, freq="D").astype(str),
                "product": np.random.choice(["A", "B", "C"], 100),
                "quantity": np.random.randint(1, 50, 100),
                "price": np.round(np.random.uniform(10, 100, 100), 2),
            }
        )
        df.to_csv(csv_path, index=False)

        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                f"raw = pd.read_csv('{csv_path}')",
                "raw['date'] = pd.to_datetime(raw['date'])\nraw['revenue'] = raw['quantity'] * raw['price']",
                "by_product = raw.groupby('product')['revenue'].sum().to_dict()",
                "total_revenue = sum(by_product.values())\ntop_product = max(by_product, key=by_product.get)",
                "print(f'total={total_revenue:.0f} top={top_product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "total=" in output
        assert "top=" in output

        # Re-run should cache
        nb_runner.run_all()
        output2 = nb_runner.get_output(6)
        assert output2.strip() == output.strip()

    @pytest.mark.stress
    def test_iterative_model_tuning(self, nb_runner):
        """Simulate iterative parameter tuning."""
        nb_runner.create_notebook(
            [
                "import numpy as np\nnp.random.seed(42)",
                "data = np.random.randn(1000)",
                "threshold = 1.0",
                "above = np.sum(data > threshold)\nbelow = np.sum(data < -threshold)\nwithin = len(data) - above - below",
                "print(f'above={above} below={below} within={within}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "above=" in output1

        # Tune threshold
        nb_runner.set_cell_source(3, "threshold = 2.0")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        # With higher threshold, fewer above/below
        above2 = int(output2.split("above=")[1].split()[0])
        above1 = int(output1.split("above=")[1].split()[0])
        assert above2 < above1  # Higher threshold → fewer above


@pytest.mark.stress
@pytest.mark.timeout(30)
class TestDataPipelineEdits:
    """Simulate a data processing pipeline with edits at various stages."""

    def test_etl_pipeline_edit_transform(self, nb_runner):
        """ETL: extract -> transform -> load. Edit transform step."""
        nb_runner.create_notebook(
            [
                "raw = [1, 2, 3, 4, 5]",
                "transformed = [x * 2 for x in raw]",
                "result = sum(transformed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit transform
        nb_runner.set_cell_source(2, "transformed = [x ** 2 for x in raw]")
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(3)

    def test_etl_pipeline_edit_source(self, nb_runner):
        """ETL: edit the source data."""
        nb_runner.create_notebook(
            [
                "raw = [10, 20, 30]",
                "filtered = [x for x in raw if x > 15]",
                "total = sum(filtered)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 50" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "raw = [5, 10, 15, 20, 25, 30]")
        nb_runner.run_all()
        assert "total = 75" in nb_runner.get_output(3)

    def test_multi_stage_pipeline_edit_multiple(self, nb_runner):
        """Multi-stage pipeline: source -> filter -> transform -> aggregate."""
        nb_runner.create_notebook(
            [
                "data = list(range(1, 11))",
                "filtered = [x for x in data if x % 2 == 0]",
                "transformed = [x * 10 for x in filtered]",
                "result = sum(transformed)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 300" in nb_runner.get_output(4)

        # Edit filter and transform
        nb_runner.set_cell_source(2, "filtered = [x for x in data if x % 2 != 0]")
        nb_runner.set_cell_source(3, "transformed = [x * 100 for x in filtered]")
        nb_runner.run_all()
        assert "result = 2500" in nb_runner.get_output(4)


@pytest.mark.core
class TestCrossDataTransformations:
    """Test complex data transformations across cells."""

    def test_pandas_modify_source_data(self, nb_runner):
        """Modify source data and verify the pipeline updates."""
        nb_runner.create_notebook(
            [
                """import pandas as pd
df = pd.DataFrame({'name': ['A', 'B'], 'val': [10, 20]})""",
                "total = df['val'].sum()\nprint(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Total: 30" in nb_runner.get_output(2)

        # Modify the DataFrame
        nb_runner.set_cell_source(
            1, "import pandas as pd\ndf = pd.DataFrame({'name': ['A', 'B', 'C'], 'val': [10, 20, 30]})"
        )
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Total: 60" in out, f"Expected Total: 60, got: {out}"


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestTypeConversionChains:
    """Tests for type conversion chains across cells."""

    @pytest.mark.core
    def test_str_to_int_to_float_chain(self, nb_runner):
        """String → int → float conversion chain."""
        nb_runner.create_notebook(
            [
                "raw = '42'",
                "as_int = int(raw)",
                "as_float = float(as_int) / 10",
                "print(f'result={as_float}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=4.2" in output

        # Change input
        nb_runner.set_cell_source(1, "raw = '100'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=10.0" in output2

    @pytest.mark.core
    def test_list_to_dict_to_dataframe(self, nb_runner):
        """List → dict → DataFrame conversion chain."""
        nb_runner.create_notebook(
            [
                "import pandas as pd",
                "names = ['Alice', 'Bob', 'Charlie']",
                "scores = [90, 85, 95]",
                "data_dict = dict(zip(names, scores))",
                "df = pd.DataFrame(list(data_dict.items()), columns=['name', 'score'])",
                "top = df.loc[df['score'].idxmax(), 'name']",
                "print(f'top={top}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(7)
        assert "top=Charlie" in output

        # Change scores
        nb_runner.set_cell_source(3, "scores = [95, 100, 85]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(7)
        assert "top=Bob" in output2

    @pytest.mark.core
    def test_json_roundtrip(self, nb_runner):
        """JSON serialization roundtrip should preserve data."""
        nb_runner.create_notebook(
            [
                "import json",
                "original = {'key': [1, 2, 3], 'nested': {'a': True, 'b': None}}",
                "serialized = json.dumps(original)",
                "restored = json.loads(serialized)",
                "match = original == restored",
                "print(f'match={match} type={type(restored).__name__}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "match=True" in output
        assert "type=dict" in output


@pytest.mark.core
class TestComplexAssignments:
    """Test complex assignment patterns."""

    def test_nested_tuple_unpacking(self, nb_runner):
        """Test nested tuple unpacking."""
        nb_runner.create_notebook(
            [
                "pairs = [(1, 'a'), (2, 'b'), (3, 'c')]",
                "firsts = [x for x, y in pairs]\nseconds = [y for x, y in pairs]",
                "print(f'firsts={firsts}, seconds={seconds}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(3)
        assert "firsts=[1, 2, 3]" in out, f"Got: {out}"
        assert "seconds=['a', 'b', 'c']" in out


@pytest.mark.core
class TestMultiStatementCells:
    """Test cells with many statements."""

    def test_mixed_assignments_and_expressions(self, nb_runner):
        """Test cells mixing assignments, print calls, and function definitions."""
        nb_runner.create_notebook(
            [
                """data = [1, 2, 3, 4, 5]
total = sum(data)
mean = total / len(data)
above = [x for x in data if x > mean]
below = [x for x in data if x <= mean]
print(f'mean={mean}, above={above}, below={below}')""",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out = nb_runner.get_output(1)
        assert "mean=3.0" in out, f"Got: {out}"
        assert "above=[4, 5]" in out
        assert "below=[1, 2, 3]" in out
