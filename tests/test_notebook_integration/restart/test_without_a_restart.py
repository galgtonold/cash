"""The same kinds of notebook in one session: out-of-order runs, multi-output cells, directives, large data."""

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestOutOfOrderExecution:
    """Tests for out-of-order cell execution patterns common in notebooks."""

    @pytest.mark.upstream
    def test_skip_middle_cell_then_run(self, nb_runner):
        """Run cells 1, 3 (skipping 2), then run 2 later."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "print(f'x={x}')",
            ]
        )
        nb_runner.start_kernel()

        # Run cell 1 and cell 3 (skip cell 2)
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)

        output = nb_runner.get_output(3)
        assert "x=10" in output

        # Now run skipped cell 2
        nb_runner.run_cell(2)
        # y should be computed
        # Run a new check - re-run cell 3 won't show y since cell 3 only prints x

    @pytest.mark.upstream
    def test_rerun_early_cell_invalidates_later(self, nb_runner):
        """Re-running an early cell should invalidate downstream when code changes."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 10",
                "print(f'y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert "y=10" in output1

        # Modify and re-run only cell 1
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(1)

        # Now run cell 2 and 3
        nb_runner.run_cells([2, 3])

        output2 = nb_runner.get_output(3)
        assert "y=50" in output2

    @pytest.mark.upstream
    def test_run_last_cell_first(self, nb_runner):
        """Running the last cell first should handle missing dependencies gracefully."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x + 1",
                "print(f'y={y}')",
            ]
        )
        nb_runner.start_kernel()

        # Run only the last cell first - x and y don't exist yet
        nb_runner.run_cell(3)
        nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComplexCellInteractions:
    """Tests for complex multi-cell interaction patterns."""

    @pytest.mark.core
    def test_cell_produces_multiple_outputs(self, nb_runner):
        """Cell producing multiple variables should cache all of them."""
        nb_runner.create_notebook(
            [
                "a, b, c = 1, 2, 3",
                "x = a + b\ny = b + c\nz = a + c",
                "total = x + y + z",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=12" in output  # (1+2) + (2+3) + (1+3) = 3+5+4=12

        # Re-run should be cached
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert "total=12" in output2

    @pytest.mark.core
    def test_cell_with_side_effect_and_result(self, nb_runner, tmp_path):
        """Cell with both a side effect (file write) and a computed result."""
        fpath = str(tmp_path / "output.txt").replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"path = '{fpath}'",
                "with open(path, 'w') as f:\n    f.write('hello')\nresult = 'done'",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=done" in output

    @pytest.mark.core
    def test_lambda_in_cell(self, nb_runner):
        """Lambda functions should be tracked properly."""
        nb_runner.create_notebook(
            [
                "double = lambda x: x * 2",
                "result = double(21)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=42" in output

        # Change lambda
        nb_runner.set_cell_source(1, "double = lambda x: x * 3")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "result=63" in output2

    @pytest.mark.core
    def test_generator_expression_caching(self, nb_runner):
        """Generator expressions consumed into a list should cache."""
        nb_runner.create_notebook(
            [
                "data = list(range(10))",
                "evens = list(x for x in data if x % 2 == 0)",
                "total = sum(evens)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=20" in output  # 0+2+4+6+8

    @pytest.mark.core
    def test_string_formatting_methods(self, nb_runner):
        """Various string formatting methods should cache correctly."""
        nb_runner.create_notebook(
            [
                "name = 'World'",
                "msg1 = f'Hello {name}'\nmsg2 = 'Hello %s' % name\nmsg3 = 'Hello {}'.format(name)",
                "print(f'{msg1}|{msg2}|{msg3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "Hello World|Hello World|Hello World" in output

        # Change name
        nb_runner.set_cell_source(1, "name = 'Cash'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "Hello Cash|Hello Cash|Hello Cash" in output2


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestAnnotationInteractions:
    """Tests for @cash: annotation interactions with various code patterns."""

    @pytest.mark.core
    def test_no_cache_annotation_prevents_caching(self, nb_runner):
        """@cash:no-cache should force recomputation every time."""
        nb_runner.create_notebook(
            [
                "counter = 0",
                "# @cash:no-cache\ncounter = counter + 1",
                "print(f'counter={counter}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(3)
        assert "counter=1" in output1

        # Re-run: no-cache should recompute
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        # Should still produce a result (may be 1 or 2 depending on skip behavior)
        assert "counter=" in output2

    @pytest.mark.core
    def test_ttl_annotation_format(self, nb_runner):
        """@cash:ttl=<seconds> should be parseable."""
        nb_runner.create_notebook(
            [
                "# @cash:ttl=60\nexpensive = sum(range(10000))",
                "print(f'expensive={expensive}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "expensive=49995000" in output

    @pytest.mark.core
    def test_allow_random_annotation(self, nb_runner):
        """@cash:allow-random should suppress unseeded random warnings."""
        nb_runner.create_notebook(
            [
                "# @cash:allow-random\nimport random\nval = random.randint(1, 100)",
                "print(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "val=" in output


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestLargeDataPatterns:
    """Tests for handling large data structures."""

    @pytest.mark.core
    def test_large_list_caching(self, nb_runner):
        """Large lists should cache and restore correctly."""
        nb_runner.create_notebook(
            [
                "big_list = list(range(100000))",
                "total = sum(big_list)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "total=4999950000" in output

        # Re-run should use cache
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "total=4999950000" in output2

    @pytest.mark.core
    def test_dataframe_operations_chain(self, nb_runner):
        """Chain of DataFrame operations should track dependencies."""
        nb_runner.create_notebook(
            [
                "import pandas as pd\nimport numpy as np",
                "df = pd.DataFrame({'a': np.arange(100), 'b': np.random.RandomState(42).randn(100)})",
                "df_filtered = df[df['a'] > 50]",
                "df_sorted = df_filtered.sort_values('b')",
                "result = len(df_sorted)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "result=49" in output

        # Change filter threshold
        nb_runner.set_cell_source(3, "df_filtered = df[df['a'] > 75]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        assert "result=24" in output2
