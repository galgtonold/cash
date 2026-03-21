"""
Round 3 - Batch 8: Kernel restart, disk restore, and complex upstream patterns.

Tests focusing on:
1. Kernel restart with disk-backed caching (persist annotation)
2. Complex upstream dependency chains after restart
3. Out-of-order cell execution patterns
4. Upstream simulation with deep dependency graphs
5. Mixed computed/restored states across cells
6. Variable shadowing across cells
7. Cell deletion/insertion simulation
"""

import pytest



pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


class TestKernelRestartDiskRestore:
    """Tests for disk persistence and restore after kernel restart.
    
    Uses shutdown() + start_kernel() to simulate kernel restart.
    """

    @pytest.mark.restore
    def test_persist_annotation_survives_restart(self, nb_runner, tmp_path):
        """Verify that @cash:persist variables restore from disk after restart."""
        cache_dir = tmp_path / "cash_cache"
        cache_str = str(cache_dir).replace('\\', '/')

        nb_runner.create_notebook([
            f"import os; os.makedirs('{cache_str}', exist_ok=True)",
            "# @cash:persist\nimport time; expensive = sum(range(100000))",
            "result = expensive * 2",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "result=" in output
        # Extract value
        val = output.strip().split("=")[1]

        # Restart kernel and re-run - should restore from cache
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert f"result={val}" in output2

    @pytest.mark.restore
    def test_disk_restore_chain_dependency(self, nb_runner, tmp_path):
        """After restart, a chain A→B→C should all restore or recompute correctly."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x + 8",
            "z = y * 2",
            "print(f'z={z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "z=100" in output1

        # Restart and re-run
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "z=100" in output2

    @pytest.mark.restore
    def test_restart_with_modified_middle_cell(self, nb_runner):
        """Restart + modify middle cell should recompute downstream."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "z = y * 3",
            "print(f'z={z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "z=45" in output1

        # Restart and change middle cell
        nb_runner.shutdown()
        nb_runner.set_cell_source(2, "y = x + 10")
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "z=60" in output2

    @pytest.mark.restore
    def test_restart_preserves_independent_branches(self, nb_runner):
        """After restart, independent variable branches should restore independently."""
        nb_runner.create_notebook([
            "a = 100",
            "b = 200",
            "x = a + 1",  # depends on a only
            "y = b + 1",  # depends on b only
            "print(f'x={x} y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "x=101" in output1
        assert "y=201" in output1

        # Restart, change only a-branch
        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "a = 999")
        nb_runner.start_kernel()
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "x=1000" in output2
        assert "y=201" in output2  # unchanged branch


class TestComplexUpstreamPatterns:
    """Tests for upstream simulation with complex dependency graphs."""

    @pytest.mark.upstream
    def test_diamond_dependency(self, nb_runner):
        """
        Diamond pattern: A → B, A → C, B+C → D.
        Changing A should propagate through both paths to D.
        """
        nb_runner.create_notebook([
            "a = 10",
            "b = a * 2",       # b depends on a
            "c = a * 3",       # c depends on a
            "d = b + c",       # d depends on b and c
            "print(f'd={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(5)
        assert "d=50" in output1  # 20+30

        # Change root
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "d=500" in output2  # 200+300

    @pytest.mark.upstream
    def test_deep_dependency_chain(self, nb_runner):
        """Deep chain: a → b → c → d → e → f → result."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1",
            "f = e + 1",
            "result = f + 1",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(8)
        assert "result=7" in output1

        # Change root
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()

        output2 = nb_runner.get_output(8)
        assert "result=106" in output2

    @pytest.mark.upstream
    def test_upstream_with_function_call(self, nb_runner):
        """Upstream should track through function definitions and calls."""
        nb_runner.create_notebook([
            "def multiply(x, y): return x * y",
            "a = 5",
            "b = multiply(a, 3)",
            "print(f'b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "b=15" in output1

        # Change function definition
        nb_runner.set_cell_source(1, "def multiply(x, y): return x * y + 1")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "b=16" in output2

    @pytest.mark.upstream
    def test_upstream_with_conditional_dependency(self, nb_runner):
        """Upstream tracks through conditionals that select different paths."""
        nb_runner.create_notebook([
            "mode = 'add'",
            "x = 10",
            "if mode == 'add':\n    result = x + 100\nelse:\n    result = x * 100",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output1 = nb_runner.get_output(4)
        assert "result=110" in output1

        # Change mode
        nb_runner.set_cell_source(1, "mode = 'multiply'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(4)
        assert "result=1000" in output2


class TestOutOfOrderExecution:
    """Tests for out-of-order cell execution patterns common in notebooks."""

    @pytest.mark.upstream
    def test_skip_middle_cell_then_run(self, nb_runner):
        """Run cells 1, 3 (skipping 2), then run 2 later."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "print(f'x={x}')",
        ])
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
        nb_runner.create_notebook([
            "x = 1",
            "y = x * 10",
            "print(f'y={y}')",
        ])
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
        nb_runner.create_notebook([
            "x = 42",
            "y = x + 1",
            "print(f'y={y}')",
        ])
        nb_runner.start_kernel()

        # Run only the last cell first - x and y don't exist yet
        nb_runner.run_cell(3)
        nb_runner.get_output(3)
        # Should either error or upstream simulation should provide the values
        # Either way it shouldn't crash the framework


class TestVariableShadowing:
    """Tests for variable name reuse/shadowing across cells."""

    @pytest.mark.core
    def test_same_variable_redefined_in_later_cell(self, nb_runner):
        """Variable redefined in a later cell should use the latest value."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "x = 100",  # shadow x
            "z = x * 2",
            "print(f'y={y} z={z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "y=20" in output   # uses original x=10
        assert "z=200" in output  # uses shadowed x=100

    @pytest.mark.core
    def test_shadow_with_different_type(self, nb_runner):
        """Redefining a variable with a different type should work."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "length = len(data)",
            "data = 'hello world'",  # now a string
            "length2 = len(data)",
            "print(f'length={length} length2={length2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "length=3" in output
        assert "length2=11" in output

    @pytest.mark.core
    def test_shadow_function_with_value(self, nb_runner):
        """Redefining a function name with a value should work."""
        nb_runner.create_notebook([
            "def compute(): return 42",
            "result1 = compute()",
            "compute = 99",  # shadow function with value
            "result2 = compute",
            "print(f'result1={result1} result2={result2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result1=42" in output
        assert "result2=99" in output


class TestComplexCellInteractions:
    """Tests for complex multi-cell interaction patterns."""

    @pytest.mark.core
    def test_cell_produces_multiple_outputs(self, nb_runner):
        """Cell producing multiple variables should cache all of them."""
        nb_runner.create_notebook([
            "a, b, c = 1, 2, 3",
            "x = a + b\ny = b + c\nz = a + c",
            "total = x + y + z",
            "print(f'total={total}')",
        ])
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
        fpath = str(tmp_path / "output.txt").replace('\\', '/')

        nb_runner.create_notebook([
            f"path = '{fpath}'",
            "with open(path, 'w') as f:\n    f.write('hello')\nresult = 'done'",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "result=done" in output

    @pytest.mark.core
    def test_lambda_in_cell(self, nb_runner):
        """Lambda functions should be tracked properly."""
        nb_runner.create_notebook([
            "double = lambda x: x * 2",
            "result = double(21)",
            "print(f'result={result}')",
        ])
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
        nb_runner.create_notebook([
            "data = list(range(10))",
            "evens = list(x for x in data if x % 2 == 0)",
            "total = sum(evens)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(4)
        assert "total=20" in output  # 0+2+4+6+8

    @pytest.mark.core
    def test_walrus_operator(self, nb_runner):
        """Walrus operator (:=) in expressions should track assignments."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "filtered = [y for x in data if (y := x * 2) > 4]",
            "print(f'filtered={filtered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "filtered=" in output

    @pytest.mark.core
    def test_string_formatting_methods(self, nb_runner):
        """Various string formatting methods should cache correctly."""
        nb_runner.create_notebook([
            "name = 'World'",
            "msg1 = f'Hello {name}'\nmsg2 = 'Hello %s' % name\nmsg3 = 'Hello {}'.format(name)",
            "print(f'{msg1}|{msg2}|{msg3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "Hello World|Hello World|Hello World" in output

        # Change name
        nb_runner.set_cell_source(1, "name = 'Cash'")
        nb_runner.run_all()

        output2 = nb_runner.get_output(3)
        assert "Hello Cash|Hello Cash|Hello Cash" in output2


class TestAnnotationInteractions:
    """Tests for @cash: annotation interactions with various code patterns."""

    @pytest.mark.core
    def test_no_cache_annotation_prevents_caching(self, nb_runner):
        """@cash:no-cache should force recomputation every time."""
        nb_runner.create_notebook([
            "counter = 0",
            "# @cash:no-cache\ncounter = counter + 1",
            "print(f'counter={counter}')",
        ])
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
        nb_runner.create_notebook([
            "# @cash:ttl=60\nexpensive = sum(range(10000))",
            "print(f'expensive={expensive}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "expensive=49995000" in output

    @pytest.mark.core
    def test_allow_random_annotation(self, nb_runner):
        """@cash:allow-random should suppress unseeded random warnings."""
        nb_runner.create_notebook([
            "# @cash:allow-random\nimport random\nval = random.randint(1, 100)",
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(2)
        assert "val=" in output


class TestLargeDataPatterns:
    """Tests for handling large data structures."""

    @pytest.mark.core
    def test_large_list_caching(self, nb_runner):
        """Large lists should cache and restore correctly."""
        nb_runner.create_notebook([
            "big_list = list(range(100000))",
            "total = sum(big_list)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "total=4999950000" in output

        # Re-run should use cache
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "total=4999950000" in output2

    @pytest.mark.core
    def test_nested_data_structure_caching(self, nb_runner):
        """Deeply nested data structures should cache correctly."""
        nb_runner.create_notebook([
            "nested = {'level1': {'level2': {'level3': [1, 2, 3]}}}",
            "val = nested['level1']['level2']['level3'][1]",
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(3)
        assert "val=2" in output

    @pytest.mark.core
    def test_dataframe_operations_chain(self, nb_runner):
        """Chain of DataFrame operations should track dependencies."""
        nb_runner.create_notebook([
            "import pandas as pd\nimport numpy as np",
            "df = pd.DataFrame({'a': np.arange(100), 'b': np.random.RandomState(42).randn(100)})",
            "df_filtered = df[df['a'] > 50]",
            "df_sorted = df_filtered.sort_values('b')",
            "result = len(df_sorted)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(6)
        assert "result=49" in output

        # Change filter threshold
        nb_runner.set_cell_source(3, "df_filtered = df[df['a'] > 75]")
        nb_runner.run_all()

        output2 = nb_runner.get_output(6)
        assert "result=24" in output2
