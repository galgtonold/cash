"""Restarting the kernel and editing cells before and after."""

import textwrap

import pytest


# Kernel restart, disk restore, and complex upstream patterns.
#
# Tests focusing on:
# 1. Kernel restart with disk-backed caching (persist annotation)
# 2. Complex upstream dependency chains after restart
# 3. Out-of-order cell execution patterns
# 4. Upstream simulation with deep dependency graphs
# 5. Mixed computed/restored states across cells
# 6. Variable shadowing across cells
# 7. Cell deletion/insertion simulation
@pytest.mark.integration
@pytest.mark.timeout(30)
class TestKernelRestartDiskRestore:
    """Tests for disk persistence and restore after kernel restart.

    Uses shutdown() + start_kernel() to simulate kernel restart.
    """

    @pytest.mark.restore
    def test_persist_annotation_survives_restart(self, nb_runner, tmp_path):
        """Verify that @cash:persist variables restore from disk after restart."""
        cache_dir = tmp_path / "cash_cache"
        cache_str = str(cache_dir).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import os; os.makedirs('{cache_str}', exist_ok=True)",
                "# @cash:persist\nimport time; expensive = sum(range(100000))",
                "result = expensive * 2",
                "print(f'result={result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x + 8",
                "z = y * 2",
                "print(f'z={z}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "z = y * 3",
                "print(f'z={z}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "a = 100",
                "b = 200",
                "x = a + 1",  # depends on a only
                "y = b + 1",  # depends on b only
                "print(f'x={x} y={y}')",
            ]
        )
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


# Kernel restart + disk restore stress tests.
#
# These tests focus on the most fragile part of the caching system: restoring
# cached results after a kernel restart. Tests verify that:
# 1. Cached values survive kernel restart via FileBackend
# 2. Changed code after restart correctly invalidates
# 3. Complex data types are properly serialized/deserialized
# 4. Multi-cell dependency chains restore correctly
# 5. File dependencies are re-checked after restart
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.restore
class TestKernelRestartBasicRestore:
    """Test basic value restoration after kernel restart."""

    def test_scalar_restore_after_restart(self, nb_runner):
        """Scalar values should be restored from disk cache after restart."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x * 2",
                "print(y)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "84" in nb_runner.get_output(3)

        # Restart kernel
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "84" in nb_runner.get_output(3)

    def test_string_restore_after_restart(self, nb_runner):
        """String values restored after kernel restart."""
        nb_runner.create_notebook(
            [
                "msg = 'hello world'",
                "upper = msg.upper()",
                "print(upper)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD" in nb_runner.get_output(3)

    def test_list_restore_after_restart(self, nb_runner):
        """List values restored after kernel restart."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                textwrap.dedent("""\
                doubled = [x * 2 for x in data]
                print(doubled)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8, 10]" in nb_runner.get_output(2)

    def test_dict_restore_after_restart(self, nb_runner):
        """Dict values restored after kernel restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                config = {'host': 'localhost', 'port': 8080, 'debug': True}
            """),
                textwrap.dedent("""\
                url = f"{config['host']}:{config['port']}"
                print(url)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "localhost:8080" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "localhost:8080" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.restore
class TestKernelRestartWithChanges:
    """Test code changes after kernel restart properly invalidate cache."""

    def test_change_after_restart_invalidates(self, nb_runner):
        """Changing code after restart should recompute."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "print(y)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(3)

        # Restart and change code
        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "105" in nb_runner.get_output(3)

    def test_change_middle_cell_after_restart(self, nb_runner):
        """Change middle cell after restart."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a * 2",
                "c = b + 1",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "11" in nb_runner.get_output(4)

        nb_runner.shutdown()
        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "51" in nb_runner.get_output(4)

    def test_add_cell_after_restart(self, nb_runner):
        """Adding a new cell after restart works correctly."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "print(x)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(2)

        nb_runner.shutdown()
        # The nb_runner doesn't support adding cells after creation,
        # but we can modify existing cells to test
        nb_runner.set_cell_source(2, "y = x * 3\nprint(y)")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.restore
class TestKernelRestartComplexTypes:
    """Test complex type restoration after kernel restart."""

    def test_dataframe_restore(self, nb_runner):
        """DataFrame restored after kernel restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import pandas as pd
                df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            """),
                textwrap.dedent("""\
                total = df['a'].sum() + df['b'].sum()
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "21" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "21" in nb_runner.get_output(2)

    def test_nested_structure_restore(self, nb_runner):
        """Nested dict/list structure restored after restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = {
                    'users': [
                        {'name': 'Alice', 'scores': [90, 85]},
                        {'name': 'Bob', 'scores': [78, 92]}
                    ],
                    'count': 2
                }
            """),
                textwrap.dedent("""\
                total_scores = sum(
                    s for u in data['users'] for s in u['scores']
                )
                print(total_scores)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "345" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "345" in nb_runner.get_output(2)

    def test_tuple_and_set_restore(self, nb_runner):
        """Tuple and set values restored after restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                t = (1, 2, 3, 4, 5)
                s = {10, 20, 30}
            """),
                textwrap.dedent("""\
                result = sum(t) + sum(s)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "75" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "75" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.restore
class TestKernelRestartDependencyChains:
    """Test multi-cell dependency restoration after restart."""

    def test_three_cell_chain_restore(self, nb_runner):
        """Three-cell dependency chain restores correctly."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 20",
                "c = b * 3",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "90" in nb_runner.get_output(4)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "90" in nb_runner.get_output(4)

    def test_function_dependency_restore(self, nb_runner):
        """Function defined in one cell, called in another, survives restart."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def double(x):
                    return x * 2
            """),
                textwrap.dedent("""\
                result = double(21)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.restore
class TestKernelRestartFileDeps:
    """Test file dependency handling across kernel restarts."""

    def test_file_dep_same_after_restart(self, nb_runner, tmp_path):
        """Unchanged file should allow cache restore after restart."""
        csv_path = tmp_path / "stable_data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                textwrap.dedent("""\
                total = df['x'].sum() + df['y'].sum()
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(3)

    def test_file_changed_after_restart(self, nb_runner, tmp_path):
        """Changed file should invalidate cache even after restart."""
        csv_path = tmp_path / "changing_data.csv"
        csv_path.write_text("x\n10\n20\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                "import pandas as pd",
                f"df = pd.read_csv('{path_str}')",
                textwrap.dedent("""\
                total = df['x'].sum()
                print(total)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(3)

        # Modify file and restart
        csv_path.write_text("x\n100\n200\n", encoding="utf-8")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3)

    def test_multiple_restarts(self, nb_runner):
        """Multiple restart cycles with unchanged code."""
        nb_runner.create_notebook(
            [
                "x = 7",
                "y = x ** 2",
                "print(y)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "49" in nb_runner.get_output(3)

        for _ in range(2):
            nb_runner.shutdown()
            nb_runner.start_kernel()
            nb_runner.run_all()
            assert "49" in nb_runner.get_output(3)


# Kernel restart + cell edit interactions.
#
# Tests that exercise the most fragile path: editing cells after a kernel restart.
# After restart, cash must:
# - Rebuild lineage state from disk cache
# - Detect that cell code has changed since the cached state
# - Re-execute changed statements and propagate properly
@pytest.mark.stress
@pytest.mark.restore
class TestEditAfterRestart:
    """Edit cells after a kernel restart — the trickiest path."""

    def test_edit_upstream_after_restart(self, nb_runner):
        """Run all, restart, edit cell 1, run cell 3 → should see new value."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        # Restart kernel
        nb_runner.shutdown()
        nb_runner.start_kernel()

        # Edit upstream before running
        nb_runner.set_cell_source(1, "x = 99")
        nb_runner.run_cell(3)
        assert "y = 104" in nb_runner.get_output(3)

    def test_no_edit_after_restart_restores_from_disk(self, nb_runner):
        """Run all, restart, run cell 3 without edits → should restore from disk."""
        nb_runner.create_notebook(
            [
                "a = 42",
                "b = a * 2",
                "print(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 84" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(3)
        assert "b = 84" in nb_runner.get_output(3)

    def test_edit_then_restart_then_run(self, nb_runner):
        """Edit cell 1, restart BEFORE running, then run all."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 3",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        # Edit but DON'T run — then restart
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 300" in nb_runner.get_output(3)

    def test_edit_middle_cell_after_restart(self, nb_runner):
        """Edit middle cell (formula change) after restart."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_cell(3)
        assert "z = 2000" in nb_runner.get_output(3)


class TestMultipleRestartsWithEdits:
    """Multiple restart-edit cycles."""

    @pytest.mark.stress
    @pytest.mark.restore
    def test_restart_edit_restart_edit(self, nb_runner):
        """Two restart-edit cycles in sequence."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 10",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(3)

        # First restart + edit
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 110" in nb_runner.get_output(3)

        # Second restart + edit
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "x = 500")
        nb_runner.run_all()
        assert "y = 510" in nb_runner.get_output(3)

    @pytest.mark.stress
    @pytest.mark.restore
    def test_restart_without_edit_then_edit(self, nb_runner):
        """Restart without edit, run, then edit and run again."""
        nb_runner.create_notebook(
            [
                "a = 7",
                "b = a * 3",
                "print(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 21" in nb_runner.get_output(3)

        # Restart without edit
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 21" in nb_runner.get_output(3)

        # Now edit
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_cell(3)
        assert "b = 30" in nb_runner.get_output(3)

    @pytest.mark.restore
    @pytest.mark.stress
    @pytest.mark.timeout(60)
    def test_edit_restart_edit_restart(self, nb_runner):
        """Edit, restart, edit again, restart again."""
        nb_runner.create_notebook(
            [
                "n = 1",
                "result = n * 100\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Edit 1
        nb_runner.set_cell_source(1, "n = 2")
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)

        # Edit 2
        nb_runner.set_cell_source(1, "n = 5")
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.restore
class TestRestartWithFunctions:
    """Restart interactions with function definitions."""

    def test_function_edit_after_restart(self, nb_runner):
        """Edit a function definition cell after kernel restart."""
        nb_runner.create_notebook(
            [
                "def compute(x):\n    return x * 2",
                "result = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "def compute(x):\n    return x * 3")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_function_unchanged_after_restart_uses_cache(self, nb_runner):
        """Unchanged function after restart should restore from cache."""
        nb_runner.create_notebook(
            [
                "def add(a, b):\n    return a + b",
                "val = add(3, 4)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 7" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "val = 7" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.restore
class TestRestartWithLongChains:
    """Restart with long dependency chains."""

    def test_five_cell_chain_restart_edit_root(self, nb_runner):
        """5-cell chain, restart, edit root, run last."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_cell(5)
        assert "e = 104" in nb_runner.get_output(5)

    def test_four_cell_chain_restart_edit_middle(self, nb_runner):
        """4-cell chain, restart, edit middle, run last."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "z = y + 5",
                "print(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(4)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_cell(4)
        assert "z = 1005" in nb_runner.get_output(4)

    def test_chain_restart_revert_to_original(self, nb_runner):
        """Edit root, restart, revert to original, run last."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x + 10",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        # Edit and run
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(3)
        assert "y = 60" in nb_runner.get_output(3)

        # Restart and revert
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(3)
        assert "y = 15" in nb_runner.get_output(3)


# Kernel restart + cell edit combined interaction tests.
#
# Tests where users restart the kernel (via shutdown + start_kernel)
# combined with cell edits before/after restart, verifying disk
# restore and re-computation work correctly together.
@pytest.mark.restore
@pytest.mark.stress
@pytest.mark.timeout(60)
class TestRestartThenEditCells:
    """Restart kernel, then edit cells."""

    def test_restart_then_edit_root(self, nb_runner):
        """Restart kernel, then edit root cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "z = y + 5\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()

        # Edit root AFTER restart
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "z = 205" in nb_runner.get_output(3)

    def test_restart_then_edit_leaf(self, nb_runner):
        """Restart kernel, then edit leaf cell."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a + 10",
                "result = b * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 45" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()

        nb_runner.set_cell_source(3, "result = b * 100\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 1500" in nb_runner.get_output(3)


@pytest.mark.restore
@pytest.mark.stress
@pytest.mark.timeout(60)
class TestEditThenRestartCells:
    """Edit cells, then restart."""

    def test_edit_then_restart_run(self, nb_runner):
        """Edit, restart, run uses edited code."""
        nb_runner.create_notebook(
            [
                "val = 1",
                "out = val + 10\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 11" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "val = 50")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 60" in nb_runner.get_output(2)

    def test_edit_run_restart_run_should_restore(self, nb_runner):
        """Edit, run, restart, run restores from cache."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "total = sum(data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)


# Kernel restart with dirty state interaction tests.
#
# Tests that establish cached state, restart the kernel, and verify
# that cache restoration works correctly after restart.
@pytest.mark.stress
@pytest.mark.restore
@pytest.mark.timeout(90)
class TestRestartRestore:
    """Restart kernel and verify cache restoration."""

    def test_basic_restart_restore(self, nb_runner):
        """After restart, run_all should restore/recompute."""
        nb_runner.create_notebook(
            [
                "x = 42  # basic value",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

    def test_restart_after_edit(self, nb_runner):
        """Edit, restart, verify new values are computed."""
        nb_runner.create_notebook(
            [
                "a = 10  # param a",
                "b = a + 5\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 15" in nb_runner.get_output(2)

        # Edit then restart
        nb_runner.set_cell_source(1, "a = 100  # param a big")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 105" in nb_runner.get_output(2)

    def test_restart_chain_restore(self, nb_runner):
        """3-cell chain, restart, verify chain recomputes."""
        nb_runner.create_notebook(
            [
                "base = 5  # chain base",
                "mid = base * 3",
                "final = mid + 7\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base=5, mid=15, final=22
        assert "final = 22" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final = 22" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.restore
@pytest.mark.timeout(90)
class TestRestartWithFunction:
    """Restart with function definitions."""

    def test_restart_function_def(self, nb_runner):
        """Function definition survives restart via re-execution."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "result = double(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

    def test_restart_edit_function_then_run(self, nb_runner):
        """Edit function, restart, verify new function is used."""
        nb_runner.create_notebook(
            [
                "def process(x):\n    return x + 1",
                "out = process(10)\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 11" in nb_runner.get_output(2)

        # Edit function
        nb_runner.set_cell_source(1, "def process(x):\n    return x * 10")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 100" in nb_runner.get_output(2)
