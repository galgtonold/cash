"""Values, chains and file dependencies restored from disk after a restart with no edit."""

import textwrap

import pytest


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
