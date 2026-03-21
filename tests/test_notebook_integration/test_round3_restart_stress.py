"""
Batch 18: Kernel restart + disk restore stress tests.

These tests focus on the most fragile part of the caching system: restoring
cached results after a kernel restart. Tests verify that:
1. Cached values survive kernel restart via FileBackend
2. Changed code after restart correctly invalidates
3. Complex data types are properly serialized/deserialized
4. Multi-cell dependency chains restore correctly
5. File dependencies are re-checked after restart
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.restore]


class TestKernelRestartBasicRestore:
    """Test basic value restoration after kernel restart."""

    def test_scalar_restore_after_restart(self, nb_runner):
        """Scalar values should be restored from disk cache after restart."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2",
            "print(y)",
        ])
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
        nb_runner.create_notebook([
            "msg = 'hello world'",
            "upper = msg.upper()",
            "print(upper)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD" in nb_runner.get_output(3)

    def test_list_restore_after_restart(self, nb_runner):
        """List values restored after kernel restart."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                doubled = [x * 2 for x in data]
                print(doubled)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[2, 4, 6, 8, 10]" in nb_runner.get_output(2)

    def test_dict_restore_after_restart(self, nb_runner):
        """Dict values restored after kernel restart."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                config = {'host': 'localhost', 'port': 8080, 'debug': True}
            """),
            textwrap.dedent("""\
                url = f"{config['host']}:{config['port']}"
                print(url)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "localhost:8080" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "localhost:8080" in nb_runner.get_output(2)


class TestKernelRestartWithChanges:
    """Test code changes after kernel restart properly invalidate cache."""

    def test_change_after_restart_invalidates(self, nb_runner):
        """Changing code after restart should recompute."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "print(y)",
        ])
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
        nb_runner.create_notebook([
            "a = 5",
            "b = a * 2",
            "c = b + 1",
            "print(c)",
        ])
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
        nb_runner.create_notebook([
            "x = 10",
            "print(x)",
        ])
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


class TestKernelRestartComplexTypes:
    """Test complex type restoration after kernel restart."""

    def test_dataframe_restore(self, nb_runner):
        """DataFrame restored after kernel restart."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import pandas as pd
                df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            """),
            textwrap.dedent("""\
                total = df['a'].sum() + df['b'].sum()
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "21" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "21" in nb_runner.get_output(2)

    def test_nested_structure_restore(self, nb_runner):
        """Nested dict/list structure restored after restart."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "345" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "345" in nb_runner.get_output(2)

    def test_tuple_and_set_restore(self, nb_runner):
        """Tuple and set values restored after restart."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                t = (1, 2, 3, 4, 5)
                s = {10, 20, 30}
            """),
            textwrap.dedent("""\
                result = sum(t) + sum(s)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "75" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "75" in nb_runner.get_output(2)


class TestKernelRestartDependencyChains:
    """Test multi-cell dependency restoration after restart."""

    def test_three_cell_chain_restore(self, nb_runner):
        """Three-cell dependency chain restores correctly."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 20",
            "c = b * 3",
            "print(c)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "90" in nb_runner.get_output(4)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "90" in nb_runner.get_output(4)

    def test_function_dependency_restore(self, nb_runner):
        """Function defined in one cell, called in another, survives restart."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def double(x):
                    return x * 2
            """),
            textwrap.dedent("""\
                result = double(21)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(2)


class TestKernelRestartFileDeps:
    """Test file dependency handling across kernel restarts."""

    def test_file_dep_same_after_restart(self, nb_runner, tmp_path):
        """Unchanged file should allow cache restore after restart."""
        csv_path = tmp_path / "stable_data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{path_str}')",
            textwrap.dedent("""\
                total = df['x'].sum() + df['y'].sum()
                print(total)
            """),
        ])
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
        csv_path.write_text("x\n10\n20\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pandas as pd",
            f"df = pd.read_csv('{path_str}')",
            textwrap.dedent("""\
                total = df['x'].sum()
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(3)

        # Modify file and restart
        csv_path.write_text("x\n100\n200\n")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(3)

    def test_multiple_restarts(self, nb_runner):
        """Multiple restart cycles with unchanged code."""
        nb_runner.create_notebook([
            "x = 7",
            "y = x ** 2",
            "print(y)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "49" in nb_runner.get_output(3)

        for _ in range(2):
            nb_runner.shutdown()
            nb_runner.start_kernel()
            nb_runner.run_all()
            assert "49" in nb_runner.get_output(3)
