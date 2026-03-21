"""Batch 162 – Global variable and side-effect interaction tests.

Tests editing cells that mutate global state, counters, accumulators,
and side-effects to verify cache consistency.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.mutations, pytest.mark.timeout(90)]


class TestGlobalCounterEdits:
    """Editing cells that use global counters."""

    def test_counter_increment_edit(self, nb_runner):
        """Edit a counter's increment value."""
        nb_runner.create_notebook([
            "counter = 0  # init counter",
            "counter = counter + 1\nprint(f'counter = {counter}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "counter = 1" in nb_runner.get_output(2)

        # Edit to increment by 5
        nb_runner.set_cell_source(2, "counter = counter + 5\nprint(f'counter = {counter}')")
        nb_runner.run_all()
        assert "counter = 5" in nb_runner.get_output(2)

    def test_counter_init_edit(self, nb_runner):
        """Edit the initial counter value."""
        nb_runner.create_notebook([
            "total = 10  # starting value",
            "total = total * 2\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 20" in nb_runner.get_output(2)

        # Change starting value
        nb_runner.set_cell_source(1, "total = 100  # starting value changed")
        nb_runner.run_all()
        assert "total = 200" in nb_runner.get_output(2)


class TestAccumulatorEdits:
    """Editing accumulator patterns."""

    def test_list_accumulator_edit_append(self, nb_runner):
        """Edit what gets appended to a list."""
        nb_runner.create_notebook([
            "items = []  # fresh list",
            "for i in range(3):\n    items.append(i)\nprint(f'items = {items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [0, 1, 2]" in nb_runner.get_output(2)

        # Change to squared values
        nb_runner.set_cell_source(
            2,
            "for i in range(3):\n    items.append(i ** 2)\nprint(f'items = {items}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "0" in out and "1" in out and "4" in out

    def test_dict_accumulator_edit(self, nb_runner):
        """Edit dictionary accumulation logic."""
        nb_runner.create_notebook([
            "data = {}  # fresh dict",
            "for k in ['a', 'b', 'c']:\n    data[k] = len(k)\nprint(f'data = {data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        # Change to uppercase keys
        nb_runner.set_cell_source(
            2,
            "for k in ['a', 'b', 'c']:\n    data[k.upper()] = len(k) * 10\nprint(f'data = {data}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': 10" in out


class TestSideEffectEdits:
    """Editing cells with file I/O side effects."""

    def test_file_write_edit_content(self, nb_runner, tmp_path):
        """Edit what gets written to a file."""
        out_path = str(tmp_path / "output.txt").replace("\\", "/")
        nb_runner.create_notebook([
            f"path = '{out_path}'",
            "with open(path, 'w') as f:\n    f.write('hello')",
            "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "content = hello" in nb_runner.get_output(3)

        # Edit what we write
        nb_runner.set_cell_source(
            2, "with open(path, 'w') as f:\n    f.write('world')"
        )
        nb_runner.run_all()
        assert "content = world" in nb_runner.get_output(3)

    def test_file_write_edit_path(self, nb_runner, tmp_path):
        """Edit the output file path."""
        path1 = str(tmp_path / "out1.txt").replace("\\", "/")
        path2 = str(tmp_path / "out2.txt").replace("\\", "/")
        nb_runner.create_notebook([
            f"path = '{path1}'",
            "with open(path, 'w') as f:\n    f.write('data1')",
            "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "content = data1" in nb_runner.get_output(3)

        # Switch path
        nb_runner.set_cell_source(1, f"path = '{path2}'")
        nb_runner.set_cell_source(2, "with open(path, 'w') as f:\n    f.write('data2')")
        nb_runner.set_cell_source(
            3,
            "with open(path) as f:\n    content = f.read()\nprint(f'content = {content}')",
        )
        nb_runner.run_all()
        assert "content = data2" in nb_runner.get_output(3)
