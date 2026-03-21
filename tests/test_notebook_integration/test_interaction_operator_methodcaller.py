"""
Interaction test: operator methodcaller for dynamic method dispatch.
Tests operator.methodcaller for calling methods by name,
with arguments, and cross-cell dynamic dispatch patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOperatorMethodcaller:
    """Test operator.methodcaller across cells."""

    def test_methodcaller_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: methodcaller for string methods
            "from operator import methodcaller\nwords = ['hello', 'WORLD', 'Python']\nupper = list(map(methodcaller('upper'), words))\nlower = list(map(methodcaller('lower'), words))\nprint(f'upper={upper}')\nprint(f'lower={lower}')",
            # Cell 2: methodcaller with args
            "texts = ['hello world', 'foo bar baz', 'one two']\nsplit2 = list(map(methodcaller('split', ' ', 1), texts))\nprint(f'split2={split2}')",
            # Cell 3: sorting with methodcaller
            "items = ['banana', 'apple', 'cherry']\nsorted_by_len = sorted(items, key=methodcaller('__len__'))\nprint(f'by_len={sorted_by_len}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "upper=['HELLO', 'WORLD', 'PYTHON']" in out1
        assert "lower=['hello', 'world', 'python']" in out1
        out2 = nb_runner.get_output(2)
        assert "['hello', 'world']" in out2
        out3 = nb_runner.get_output(3)
        assert "by_len=['apple', 'banana', 'cherry']" in out3

    def test_methodcaller_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import methodcaller\ndata = ['  hello  ', '  world  ']\nstripped = list(map(methodcaller('strip'), data))\nprint(f'stripped={stripped}')",
            "joined = '-'.join(stripped)\nprint(f'joined={joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "joined=hello-world" in nb_runner.get_output(2)

        # Edit to title instead of strip
        nb_runner.set_cell_source(1, "from operator import methodcaller\ndata = ['hello there', 'world here']\nstripped = list(map(methodcaller('title'), data))\nprint(f'stripped={stripped}')")
        nb_runner.run_cells([1, 2])
        assert "joined=Hello There-World Here" in nb_runner.get_output(2)

    def test_methodcaller_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import methodcaller\nnums_str = ['1', '2', '3']\npadded = list(map(methodcaller('zfill', 3), nums_str))\nprint(f'padded={padded}')",
            "joined = ','.join(padded)\nprint(f'joined={joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "padded=['001', '002', '003']" in nb_runner.get_output(1)
        assert "joined=001,002,003" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "joined=001,002,003" in nb_runner.get_output(2)
