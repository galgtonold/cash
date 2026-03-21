"""
Interaction test: string join with generator expressions.
Tests str.join with various iterables including generators,
conditional joins, and cross-cell string building.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringJoinGenerator:
    """Test str.join with generator expressions across cells."""

    def test_join_generators(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: join with generators
            "nums = list(range(1, 6))\ncsv_line = ','.join(str(n) for n in nums)\ndashed = '-'.join(f'{n:02d}' for n in nums)\nprint(f'csv={csv_line}')\nprint(f'dashed={dashed}')",
            # Cell 2: conditional join
            "words = ['hello', 'world', 'foo', 'bar', 'baz']\nlong_words = ' '.join(w.upper() for w in words if len(w) > 3)\nprint(f'long={long_words}')",
            # Cell 3: nested join
            "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\ntable = '\\n'.join(' | '.join(str(c) for c in row) for row in matrix)\nprint(f'table:\\n{table}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "csv=1,2,3,4,5" in out1
        assert "dashed=01-02-03-04-05" in out1
        out2 = nb_runner.get_output(2)
        assert "long=HELLO WORLD" in out2
        out3 = nb_runner.get_output(3)
        assert "1 | 2 | 3" in out3
        assert "7 | 8 | 9" in out3

    def test_join_edit(self, nb_runner):
        nb_runner.create_notebook([
            "items = ['apple', 'banana', 'cherry']\nresult = ', '.join(items)\nprint(f'result={result}')",
            "upper_result = result.upper()\nword_count = len(result.split(', '))\nprint(f'upper={upper_result}')\nprint(f'count={word_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=apple, banana, cherry" in nb_runner.get_output(1)
        assert "count=3" in nb_runner.get_output(2)

        # Add more items
        nb_runner.set_cell_source(1, "items = ['apple', 'banana', 'cherry', 'date', 'elderberry']\nresult = ', '.join(items)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "count=5" in nb_runner.get_output(2)

    def test_join_cache(self, nb_runner):
        nb_runner.create_notebook([
            "path_parts = ['usr', 'local', 'bin', 'python']\npath = '/'.join(path_parts)\nprint(f'path=/{path}')",
            "depth = path.count('/')\nprint(f'depth={depth}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "path=/usr/local/bin/python" in nb_runner.get_output(1)
        assert "depth=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "depth=3" in nb_runner.get_output(2)
