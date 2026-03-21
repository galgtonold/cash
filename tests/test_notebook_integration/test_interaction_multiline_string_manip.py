"""Batch 413: multi-line string manipulation and triple quotes."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultilineStringManip:
    def test_triple_quote_strip(self, nb_runner):
        nb_runner.create_notebook([
            "text = '''  \n  hello  \n  world  \n  '''",
            "lines = [line.strip() for line in text.strip().split('\\n')]\nfiltered = [l for l in lines if l]\nprint(f'filtered={filtered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filtered=['hello', 'world']" in nb_runner.get_output(2)

    def test_multiline_join(self, nb_runner):
        nb_runner.create_notebook([
            "parts = ['line one', 'line two', 'line three']",
            "combined = '\\n'.join(parts)\ncount = combined.count('\\n')\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)

    def test_multiline_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = 'a,b,c\\n1,2,3\\n4,5,6'",
            "rows = data.strip().split('\\n')\nheader = rows[0].split(',')\nnum_rows = len(rows) - 1\nprint(f'header={header} num_rows={num_rows}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=['a', 'b', 'c']" in nb_runner.get_output(2)
        assert "num_rows=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = 'x,y\\n10,20\\n30,40\\n50,60'")
        nb_runner.run_all()
        assert "header=['x', 'y']" in nb_runner.get_output(2)
        assert "num_rows=3" in nb_runner.get_output(2)
