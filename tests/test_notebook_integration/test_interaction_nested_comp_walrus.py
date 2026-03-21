"""
Interaction test: nested comprehension with walrus operator.
Tests complex nested list/dict comprehensions using := for
intermediate results and cross-cell consumption.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNestedCompWalrus:
    """Test nested comprehension with walrus operator across cells."""

    def test_nested_comp_walrus(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: nested list comp with walrus
            "data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nresults = [(s := sum(row), len(row), round(s / len(row), 1)) for row in data]\nprint(f'results={results}')",
            # Cell 2: dict comp with walrus filter
            "words = ['hello', 'world', 'hi', 'python', 'ok', 'testing']\nlong_upper = {w: u for w in words if len(u := w.upper()) > 3}\nprint(f'filtered={long_upper}')",
            # Cell 3: combine
            "total_sum = sum(r[0] for r in results)\nlong_count = len(long_upper)\nprint(f'total={total_sum}')\nprint(f'long={long_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "results=[(6, 3, 2.0), (9, 2, 4.5), (30, 4, 7.5)]" in out1
        out2 = nb_runner.get_output(2)
        assert "'HELLO'" in out2
        assert "'WORLD'" in out2
        assert "'PYTHON'" in out2
        assert "'TESTING'" in out2
        out3 = nb_runner.get_output(3)
        assert "total=45" in out3
        assert "long=4" in out3

    def test_nested_comp_walrus_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nresults = [(s := sum(row), len(row)) for row in data]\nprint(f'count={len(results)}')",
            "avgs = [round(s / n, 1) for s, n in results]\nprint(f'avgs={avgs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avgs=[2.0, 4.5, 7.5]" in nb_runner.get_output(2)

        # Add more data
        nb_runner.set_cell_source(1, "data = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10, 10]]\nresults = [(s := sum(row), len(row)) for row in data]\nprint(f'count={len(results)}')")
        nb_runner.run_cells([1, 2])
        assert "count=4" in nb_runner.get_output(1)
        assert "avgs=[2.0, 4.5, 7.5, 10.0]" in nb_runner.get_output(2)

    def test_nested_comp_walrus_cache(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [10, 20, 30, 40, 50]\nfiltered = [sq for n in nums if (sq := n * n) > 500]\nprint(f'filtered={filtered}')",
            "total = sum(filtered)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filtered=[900, 1600, 2500]" in nb_runner.get_output(1)
        assert "total=5000" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=5000" in nb_runner.get_output(2)
