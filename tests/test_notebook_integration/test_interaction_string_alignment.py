"""Batch 395: string ljust/rjust/center and formatting alignment."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringAlignment:
    def test_ljust_rjust(self, nb_runner):
        nb_runner.create_notebook([
            "items = [('apple', 3), ('banana', 12), ('cherry', 7)]",
            "lines = []\nfor name, qty in items:\n    lines.append(f'{name.ljust(10)}{str(qty).rjust(5)}')\nresult = '|'.join(lines)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple" in out
        assert "banana" in out

    def test_center_edit(self, nb_runner):
        nb_runner.create_notebook([
            "title = 'Hello'",
            "centered = title.center(20, '-')\nprint(f'centered={centered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "-------Hello--------" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "title = 'Hi'")
        nb_runner.run_all()
        assert "---------Hi---------" in nb_runner.get_output(2)

    def test_format_spec(self, nb_runner):
        nb_runner.create_notebook([
            "values = [3.14159, 2.71828, 1.41421]",
            "formatted = [f'{v:.2f}' for v in values]\nprint(f'formatted={formatted}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted=['3.14', '2.72', '1.41']" in nb_runner.get_output(2)
