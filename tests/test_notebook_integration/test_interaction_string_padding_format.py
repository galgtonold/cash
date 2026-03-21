"""Batch 518: string ljust rjust center zfill formatting."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringPaddingFormatting:
    def test_ljust_rjust_center(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'hello'",
            "lj = text.ljust(10, '-')\nrj = text.rjust(10, '-')\nct = text.center(11, '*')\nprint(f'lj={lj} rj={rj} ct={ct}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "lj=hello-----" in out
        assert "rj=-----hello" in out
        assert "ct=***hello***" in out

    def test_zfill(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [1, 42, 100, 7]",
            "filled = [str(n).zfill(4) for n in nums]\nprint(f'filled={filled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "filled=['0001', '0042', '0100', '0007']" in nb_runner.get_output(2)

    def test_padding_edit(self, nb_runner):
        nb_runner.create_notebook([
            "word = 'hi'",
            "result = word.center(6, '=')\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result===hi==" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "word = 'test'")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "result==test=" in out2
