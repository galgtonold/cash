"""Batch 435: string zfill and numeric formatting."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringZfillNumFormat:

    def test_format_thousands(self, nb_runner):
        nb_runner.create_notebook([
            "val = 1234567890",
            "formatted = f'{val:,}'\nprint(f'formatted={formatted}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted=1,234,567,890" in nb_runner.get_output(2)

    def test_zfill_edit(self, nb_runner):
        nb_runner.create_notebook([
            "code = '42'",
            "padded = code.zfill(6)\nprint(f'padded={padded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "padded=000042" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "code = '1234'")
        nb_runner.run_all()
        assert "padded=001234" in nb_runner.get_output(2)
