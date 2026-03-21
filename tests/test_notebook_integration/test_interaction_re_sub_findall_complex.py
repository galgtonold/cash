"""Batch 483: re.sub and re.findall complex patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestReSubFindallComplex:
    def test_sub_replacement(self, nb_runner):
        nb_runner.create_notebook([
            "import re",
            "text = 'Call 123-456-7890 or 098-765-4321'\nmasked = re.sub(r'(\\d{3})-(\\d{3})-(\\d{4})', r'XXX-XXX-\\3', text)\nprint(f'masked={masked}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "XXX-XXX-7890" in out
        assert "XXX-XXX-4321" in out

    def test_findall_groups(self, nb_runner):
        nb_runner.create_notebook([
            "import re",
            "text = 'prices: $12.50, $3.99, $100.00'\nprices = re.findall(r'\\$(\\d+\\.\\d{2})', text)\ntotal = sum(float(p) for p in prices)\nprint(f'prices={prices} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "12.50" in out
        assert "total=116.49" in out

    def test_re_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import re",
            "text = 'abc 123 def'\nnums = re.findall(r'\\d+', text)\nprint(f'nums={nums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "nums=['123']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "text = 'a1 b22 c333'\nnums = re.findall(r'\\d+', text)\nprint(f'nums={nums}')")
        nb_runner.run_all()
        assert "nums=['1', '22', '333']" in nb_runner.get_output(2)
