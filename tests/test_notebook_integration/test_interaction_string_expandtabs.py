"""Batch 459: string expandtabs and whitespace handling."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringExpandtabs:
    def test_expandtabs(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'col1\\tcol2\\tcol3'",
            "expanded = text.expandtabs(8)\ncols = expanded.split()\nprint(f'cols={cols}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cols=['col1', 'col2', 'col3']" in nb_runner.get_output(2)

    def test_strip_variations(self, nb_runner):
        nb_runner.create_notebook([
            "s = '  hello  '",
            "l = s.lstrip()\nr = s.rstrip()\nb = s.strip()\nprint(f'l=[{l}] r=[{r}] b=[{b}]')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "l=[hello  ]" in out
        assert "r=[  hello]" in out
        assert "b=[hello]" in out

    def test_whitespace_edit(self, nb_runner):
        nb_runner.create_notebook([
            "raw = '  spaces  and\\ttabs  '",
            "cleaned = ' '.join(raw.split())\nprint(f'cleaned={cleaned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=spaces and tabs" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "raw = '\\t\\thello\\t\\tworld\\t\\t'")
        nb_runner.run_all()
        assert "cleaned=hello world" in nb_runner.get_output(2)
