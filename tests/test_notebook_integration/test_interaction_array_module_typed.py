"""Batch 488: array module typed numeric arrays."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestArrayModuleTyped:
    def test_int_array(self, nb_runner):
        nb_runner.create_notebook([
            "import array",
            "a = array.array('i', [1, 2, 3, 4, 5])\na.append(6)\ntotal = sum(a)\nprint(f'len={len(a)} sum={total} type={a.typecode}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len=6" in out
        assert "sum=21" in out
        assert "type=i" in out

    def test_float_array(self, nb_runner):
        nb_runner.create_notebook([
            "import array",
            "f = array.array('d', [1.1, 2.2, 3.3])\nf.extend([4.4, 5.5])\navg = round(sum(f) / len(f), 2)\nprint(f'len={len(f)} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len=5" in out
        assert "avg=3.3" in out

    def test_array_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import array",
            "a = array.array('i', [10, 20, 30])\nprint(f'sum={sum(a)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=60" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "a = array.array('i', [100, 200])\nprint(f'sum={sum(a)}')")
        nb_runner.run_all()
        assert "sum=300" in nb_runner.get_output(2)
