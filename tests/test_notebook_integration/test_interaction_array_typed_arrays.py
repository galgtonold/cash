"""
Interaction test: array module typed arrays.
Tests array.array for typed numeric arrays, buffer protocol,
tobytes/frombytes, and cross-cell array processing pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestArrayTypedArrays:
    """Test array module typed arrays across cells."""

    def test_array_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create typed arrays
            "from array import array\nints = array('i', [1, 2, 3, 4, 5])\nfloats = array('d', [1.1, 2.2, 3.3])\nprint(f'ints={ints.tolist()}')\nprint(f'floats={floats.tolist()}')\nprint(f'typecode={ints.typecode}')",
            # Cell 2: array operations
            "ints.append(6)\nints.extend([7, 8])\nprint(f'extended={ints.tolist()}')\nprint(f'count={len(ints)}')",
            # Cell 3: tobytes/frombytes roundtrip
            "b = ints.tobytes()\nrestored = array('i')\nrestored.frombytes(b)\nprint(f'restored={restored.tolist()}')\nprint(f'matches={ints == restored}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "ints=[1, 2, 3, 4, 5]" in out1
        assert "typecode=i" in out1
        out2 = nb_runner.get_output(2)
        assert "extended=[1, 2, 3, 4, 5, 6, 7, 8]" in out2
        assert "count=8" in out2
        out3 = nb_runner.get_output(3)
        assert "matches=True" in out3

    def test_array_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from array import array\narr = array('f', [1.0, 2.0, 3.0])\nprint(f'arr={arr.tolist()}')",
            "total = sum(arr)\nprint(f'total={total:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=6.0" in nb_runner.get_output(2)

        # Edit array values
        nb_runner.set_cell_source(1, "from array import array\narr = array('f', [10.0, 20.0, 30.0, 40.0])\nprint(f'arr={arr.tolist()}')")
        nb_runner.run_cells([1, 2])
        assert "total=100.0" in nb_runner.get_output(2)

    def test_array_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from array import array\na = array('i', range(10))\nprint(f'length={len(a)}')",
            "s = sum(a)\nprint(f'sum={s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length=10" in nb_runner.get_output(1)
        assert "sum=45" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "sum=45" in nb_runner.get_output(2)
