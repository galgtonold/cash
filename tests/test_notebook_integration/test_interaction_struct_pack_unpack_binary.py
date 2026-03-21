"""Batch 527: struct pack unpack binary format."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStructPackUnpackBinary:
    def test_pack_unpack(self, nb_runner):
        nb_runner.create_notebook([
            "import struct",
            "packed = struct.pack('>ihf', 42, 1000, 3.14)\nsize = len(packed)\nunpacked = struct.unpack('>ihf', packed)\nprint(f'size={size} val0={unpacked[0]} val1={unpacked[1]} val2={unpacked[2]:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "size=10" in out
        assert "val0=42" in out
        assert "val1=1000" in out

    def test_struct_calcsize(self, nb_runner):
        nb_runner.create_notebook([
            "import struct",
            "fmt = '>3i2f'\nsize = struct.calcsize(fmt)\npacked = struct.pack(fmt, 1, 2, 3, 4.0, 5.0)\nvals = struct.unpack(fmt, packed)\nprint(f'size={size} vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "size=20" in out

    def test_struct_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import struct",
            "packed = struct.pack('>2i', 10, 20)\na, b = struct.unpack('>2i', packed)\nprint(f'a={a} b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "packed = struct.pack('>2i', 100, 200)\na, b = struct.unpack('>2i', packed)\nprint(f'a={a} b={b}')")
        nb_runner.run_all()
        assert "a=100 b=200" in nb_runner.get_output(2)
