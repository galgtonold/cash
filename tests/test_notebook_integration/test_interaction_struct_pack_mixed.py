"""Batch 482: struct pack unpack mixed binary formats."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStructPackMixedFormats:
    def test_pack_unpack_ints(self, nb_runner):
        nb_runner.create_notebook([
            "import struct",
            "packed = struct.pack('>3i', 10, 20, 30)\nunpacked = struct.unpack('>3i', packed)\nprint(f'size={len(packed)} vals={unpacked}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "size=12" in out
        assert "(10, 20, 30)" in out

    def test_pack_mixed_types(self, nb_runner):
        nb_runner.create_notebook([
            "import struct",
            "packed = struct.pack('>if10s', 42, 3.14, b'helloworld')\ni, f, s = struct.unpack('>if10s', packed)\nprint(f'i={i} f={round(f,2)} s={s.decode()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "i=42" in out
        assert "f=3.14" in out
        assert "s=helloworld" in out

    def test_struct_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import struct",
            "data = struct.pack('>2h', 100, 200)\nvals = struct.unpack('>2h', data)\nprint(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=(100, 200)" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "data = struct.pack('>3h', 10, 20, 30)\nvals = struct.unpack('>3h', data)\nprint(f'vals={vals}')")
        nb_runner.run_all()
        assert "vals=(10, 20, 30)" in nb_runner.get_output(2)
