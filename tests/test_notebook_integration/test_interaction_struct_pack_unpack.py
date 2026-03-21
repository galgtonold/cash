"""Batch 448: struct pack/unpack binary data."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStructPackUnpack:
    def test_pack_unpack(self, nb_runner):
        nb_runner.create_notebook([
            "import struct\nfmt = '>2i'\ndata = struct.pack(fmt, 100, 200)",
            "a, b = struct.unpack(fmt, data)\nsize = struct.calcsize(fmt)\nprint(f'a={a} b={b} size={size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=100" in nb_runner.get_output(2)
        assert "b=200" in nb_runner.get_output(2)
        assert "size=8" in nb_runner.get_output(2)

    def test_pack_float(self, nb_runner):
        nb_runner.create_notebook([
            "import struct\nfmt = '>f'\npacked = struct.pack(fmt, 3.14)",
            "val = struct.unpack(fmt, packed)[0]\nresult = round(val, 2)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3.14" in nb_runner.get_output(2)

    def test_struct_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import struct\nval = 42",
            "packed = struct.pack('>i', val)\nunpacked = struct.unpack('>i', packed)[0]\nprint(f'unpacked={unpacked}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unpacked=42" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import struct\nval = 9999")
        nb_runner.run_all()
        assert "unpacked=9999" in nb_runner.get_output(2)
