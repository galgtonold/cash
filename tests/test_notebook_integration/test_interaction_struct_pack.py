"""
Batch 332: struct pack/unpack patterns with caching.
Tests struct.pack, struct.unpack, calcsize, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestStructPackUnpack:
    """Test struct packing/unpacking caching."""

    def test_struct_pack_unpack(self, nb_runner):
        """Pack and unpack with struct, verify caching."""
        nb_runner.create_notebook([
            "import struct",
            "fmt = '>iif'\ndata = struct.pack(fmt, 1, 2, 3.14)",
            "unpacked = struct.unpack(fmt, data)\nprint(f'vals={unpacked[0]},{unpacked[1]},{unpacked[2]:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "vals=1,2,3.14" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "vals=1,2,3.14" in out2

    def test_struct_edit_values(self, nb_runner):
        """Edit packed values, verify unpacking changes."""
        nb_runner.create_notebook([
            "import struct",
            "values = (10, 20)",
            "packed = struct.pack('!hh', *values)\nresult = struct.unpack('!hh', packed)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=(10, 20)" in out

        nb_runner.set_cell_source(2, "values = (100, 200)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=(100, 200)" in out2

    def test_struct_calcsize(self, nb_runner):
        """struct.calcsize for format strings."""
        nb_runner.create_notebook([
            "import struct",
            "fmt = '>3i2f'",
            "size = struct.calcsize(fmt)\nprint(f'size={size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "size=20" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "size=20" in out2
