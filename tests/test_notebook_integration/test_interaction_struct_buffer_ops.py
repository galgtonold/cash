"""
Interaction test: struct pack_into and unpack_from with buffer.
Tests struct.pack_into and unpack_from for buffer operations,
Struct class precompilation, and cross-cell binary data processing.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStructBufferOps:
    """Test struct buffer operations across cells."""

    def test_struct_buffer(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: pack_into buffer
            "import struct\nimport ctypes\nbuf = bytearray(16)\nstruct.pack_into('>I', buf, 0, 12345)\nstruct.pack_into('>I', buf, 4, 67890)\nprint(f'buf_hex={buf[:8].hex()}')",
            # Cell 2: unpack_from
            "val1 = struct.unpack_from('>I', buf, 0)[0]\nval2 = struct.unpack_from('>I', buf, 4)[0]\nprint(f'val1={val1}')\nprint(f'val2={val2}')",
            # Cell 3: precompiled Struct
            "s = struct.Struct('>2I')\npacked = s.pack(111, 222)\na, b = s.unpack(packed)\nprint(f'a={a} b={b}')\nprint(f'struct_size={s.size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "val1=12345" in out2
        assert "val2=67890" in out2
        out3 = nb_runner.get_output(3)
        assert "a=111 b=222" in out3
        assert "struct_size=8" in out3

    def test_struct_buffer_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import struct\nfmt = struct.Struct('<3i')\ndata = fmt.pack(10, 20, 30)\nprint(f'size={fmt.size}')",
            "vals = fmt.unpack(data)\ntotal = sum(vals)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(2)

        # Edit values
        nb_runner.set_cell_source(1, "import struct\nfmt = struct.Struct('<3i')\ndata = fmt.pack(100, 200, 300)\nprint(f'size={fmt.size}')")
        nb_runner.run_cells([1, 2])
        assert "total=600" in nb_runner.get_output(2)

    def test_struct_buffer_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import struct\nheader = struct.pack('>BHI', 1, 256, 65536)\nprint(f'header_len={len(header)}')",
            "ver, length, offset = struct.unpack('>BHI', header)\nprint(f'ver={ver} length={length} offset={offset}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header_len=7" in nb_runner.get_output(1)
        assert "ver=1 length=256 offset=65536" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "ver=1 length=256 offset=65536" in nb_runner.get_output(2)
