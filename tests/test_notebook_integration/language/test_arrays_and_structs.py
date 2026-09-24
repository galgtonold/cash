"""array and struct: typed arrays and packed binary data across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestArrayModuleTyped:
    """array module typed numeric arrays."""

    def test_int_array(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import array",
                "a = array.array('i', [1, 2, 3, 4, 5])\na.append(6)\ntotal = sum(a)\nprint(f'len={len(a)} sum={total} type={a.typecode}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len=6" in out
        assert "sum=21" in out
        assert "type=i" in out

    def test_float_array(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import array",
                "f = array.array('d', [1.1, 2.2, 3.3])\nf.extend([4.4, 5.5])\navg = round(sum(f) / len(f), 2)\nprint(f'len={len(f)} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "len=5" in out
        assert "avg=3.3" in out

    def test_array_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import array",
                "a = array.array('i', [10, 20, 30])\nprint(f'sum={sum(a)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=60" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "a = array.array('i', [100, 200])\nprint(f'sum={sum(a)}')")
        nb_runner.run_all()
        assert "sum=300" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestArrayTypedArrays:
    """Test array module typed arrays across cells."""

    def test_array_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create typed arrays
                "from array import array\nints = array('i', [1, 2, 3, 4, 5])\nfloats = array('d', [1.1, 2.2, 3.3])\nprint(f'ints={ints.tolist()}')\nprint(f'floats={floats.tolist()}')\nprint(f'typecode={ints.typecode}')",
                # Cell 2: array operations
                "ints.append(6)\nints.extend([7, 8])\nprint(f'extended={ints.tolist()}')\nprint(f'count={len(ints)}')",
                # Cell 3: tobytes/frombytes roundtrip
                "b = ints.tobytes()\nrestored = array('i')\nrestored.frombytes(b)\nprint(f'restored={restored.tolist()}')\nprint(f'matches={ints == restored}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from array import array\narr = array('f', [1.0, 2.0, 3.0])\nprint(f'arr={arr.tolist()}')",
                "total = sum(arr)\nprint(f'total={total:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=6.0" in nb_runner.get_output(2)

        # Edit array values
        nb_runner.set_cell_source(
            1, "from array import array\narr = array('f', [10.0, 20.0, 30.0, 40.0])\nprint(f'arr={arr.tolist()}')"
        )
        nb_runner.run_cells([1, 2])
        assert "total=100.0" in nb_runner.get_output(2)

    def test_array_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from array import array\na = array('i', range(10))\nprint(f'length={len(a)}')",
                "s = sum(a)\nprint(f'sum={s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length=10" in nb_runner.get_output(1)
        assert "sum=45" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "sum=45" in nb_runner.get_output(2)


class TestStructPackUnpack:
    """Test struct packing/unpacking caching."""

    @pytest.mark.integration
    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_struct_pack_unpack(self, nb_runner):
        """Pack and unpack with struct, verify caching."""
        nb_runner.create_notebook(
            [
                "import struct",
                "fmt = '>iif'\ndata = struct.pack(fmt, 1, 2, 3.14)",
                "unpacked = struct.unpack(fmt, data)\nprint(f'vals={unpacked[0]},{unpacked[1]},{unpacked[2]:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "vals=1,2,3.14" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "vals=1,2,3.14" in out2

    @pytest.mark.integration
    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_struct_edit_values(self, nb_runner):
        """Edit packed values, verify unpacking changes."""
        nb_runner.create_notebook(
            [
                "import struct",
                "values = (10, 20)",
                "packed = struct.pack('!hh', *values)\nresult = struct.unpack('!hh', packed)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=(10, 20)" in out

        nb_runner.set_cell_source(2, "values = (100, 200)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=(100, 200)" in out2

    @pytest.mark.integration
    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_struct_calcsize(self, nb_runner):
        """struct.calcsize for format strings."""
        nb_runner.create_notebook(
            [
                "import struct",
                "fmt = '>3i2f'",
                "size = struct.calcsize(fmt)\nprint(f'size={size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "size=20" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "size=20" in out2

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_pack_unpack(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct\nfmt = '>2i'\ndata = struct.pack(fmt, 100, 200)",
                "a, b = struct.unpack(fmt, data)\nsize = struct.calcsize(fmt)\nprint(f'a={a} b={b} size={size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=100" in nb_runner.get_output(2)
        assert "b=200" in nb_runner.get_output(2)
        assert "size=8" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_pack_float(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct\nfmt = '>f'\npacked = struct.pack(fmt, 3.14)",
                "val = struct.unpack(fmt, packed)[0]\nresult = round(val, 2)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3.14" in nb_runner.get_output(2)

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_struct_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct\nval = 42",
                "packed = struct.pack('>i', val)\nunpacked = struct.unpack('>i', packed)[0]\nprint(f'unpacked={unpacked}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unpacked=42" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "import struct\nval = 9999")
        nb_runner.run_all()
        assert "unpacked=9999" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStructPackUnpackBinary:
    """struct pack unpack binary format."""

    def test_pack_unpack(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct",
                "packed = struct.pack('>ihf', 42, 1000, 3.14)\nsize = len(packed)\nunpacked = struct.unpack('>ihf', packed)\nprint(f'size={size} val0={unpacked[0]} val1={unpacked[1]} val2={unpacked[2]:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "size=10" in out
        assert "val0=42" in out
        assert "val1=1000" in out

    def test_struct_calcsize(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct",
                "fmt = '>3i2f'\nsize = struct.calcsize(fmt)\npacked = struct.pack(fmt, 1, 2, 3, 4.0, 5.0)\nvals = struct.unpack(fmt, packed)\nprint(f'size={size} vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "size=20" in out

    def test_struct_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct",
                "packed = struct.pack('>2i', 10, 20)\na, b = struct.unpack('>2i', packed)\nprint(f'a={a} b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "packed = struct.pack('>2i', 100, 200)\na, b = struct.unpack('>2i', packed)\nprint(f'a={a} b={b}')"
        )
        nb_runner.run_all()
        assert "a=100 b=200" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStructPackMixedFormats:
    """struct pack unpack mixed binary formats."""

    def test_pack_unpack_ints(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct",
                "packed = struct.pack('>3i', 10, 20, 30)\nunpacked = struct.unpack('>3i', packed)\nprint(f'size={len(packed)} vals={unpacked}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "size=12" in out
        assert "(10, 20, 30)" in out

    def test_pack_mixed_types(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct",
                "packed = struct.pack('>if10s', 42, 3.14, b'helloworld')\ni, f, s = struct.unpack('>if10s', packed)\nprint(f'i={i} f={round(f,2)} s={s.decode()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "i=42" in out
        assert "f=3.14" in out
        assert "s=helloworld" in out

    def test_struct_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct",
                "data = struct.pack('>2h', 100, 200)\nvals = struct.unpack('>2h', data)\nprint(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=(100, 200)" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "data = struct.pack('>3h', 10, 20, 30)\nvals = struct.unpack('>3h', data)\nprint(f'vals={vals}')"
        )
        nb_runner.run_all()
        assert "vals=(10, 20, 30)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStructBufferOps:
    """Test struct buffer operations across cells."""

    def test_struct_buffer(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: pack_into buffer
                "import struct\nimport ctypes\nbuf = bytearray(16)\nstruct.pack_into('>I', buf, 0, 12345)\nstruct.pack_into('>I', buf, 4, 67890)\nprint(f'buf_hex={buf[:8].hex()}')",
                # Cell 2: unpack_from
                "val1 = struct.unpack_from('>I', buf, 0)[0]\nval2 = struct.unpack_from('>I', buf, 4)[0]\nprint(f'val1={val1}')\nprint(f'val2={val2}')",
                # Cell 3: precompiled Struct
                "s = struct.Struct('>2I')\npacked = s.pack(111, 222)\na, b = s.unpack(packed)\nprint(f'a={a} b={b}')\nprint(f'struct_size={s.size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "val1=12345" in out2
        assert "val2=67890" in out2
        out3 = nb_runner.get_output(3)
        assert "a=111 b=222" in out3
        assert "struct_size=8" in out3

    def test_struct_buffer_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct\nfmt = struct.Struct('<3i')\ndata = fmt.pack(10, 20, 30)\nprint(f'size={fmt.size}')",
                "vals = fmt.unpack(data)\ntotal = sum(vals)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(2)

        # Edit values
        nb_runner.set_cell_source(
            1, "import struct\nfmt = struct.Struct('<3i')\ndata = fmt.pack(100, 200, 300)\nprint(f'size={fmt.size}')"
        )
        nb_runner.run_cells([1, 2])
        assert "total=600" in nb_runner.get_output(2)

    def test_struct_buffer_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import struct\nheader = struct.pack('>BHI', 1, 256, 65536)\nprint(f'header_len={len(header)}')",
                "ver, length, offset = struct.unpack('>BHI', header)\nprint(f'ver={ver} length={length} offset={offset}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header_len=7" in nb_runner.get_output(1)
        assert "ver=1 length=256 offset=65536" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "ver=1 length=256 offset=65536" in nb_runner.get_output(2)


@pytest.mark.stress
class TestBytearrayPatterns:
    """Test bytearray and memoryview patterns."""

    def test_memoryview_slicing(self, nb_runner):
        """Memoryview zero-copy slicing across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = bytearray(range(20))
                view = memoryview(data)
                slice1 = bytes(view[5:10])
                slice2 = bytes(view[10:15])
                print(f"slice1={list(slice1)}")
                print(f"slice2={list(slice2)}")
            """),
                textwrap.dedent("""\
                combined = list(slice1) + list(slice2)
                print(f"combined={combined}")
                print(f"total={sum(combined)}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "slice1=[5, 6, 7, 8, 9]" in nb_runner.get_output(1)
        assert "slice2=[10, 11, 12, 13, 14]" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "total=95" in out2

    def test_binary_change_propagation(self, nb_runner):
        """Binary data propagates when format changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import struct
                values = [1.0, 2.0, 3.0]
                packed = struct.pack(f'>{len(values)}f', *values)
                print(f"bytes={len(packed)}")
            """),
                textwrap.dedent("""\
                import struct
                count = len(packed) // 4
                unpacked = list(struct.unpack(f'>{count}f', packed))
                print(f"values={unpacked}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes=12" in nb_runner.get_output(1)
        assert "values=[1.0, 2.0, 3.0]" in nb_runner.get_output(2)

        # Change to doubles
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            import struct
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            packed = struct.pack(f'>{len(values)}f', *values)
            print(f"bytes={len(packed)}")
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "bytes=20" in nb_runner.get_output(1)
        assert "5.0" in nb_runner.get_output(2)
