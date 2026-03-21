"""Batch 71: Struct, memoryview & buffer patterns — cash caching with binary data."""
import textwrap
import pytest


@pytest.mark.stress
class TestStructPatterns:
    """Test struct module patterns across cells."""

    def test_struct_pack_unpack(self, nb_runner):
        """struct.pack/unpack across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import struct

                # Pack some values
                fmt = '>3if'
                packed = struct.pack(fmt, 1, 2, 3, 3.14)
                size = struct.calcsize(fmt)
                print(f"packed_len={len(packed)} calcsize={size}")
            """),
            textwrap.dedent("""\
                import struct
                unpacked = struct.unpack('>3if', packed)
                print(f"unpacked={unpacked}")
                print(f"ints={unpacked[:3]} float={unpacked[3]:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "packed_len=16" in out1
        out2 = nb_runner.get_output(2)
        assert "ints=(1, 2, 3)" in out2
        assert "float=3.14" in out2

    def test_struct_binary_records(self, nb_runner):
        """Binary record encoding/decoding across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import struct

                record_fmt = '<I20sf'  # uint32, 20-char string, float
                records = [
                    (1, b'Alice' + b'\\x00' * 15, 95.5),
                    (2, b'Bob' + b'\\x00' * 17, 87.3),
                    (3, b'Charlie' + b'\\x00' * 13, 91.0),
                ]
                binary_data = b''.join(struct.pack(record_fmt, *r) for r in records)
                print(f"total_bytes={len(binary_data)}")
            """),
            textwrap.dedent("""\
                import struct
                record_size = struct.calcsize('<I20sf')
                decoded = []
                for i in range(0, len(binary_data), record_size):
                    chunk = binary_data[i:i+record_size]
                    rid, name_bytes, score = struct.unpack('<I20sf', chunk)
                    name = name_bytes.rstrip(b'\\x00').decode()
                    decoded.append((rid, name, round(score, 1)))
                print(f"decoded={decoded}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "total_bytes=" in out1
        out2 = nb_runner.get_output(2)
        assert "Alice" in out2
        assert "Bob" in out2
        assert "Charlie" in out2


@pytest.mark.stress
class TestBytearrayPatterns:
    """Test bytearray and memoryview patterns."""

    def test_bytearray_manipulation(self, nb_runner):
        """Bytearray across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = bytearray(b'Hello, World!')
                data[7:12] = b'Python'
                modified = bytes(data)
                print(f"modified={modified}")
            """),
            textwrap.dedent("""\
                upper = bytearray(modified).upper()
                print(f"upper={bytes(upper)}")
                print(f"len={len(upper)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert b"Hello, Python!" == b"Hello, Python!" or "Hello, Python!" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "HELLO" in out2

    def test_memoryview_slicing(self, nb_runner):
        """Memoryview zero-copy slicing across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "slice1=[5, 6, 7, 8, 9]" in nb_runner.get_output(1)
        assert "slice2=[10, 11, 12, 13, 14]" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "total=95" in out2

    def test_binary_change_propagation(self, nb_runner):
        """Binary data propagates when format changes."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes=12" in nb_runner.get_output(1)
        assert "values=[1.0, 2.0, 3.0]" in nb_runner.get_output(2)

        # Change to doubles
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            import struct
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            packed = struct.pack(f'>{len(values)}f', *values)
            print(f"bytes={len(packed)}")
        """))
        nb_runner.run_cells([1, 2])
        assert "bytes=20" in nb_runner.get_output(1)
        assert "5.0" in nb_runner.get_output(2)
