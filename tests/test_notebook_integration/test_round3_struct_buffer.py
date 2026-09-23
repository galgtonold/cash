"""Struct, memoryview & buffer patterns — cash caching with binary data."""

import textwrap

import pytest


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
