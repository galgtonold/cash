"""
Interaction test: bytes and bytearray encoding operations.
Tests bytes/bytearray construction, hex conversion,
encoding/decoding, and cross-cell binary pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBytesEncodingOps:
    """Test bytes and bytearray encoding across cells."""

    def test_bytes_encoding(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create bytes
            "text = 'Hello, World!'\nencoded = text.encode('utf-8')\nhex_str = encoded.hex()\nprint(f'length={len(encoded)}')\nprint(f'hex={hex_str}')",
            # Cell 2: decode and bytearray
            "decoded = encoded.decode('utf-8')\nba = bytearray(encoded)\nba[0] = ord('h')  # lowercase\nmodified = ba.decode('utf-8')\nprint(f'decoded={decoded}')\nprint(f'modified={modified}')",
            # Cell 3: from hex roundtrip
            "restored = bytes.fromhex(hex_str)\nprint(f'restored={restored.decode(\"utf-8\")}')\nprint(f'matches={restored == encoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "length=13" in out1
        out2 = nb_runner.get_output(2)
        assert "decoded=Hello, World!" in out2
        assert "modified=hello, World!" in out2
        out3 = nb_runner.get_output(3)
        assert "restored=Hello, World!" in out3
        assert "matches=True" in out3

    def test_bytes_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = b'\\x01\\x02\\x03\\x04'\nprint(f'hex={data.hex()}')",
            "total = sum(data)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=10" in nb_runner.get_output(2)

        # Edit to different bytes
        nb_runner.set_cell_source(1, "data = b'\\x0a\\x14\\x1e\\x28'\nprint(f'hex={data.hex()}')")
        nb_runner.run_cells([1, 2])
        # 0x0a=10, 0x14=20, 0x1e=30, 0x28=40 => 100
        assert "total=100" in nb_runner.get_output(2)

    def test_bytes_cache(self, nb_runner):
        nb_runner.create_notebook([
            "msg = 'Python'\nencoded = msg.encode('ascii')\nprint(f'bytes={list(encoded)}')",
            "upper = bytes([b - 32 if 97 <= b <= 122 else b for b in encoded])\nprint(f'upper={upper.decode(\"ascii\")}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "upper=PYTHON" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "upper=PYTHON" in nb_runner.get_output(2)
