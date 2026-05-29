"""Batch 402: string encoding and byte operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringEncodingBytes:
    def test_encode_decode(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'hello world'",
            "encoded = text.encode('utf-8')\ndecoded = encoded.decode('utf-8')\nhex_repr = encoded.hex()\nprint(f'decoded={decoded} hex={hex_repr}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "decoded=hello world" in nb_runner.get_output(2)
        assert "hex=68656c6c6f20776f726c64" in nb_runner.get_output(2)

    def test_bytes_from_hex(self, nb_runner):
        nb_runner.create_notebook([
            "hex_str = '48454c4c4f'",
            "data = bytes.fromhex(hex_str)\ntext = data.decode('ascii')\nprint(f'text={text} len={len(data)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text=HELLO" in nb_runner.get_output(2)
        assert "len=5" in nb_runner.get_output(2)

