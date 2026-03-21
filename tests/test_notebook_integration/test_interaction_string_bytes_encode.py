"""Batch 374: string encode/decode, base64, and bytes operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringBytesEncode:
    def test_encode_decode(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'Hello, World!'",
            "encoded = text.encode('utf-8')\ndecoded = encoded.decode('utf-8')\nhex_str = encoded.hex()\nprint(f'decoded={decoded} hex={hex_str[:10]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "decoded=Hello, World!" in out

    def test_base64_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import base64\nmessage = 'Hello'",
            "encoded = base64.b64encode(message.encode()).decode()\nprint(f'encoded={encoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=SGVsbG8=" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import base64\nmessage = 'World'")
        nb_runner.run_all()
        assert "encoded=V29ybGQ=" in nb_runner.get_output(2)

    def test_bytes_operations(self, nb_runner):
        nb_runner.create_notebook([
            "data = bytes([72, 101, 108, 108, 111])",
            "text = data.decode('ascii')\nlength = len(data)\nprint(f'text={text} length={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text=Hello length=5" in nb_runner.get_output(2)
