"""
Batch 317: bytes encoding and decoding patterns with caching.
Tests str.encode, bytes.decode, hex conversion, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestBytesEncode:
    """Test bytes/encoding operation caching."""

    def test_encode_decode_roundtrip(self, nb_runner):
        """Encode string to bytes, decode back, verify caching."""
        nb_runner.create_notebook([
            "text = 'Hello, World!'",
            "encoded = text.encode('utf-8')",
            "decoded = encoded.decode('utf-8')\nprint(f'match={decoded == text}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "match=True" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "match=True" in out2

    def test_hex_conversion_edit(self, nb_runner):
        """Hex string conversion with edit."""
        nb_runner.create_notebook([
            "data = b'\\x48\\x65\\x6c\\x6c\\x6f'",
            "hex_str = data.hex()\ntext = data.decode('ascii')",
            "print(f'hex={hex_str} text={text}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "hex=48656c6c6f" in out
        assert "text=Hello" in out

        nb_runner.set_cell_source(1, "data = b'\\x57\\x6f\\x72\\x6c\\x64'")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "text=World" in out2

    def test_base64_encode(self, nb_runner):
        """Base64 encoding with caching."""
        nb_runner.create_notebook([
            "import base64",
            "msg = 'test data'",
            "enc = base64.b64encode(msg.encode()).decode()\ndec = base64.b64decode(enc).decode()",
            "print(f'enc={enc} match={dec == msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "match=True" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "match=True" in out2
