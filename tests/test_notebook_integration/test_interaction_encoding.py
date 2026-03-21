"""Batch 205 – String encoding and bytes conversion interaction tests.

Tests editing string/bytes conversions, encoding schemes,
and byte manipulation patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestEncodingEdits:
    """Editing encoding/decoding operations."""

    def test_edit_encoding_scheme(self, nb_runner):
        """Edit the encoding scheme."""
        nb_runner.create_notebook([
            "text = 'Hello'  # encoding source",
            "encoded = text.encode('utf-8')\nprint(f'bytes = {encoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes = b'Hello'" in nb_runner.get_output(2)

        # Change to ascii
        nb_runner.set_cell_source(
            2, "encoded = text.encode('ascii')\nprint(f'bytes = {encoded}')"
        )
        nb_runner.run_all()
        assert "bytes = b'Hello'" in nb_runner.get_output(2)

    def test_edit_bytes_to_hex(self, nb_runner):
        """Edit bytes to hex conversion."""
        nb_runner.create_notebook([
            "data = b'\\x48\\x65\\x6c\\x6c\\x6f'  # bytes hex source",
            "hex_str = data.hex()\nprint(f'hex = {hex_str}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex = 48656c6c6f" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = b'\\x41\\x42\\x43'  # bytes hex source v2")
        nb_runner.run_all()
        assert "hex = 414243" in nb_runner.get_output(2)

    def test_edit_base64_roundtrip(self, nb_runner):
        """Edit base64 encode/decode."""
        nb_runner.create_notebook([
            "import base64",
            "original = 'Hello World'  # base64 source",
            "encoded = base64.b64encode(original.encode()).decode()\nprint(f'encoded = {encoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded = SGVsbG8gV29ybGQ=" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "original = 'Python'  # base64 source v2")
        nb_runner.run_all()
        assert "encoded = UHl0aG9u" in nb_runner.get_output(3)
