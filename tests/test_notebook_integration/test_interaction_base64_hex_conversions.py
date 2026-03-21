"""Batch 522: base64 encode decode and hex conversions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBase64HexConversions:
    def test_base64_roundtrip(self, nb_runner):
        nb_runner.create_notebook([
            "import base64",
            "text = 'Hello, World!'\nencoded = base64.b64encode(text.encode()).decode()\ndecoded = base64.b64decode(encoded).decode()\nprint(f'encoded={encoded} decoded={decoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "encoded=SGVsbG8sIFdvcmxkIQ==" in out
        assert "decoded=Hello, World!" in out

    def test_hex_conversion(self, nb_runner):
        nb_runner.create_notebook([
            "pass  # setup",
            "data = bytes([0, 127, 255, 16])\nhex_str = data.hex()\nback = bytes.fromhex(hex_str)\nprint(f'hex={hex_str} match={data == back}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "hex=007fff10" in out
        assert "match=True" in out

    def test_base64_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import base64",
            "msg = 'abc'\nenc = base64.b64encode(msg.encode()).decode()\nprint(f'enc={enc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "enc=YWJj" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "msg = 'xyz'\nenc = base64.b64encode(msg.encode()).decode()\nprint(f'enc={enc}')")
        nb_runner.run_all()
        assert "enc=eHl6" in nb_runner.get_output(2)
