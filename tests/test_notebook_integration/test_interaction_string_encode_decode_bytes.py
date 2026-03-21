"""Batch 506: string encode decode and bytes operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringEncodeDecodeBytes:
    def test_encode_decode_utf8(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'Hello \\u00e9\\u00e8 \\u4e16\\u754c'",
            "encoded = text.encode('utf-8')\nsize = len(encoded)\ndecoded = encoded.decode('utf-8')\nprint(f'size={size} match={text == decoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "match=True" in out

    def test_bytes_operations(self, nb_runner):
        nb_runner.create_notebook([
            "data = b'Hello World'",
            "upper = data.upper()\nlower = data.lower()\nfound = data.find(b'World')\nprint(f'upper={upper} lower={lower} found={found}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "upper=b'HELLO WORLD'" in out
        assert "lower=b'hello world'" in out
        assert "found=6" in out

    def test_encode_edit(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'abc'",
            "b = text.encode('ascii')\nprint(f'len={len(b)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "text = 'abcdef'")
        nb_runner.run_all()
        assert "len=6" in nb_runner.get_output(2)
