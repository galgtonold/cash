"""
Interaction test: hashlib HMAC for message authentication.
Tests hmac.new with hashlib digests, compare_digest for timing-safe
comparison, and cross-cell HMAC verification pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHmacAuth:
    """Test HMAC message authentication across cells."""

    def test_hmac_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create HMAC
            "import hmac\nimport hashlib\nkey = b'secret-key'\nmessage = b'important data'\nmac = hmac.new(key, message, hashlib.sha256).hexdigest()\nprint(f'mac_len={len(mac)}')\nprint(f'mac_prefix={mac[:8]}')",
            # Cell 2: verify HMAC
            "mac2 = hmac.new(key, message, hashlib.sha256).hexdigest()\nis_valid = hmac.compare_digest(mac, mac2)\nprint(f'valid={is_valid}')",
            # Cell 3: different message = different HMAC
            "bad_mac = hmac.new(key, b'tampered data', hashlib.sha256).hexdigest()\nis_tampered = not hmac.compare_digest(mac, bad_mac)\nprint(f'tampered={is_tampered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mac_len=64" in out1
        out2 = nb_runner.get_output(2)
        assert "valid=True" in out2
        out3 = nb_runner.get_output(3)
        assert "tampered=True" in out3

    def test_hmac_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import hmac, hashlib\nkey = b'key1'\nmsg = b'hello'\ntag = hmac.new(key, msg, hashlib.sha256).hexdigest()\nprint(f'tag={tag[:12]}')",
            "tag_short = tag[:8]\nprint(f'short={tag_short}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2a = nb_runner.get_output(2)
        assert "short=" in out2a

        # Edit key
        nb_runner.set_cell_source(1, "import hmac, hashlib\nkey = b'key2'\nmsg = b'hello'\ntag = hmac.new(key, msg, hashlib.sha256).hexdigest()\nprint(f'tag={tag[:12]}')")
        nb_runner.run_cells([1, 2])
        out2b = nb_runner.get_output(2)
        assert "short=" in out2b

    def test_hmac_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import hmac, hashlib\nresult = hmac.new(b'k', b'data', hashlib.md5).hexdigest()\nprint(f'result_len={len(result)}')",
            "is_32 = len(result) == 32\nprint(f'is_32={is_32}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result_len=32" in nb_runner.get_output(1)
        assert "is_32=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_32=True" in nb_runner.get_output(2)
