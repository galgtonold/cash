"""Batch 383: hashlib hashing and digest comparison."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHashlibDigest:
    def test_sha256(self, nb_runner):
        nb_runner.create_notebook([
            "import hashlib\ntext = 'Hello, World!'",
            "h = hashlib.sha256(text.encode()).hexdigest()\nprint(f'hash={h[:16]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hash=dffd6021bb2bd5b0" in nb_runner.get_output(2)

    def test_hash_edit_input(self, nb_runner):
        nb_runner.create_notebook([
            "import hashlib\nmsg = 'test'",
            "h = hashlib.md5(msg.encode()).hexdigest()\nprint(f'md5={h}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "md5=098f6bcd4621d373cade4e832627b4f6" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import hashlib\nmsg = 'hello'")
        nb_runner.run_all()
        assert "md5=5d41402abc4b2a76b9719d911017c592" in nb_runner.get_output(2)

    def test_hash_compare(self, nb_runner):
        nb_runner.create_notebook([
            "import hashlib\ndef hash_str(s):\n    return hashlib.sha1(s.encode()).hexdigest()[:8]",
            "h1 = hash_str('abc')\nh2 = hash_str('abc')\nh3 = hash_str('xyz')\nmatch = h1 == h2\ndiff = h1 != h3\nprint(f'match={match} diff={diff}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match=True diff=True" in nb_runner.get_output(2)
