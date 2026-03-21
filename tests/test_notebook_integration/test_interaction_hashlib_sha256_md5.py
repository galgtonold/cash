"""Batch 525: hashlib sha256 md5 hexdigest computation."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestHashlibSha256Md5:
    def test_sha256(self, nb_runner):
        nb_runner.create_notebook([
            "import hashlib",
            "data = b'hello world'\nh = hashlib.sha256(data).hexdigest()\nprint(f'sha256={h[:16]}... len={len(h)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sha256=b94d27b9934d3e08" in out
        assert "len=64" in out

    def test_md5(self, nb_runner):
        nb_runner.create_notebook([
            "import hashlib",
            "data = b'test'\nh = hashlib.md5(data).hexdigest()\nprint(f'md5={h} len={len(h)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "md5=098f6bcd4621d373cade4e832627b4f6" in out
        assert "len=32" in out

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import hashlib",
            "h = hashlib.sha256(b'abc').hexdigest()[:8]\nprint(f'h={h}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "h=ba7816bf" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "h = hashlib.sha256(b'xyz').hexdigest()[:8]\nprint(f'h={h}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "h=" in out2
        assert "ba7816bf" not in out2  # different from abc
