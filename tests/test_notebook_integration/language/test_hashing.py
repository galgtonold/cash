"""hashlib, hmac and secrets across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHashlibDigest:
    """hashlib hashing and digest comparison."""

    def test_sha256(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib\ntext = 'Hello, World!'",
                "h = hashlib.sha256(text.encode()).hexdigest()\nprint(f'hash={h[:16]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hash=dffd6021bb2bd5b0" in nb_runner.get_output(2)

    def test_hash_edit_input(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib\nmsg = 'test'",
                "h = hashlib.md5(msg.encode()).hexdigest()\nprint(f'md5={h}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "md5=098f6bcd4621d373cade4e832627b4f6" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import hashlib\nmsg = 'hello'")
        nb_runner.run_all()
        assert "md5=5d41402abc4b2a76b9719d911017c592" in nb_runner.get_output(2)

    def test_hash_compare(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib\ndef hash_str(s):\n    return hashlib.sha1(s.encode()).hexdigest()[:8]",
                "h1 = hash_str('abc')\nh2 = hash_str('abc')\nh3 = hash_str('xyz')\nmatch = h1 == h2\ndiff = h1 != h3\nprint(f'match={match} diff={diff}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "match=True diff=True" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHashlibSha256Md5:
    """hashlib sha256 md5 hexdigest computation."""

    def test_sha256(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib",
                "data = b'hello world'\nh = hashlib.sha256(data).hexdigest()\nprint(f'sha256={h[:16]}... len={len(h)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sha256=b94d27b9934d3e08" in out
        assert "len=64" in out

    def test_md5(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib",
                "data = b'test'\nh = hashlib.md5(data).hexdigest()\nprint(f'md5={h} len={len(h)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "md5=098f6bcd4621d373cade4e832627b4f6" in out
        assert "len=32" in out

    def test_hash_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hashlib",
                "h = hashlib.sha256(b'abc').hexdigest()[:8]\nprint(f'h={h}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "h=ba7816bf" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "h = hashlib.sha256(b'xyz').hexdigest()[:8]\nprint(f'h={h}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "h=" in out2
        assert "ba7816bf" not in out2  # different from abc


@pytest.mark.stress
class TestHashlibPatterns:
    """Test hashlib patterns across cells."""

    def test_hash_file_content(self, nb_runner, tmp_path):
        """Hash file content across cells."""
        test_file = tmp_path / "hashdata" / "sample.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Test content for hashing", encoding="utf-8")
        fpath = str(test_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                textwrap.dedent(f"""\
                import hashlib

                with open('{fpath}', 'rb') as f:
                    content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                print(f"hash={{file_hash[:16]}}")
            """),
                textwrap.dedent("""\
                print(f"content_size={len(content)}")
                print(f"hash_type=sha256")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hash=" in nb_runner.get_output(1)
        assert "content_size=24" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestHmacAuth:
    """Test HMAC message authentication across cells."""

    def test_hmac_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create HMAC
                "import hmac\nimport hashlib\nkey = b'secret-key'\nmessage = b'important data'\nmac = hmac.new(key, message, hashlib.sha256).hexdigest()\nprint(f'mac_len={len(mac)}')\nprint(f'mac_prefix={mac[:8]}')",
                # Cell 2: verify HMAC
                "mac2 = hmac.new(key, message, hashlib.sha256).hexdigest()\nis_valid = hmac.compare_digest(mac, mac2)\nprint(f'valid={is_valid}')",
                # Cell 3: different message = different HMAC
                "bad_mac = hmac.new(key, b'tampered data', hashlib.sha256).hexdigest()\nis_tampered = not hmac.compare_digest(mac, bad_mac)\nprint(f'tampered={is_tampered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "mac_len=64" in out1
        out2 = nb_runner.get_output(2)
        assert "valid=True" in out2
        out3 = nb_runner.get_output(3)
        assert "tampered=True" in out3

    def test_hmac_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hmac, hashlib\nkey = b'key1'\nmsg = b'hello'\ntag = hmac.new(key, msg, hashlib.sha256).hexdigest()\nprint(f'tag={tag[:12]}')",
                "tag_short = tag[:8]\nprint(f'short={tag_short}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2a = nb_runner.get_output(2)
        assert "short=" in out2a

        # Edit key
        nb_runner.set_cell_source(
            1,
            "import hmac, hashlib\nkey = b'key2'\nmsg = b'hello'\ntag = hmac.new(key, msg, hashlib.sha256).hexdigest()\nprint(f'tag={tag[:12]}')",
        )
        nb_runner.run_cells([1, 2])
        out2b = nb_runner.get_output(2)
        assert "short=" in out2b

    def test_hmac_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import hmac, hashlib\nresult = hmac.new(b'k', b'data', hashlib.md5).hexdigest()\nprint(f'result_len={len(result)}')",
                "is_32 = len(result) == 32\nprint(f'is_32={is_32}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result_len=32" in nb_runner.get_output(1)
        assert "is_32=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_32=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestSecretsPatterns:
    """Test secrets module patterns."""

    def test_hash_change_propagation(self, nb_runner):
        """Hash result propagates when input changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import hashlib
                data = "version1"
                digest = hashlib.sha256(data.encode()).hexdigest()
                print(f"digest={digest[:16]}")
            """),
                textwrap.dedent("""\
                short = digest[:8]
                print(f"short={short}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        first_short = nb_runner.get_output(2).split("short=")[1].strip()

        # Change data
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            import hashlib
            data = "version2"
            digest = hashlib.sha256(data.encode()).hexdigest()
            print(f"digest={digest[:16]}")
        """),
        )
        nb_runner.run_cells([1, 2])
        second_short = nb_runner.get_output(2).split("short=")[1].strip()
        assert first_short != second_short, "Hash should change when input changes"
