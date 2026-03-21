"""Batch 69: Hashlib, secrets & crypto patterns — cash caching with hashing/security."""
import textwrap
import pytest


@pytest.mark.stress
class TestHashlibPatterns:
    """Test hashlib patterns across cells."""

    def test_hash_computation(self, nb_runner):
        """Hash computation results cached across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import hashlib

                data = b"Hello, World!"
                md5 = hashlib.md5(data).hexdigest()
                sha256 = hashlib.sha256(data).hexdigest()
                print(f"md5={md5}")
                print(f"sha256_prefix={sha256[:16]}")
            """),
            textwrap.dedent("""\
                # Use hashes downstream
                combined = md5 + ':' + sha256
                print(f"combined_len={len(combined)}")
                print(f"md5_len={len(md5)} sha256_len={len(sha256)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "md5=" in out1
        assert "sha256_prefix=" in out1
        out2 = nb_runner.get_output(2)
        assert "md5_len=32" in out2
        assert "sha256_len=64" in out2

    def test_hash_file_content(self, nb_runner, tmp_path):
        """Hash file content across cells."""
        test_file = tmp_path / "hashdata" / "sample.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Test content for hashing")
        fpath = str(test_file).replace('\\', '/')

        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hash=" in nb_runner.get_output(1)
        assert "content_size=24" in nb_runner.get_output(2)

    def test_hmac_computation(self, nb_runner):
        """HMAC computation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import hmac
                import hashlib

                key = b'secret-key'
                message = b'important data'
                mac = hmac.new(key, message, hashlib.sha256).hexdigest()
                print(f"mac_len={len(mac)}")
            """),
            textwrap.dedent("""\
                # Verify
                import hmac as hmac2
                import hashlib as hashlib2
                verify_mac = hmac2.new(b'secret-key', b'important data', hashlib2.sha256).hexdigest()
                match = hmac.compare_digest(mac, verify_mac)
                print(f"match={match}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mac_len=64" in nb_runner.get_output(1)
        assert "match=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestSecretsPatterns:
    """Test secrets module patterns."""

    def test_token_generation_deterministic(self, nb_runner):
        """Deterministic token patterns (length checks, not value)."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import secrets

                token_hex = secrets.token_hex(16)
                token_url = secrets.token_urlsafe(16)
                print(f"hex_len={len(token_hex)}")
                print(f"url_len={len(token_url)}")
            """),
            textwrap.dedent("""\
                # Verify they are different types
                is_hex = all(c in '0123456789abcdef' for c in token_hex)
                print(f"is_hex={is_hex}")
                print(f"types_differ={token_hex != token_url}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex_len=32" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "is_hex=True" in out2

    def test_hash_change_propagation(self, nb_runner):
        """Hash result propagates when input changes."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        first_short = nb_runner.get_output(2).split("short=")[1].strip()

        # Change data
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            import hashlib
            data = "version2"
            digest = hashlib.sha256(data.encode()).hexdigest()
            print(f"digest={digest[:16]}")
        """))
        nb_runner.run_cells([1, 2])
        second_short = nb_runner.get_output(2).split("short=")[1].strip()
        assert first_short != second_short, "Hash should change when input changes"
