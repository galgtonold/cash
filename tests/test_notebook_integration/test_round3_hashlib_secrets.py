"""Hashlib, secrets & crypto patterns — cash caching with hashing/security."""

import textwrap

import pytest


@pytest.mark.stress
class TestHashlibPatterns:
    """Test hashlib patterns across cells."""

    def test_hash_file_content(self, nb_runner, tmp_path):
        """Hash file content across cells."""
        test_file = tmp_path / "hashdata" / "sample.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Test content for hashing")
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
