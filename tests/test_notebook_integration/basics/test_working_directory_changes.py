"""Relative file reads after the notebook changes its working directory."""

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestWorkingDirectoryChanges:
    """Test caching when working directory changes between cells."""

    @pytest.mark.files
    def test_chdir_and_relative_file_read(self, nb_runner, tmp_path):
        """Change working directory then read file with relative path."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        data_file = subdir / "data.txt"
        data_file.write_text("hello from subdir", encoding="utf-8")
        subdir_str = str(subdir).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import os\nos.chdir('{subdir_str}')",
                "with open('data.txt') as f:\n    content = f.read()",
                "print(f'Content: {content}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Content: hello from subdir" in out

    @pytest.mark.files
    def test_chdir_back_and_forth(self, nb_runner, tmp_path):
        """Change to directory, read file, change back."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.txt").write_text("from dir1", encoding="utf-8")
        (dir2 / "b.txt").write_text("from dir2", encoding="utf-8")
        dir1_str = str(dir1).replace("\\", "/")
        dir2_str = str(dir2).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import os\nos.chdir('{dir1_str}')\nwith open('a.txt') as f:\n    a = f.read()",
                f"os.chdir('{dir2_str}')\nwith open('b.txt') as f:\n    b = f.read()",
                "print(f'a={a}, b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "a=from dir1" in out
        assert "b=from dir2" in out
