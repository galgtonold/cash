"""Batch 344: os.path and pathlib path manipulation edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPathManipulation:
    def test_pathlib_operations(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath\np = PurePosixPath('/home/user/docs/file.txt')",
            "parts = list(p.parts)\nstem = p.stem\nsuffix = p.suffix\nparent = str(p.parent)\nprint(f'stem={stem} suffix={suffix} parent={parent}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=file" in out
        assert "suffix=.txt" in out
        assert "parent=/home/user/docs" in out

    def test_pathlib_edit_path(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath\np = PurePosixPath('/a/b/c.py')",
            "new_p = p.with_suffix('.txt')\nresult = str(new_p)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=/a/b/c.txt" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from pathlib import PurePosixPath\np = PurePosixPath('/x/y/data.csv')")
        nb_runner.run_all()
        assert "result=/x/y/data.txt" in nb_runner.get_output(2)

    def test_path_join_resolve(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath\nbase = PurePosixPath('/project')\nsub = 'src/main.py'",
            "full = base / sub\nname = full.name\nprint(f'full={full} name={name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "full=/project/src/main.py" in out
        assert "name=main.py" in out
