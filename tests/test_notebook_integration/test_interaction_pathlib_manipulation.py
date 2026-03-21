"""Batch 426: pathlib operations for path manipulation."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPathlibOps:
    def test_pathlib_parts(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath\np = PurePosixPath('/home/user/docs/file.txt')",
            "stem = p.stem\nsuffix = p.suffix\nparent = str(p.parent)\nprint(f'stem={stem} suffix={suffix} parent={parent}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=file" in out
        assert "suffix=.txt" in out
        assert "parent=/home/user/docs" in out

    def test_pathlib_join(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath\nbase = PurePosixPath('/data')",
            "full = base / 'subdir' / 'output.csv'\nprint(f'full={full}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "full=/data/subdir/output.csv" in nb_runner.get_output(2)

    def test_pathlib_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath\nf = PurePosixPath('report.pdf')",
            "new_f = f.with_suffix('.docx')\nprint(f'new={new_f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "new=report.docx" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from pathlib import PurePosixPath\nf = PurePosixPath('data.csv')")
        nb_runner.run_all()
        assert "new=data.docx" in nb_runner.get_output(2)
