"""Batch 470: os.path and pathlib manipulation."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOsPathPathlib:
    def test_os_path_ops(self, nb_runner):
        nb_runner.create_notebook([
            "import os.path",
            "p = '/home/user/docs/file.txt'\nd = os.path.dirname(p)\nb = os.path.basename(p)\nname, ext = os.path.splitext(b)\nprint(f'd={d} b={b} name={name} ext={ext}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=/home/user/docs" in out
        assert "b=file.txt" in out
        assert "name=file" in out
        assert "ext=.txt" in out

    def test_pathlib_parts(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath",
            "p = PurePosixPath('/usr/local/bin/app')\nprint(f'parent={p.parent} stem={p.stem} suffix={p.suffix} name={p.name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "parent=/usr/local/bin" in out
        assert "stem=app" in out
        assert "name=app" in out

    def test_path_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import os.path",
            "p = '/a/b/c.py'\nprint(f'ext={os.path.splitext(p)[1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ext=.py" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "p = '/x/y/data.csv'\nprint(f'ext={os.path.splitext(p)[1]}')")
        nb_runner.run_all()
        assert "ext=.csv" in nb_runner.get_output(2)
