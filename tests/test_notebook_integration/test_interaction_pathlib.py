"""
Batch 283: Pathlib / os.path interaction tests.
Tests that editing path construction logic properly invalidates
downstream cells that use the constructed paths.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestPathlibInteraction:
    """Test pathlib/os.path patterns with cache invalidation."""

    def test_pathlib_join_edit(self, nb_runner):
        """Editing path components should propagate."""
        nb_runner.create_notebook([
            "from pathlib import Path\nbase = Path('/home/user')",
            "full = base / 'documents' / 'report.txt'",
            "result = str(full)",
            "print(f'path={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        # pathlib normalizes to OS-specific separator
        assert "report.txt" in out
        assert "documents" in out

        nb_runner.set_cell_source(1, "from pathlib import Path\nbase = Path('/srv/data')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "report.txt" in out
        assert "data" in out

    def test_path_parts_edit(self, nb_runner):
        """Editing a path and using its parts should propagate."""
        nb_runner.create_notebook([
            "from pathlib import Path\np = Path('/a/b/c/file.txt')",
            "stem = p.stem\nsuffix = p.suffix\nparent_name = p.parent.name",
            "info = f'{stem}{suffix} in {parent_name}'",
            "print(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=file.txt in c" in out

        nb_runner.set_cell_source(1, "from pathlib import Path\np = Path('/x/y/z/data.csv')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "info=data.csv in z" in out

    def test_os_path_manipulation_edit(self, nb_runner):
        """Editing os.path operations should propagate."""
        nb_runner.create_notebook([
            "import os.path\ndir_part = '/home/user'\nfile_part = 'notes.md'",
            "joined = os.path.join(dir_part, file_part)",
            "base = os.path.basename(joined)\next = os.path.splitext(base)[1]",
            "print(f'base={base},ext={ext}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "base=notes.md,ext=.md" in out

        nb_runner.set_cell_source(1, "import os.path\ndir_part = '/var/log'\nfile_part = 'access.log'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "base=access.log,ext=.log" in out
