"""
Interaction test: pathlib Path operations and manipulation.
Tests Path construction, parts, stem, suffix,
parent traversal, and cross-cell path pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPathlibPathOps:
    """Test pathlib Path operations across cells."""

    def test_pathlib_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: path construction and parts
            "from pathlib import PurePosixPath as PP\np = PP('/home/user/documents/report.pdf')\nprint(f'name={p.name}')\nprint(f'stem={p.stem}')\nprint(f'suffix={p.suffix}')\nprint(f'parts={p.parts}')",
            # Cell 2: parent and ancestors
            "parent = p.parent\ngrandparent = p.parent.parent\nprint(f'parent={parent}')\nprint(f'grandparent={grandparent}')",
            # Cell 3: path manipulation
            "new_path = p.with_suffix('.txt')\nrenamed = p.with_name('summary.docx')\nprint(f'new_suffix={new_path}')\nprint(f'renamed={renamed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "name=report.pdf" in out1
        assert "stem=report" in out1
        assert "suffix=.pdf" in out1
        out2 = nb_runner.get_output(2)
        assert "parent=/home/user/documents" in out2
        assert "grandparent=/home/user" in out2
        out3 = nb_runner.get_output(3)
        assert "new_suffix=/home/user/documents/report.txt" in out3
        assert "renamed=/home/user/documents/summary.docx" in out3

    def test_pathlib_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath as PP\np = PP('/data/input.csv')\ninfo = f'{p.stem}{p.suffix}'\nprint(f'info={info}')",
            "full = str(p)\nprint(f'full={full}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=input.csv" in nb_runner.get_output(1)
        assert "full=/data/input.csv" in nb_runner.get_output(2)

        # Edit path
        nb_runner.set_cell_source(1, "from pathlib import PurePosixPath as PP\np = PP('/output/results.json')\ninfo = f'{p.stem}{p.suffix}'\nprint(f'info={info}')")
        nb_runner.run_cells([1, 2])
        assert "info=results.json" in nb_runner.get_output(1)
        assert "full=/output/results.json" in nb_runner.get_output(2)

    def test_pathlib_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from pathlib import PurePosixPath as PP\np = PP('/a/b/c/d.txt')\ndepth = len(p.parts) - 1  # minus root\nprint(f'depth={depth}')",
            "is_deep = depth > 2\nprint(f'is_deep={is_deep}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "depth=4" in nb_runner.get_output(1)
        assert "is_deep=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_deep=True" in nb_runner.get_output(2)
