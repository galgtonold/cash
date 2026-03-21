"""
Interaction test: os.path vs pathlib cross-usage.
Tests mixing os.path and pathlib operations with string conversions
and cross-cell path manipulation.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOsPathPathlibCross:
    """Test os.path and pathlib cross-usage across cells."""

    def test_path_cross_usage(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: build paths with both APIs
            "import os\nfrom pathlib import PurePosixPath\nop = os.path.join('home', 'user', 'docs', 'file.txt')\npp = PurePosixPath('home') / 'user' / 'docs' / 'file.txt'\nprint(f'op_base={os.path.basename(op)}')\nprint(f'pp_name={pp.name}')",
            # Cell 2: extract components
            "op_dir = os.path.dirname(op)\npp_parent = str(pp.parent)\nop_ext = os.path.splitext(op)[1]\npp_suffix = pp.suffix\nprint(f'op_dir={op_dir}')\nprint(f'pp_parent={pp_parent}')\nprint(f'ext_match={op_ext == pp_suffix}')",
            # Cell 3: manipulate
            "new_pp = pp.with_suffix('.md')\nnew_op = os.path.splitext(op)[0] + '.md'\nprint(f'new_pp={new_pp}')\nprint(f'names_match={new_pp.name == os.path.basename(new_op)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "op_base=file.txt" in out1
        assert "pp_name=file.txt" in out1
        out2 = nb_runner.get_output(2)
        assert "ext_match=True" in out2
        out3 = nb_runner.get_output(3)
        assert "names_match=True" in out3

    def test_path_edit_propagation(self, nb_runner):
        nb_runner.create_notebook([
            "import os\nfrom pathlib import PurePosixPath\nop = os.path.join('home', 'user', 'docs', 'file.txt')\npp = PurePosixPath('home') / 'user' / 'docs' / 'file.txt'\nprint(f'pp_name={pp.name}')",
            "parts_count = len(pp.parts)\nprint(f'parts={parts_count}')",
            "depth = op.count(os.sep) + op.count('/')\nprint(f'depth={depth}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parts=4" in nb_runner.get_output(2)

        # Change base path depth
        nb_runner.set_cell_source(1, "import os\nfrom pathlib import PurePosixPath\nop = os.path.join('srv', 'data', 'output.csv')\npp = PurePosixPath('srv') / 'data' / 'output.csv'\nprint(f'pp_name={pp.name}')")
        nb_runner.run_cells([1, 2, 3])
        assert "pp_name=output.csv" in nb_runner.get_output(1)
        assert "parts=3" in nb_runner.get_output(2)

    def test_path_cache_correctness(self, nb_runner):
        nb_runner.create_notebook([
            "import os\nfrom pathlib import PurePosixPath\npp = PurePosixPath('a') / 'b' / 'c.txt'\nprint(f'stem={pp.stem}')",
            "result = pp.with_name('d.csv')\nprint(f'new_name={result.name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stem=c" in nb_runner.get_output(1)
        assert "new_name=d.csv" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "new_name=d.csv" in nb_runner.get_output(2)
