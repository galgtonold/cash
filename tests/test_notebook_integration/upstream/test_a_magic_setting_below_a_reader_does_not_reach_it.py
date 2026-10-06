"""An environment variable or folder a magic sets below a reader does not reach the reader.

``%env MODE=a`` | ``x = os.environ["MODE"] * 2`` | ``%env MODE=b`` |
``print(x)``: a cell of magics alone runs in IPython, not through cash, and a
magic among statements runs as ``get_ipython().run_line_magic(...)``. Neither
was noted as the notebook's own change, so the upstream check of the last
cell ran ``x`` again with ``"b"``: ``"bb"`` where a plain kernel prints
``"aa"``. The same for ``%cd``. And ``ROOT = os.getcwd()`` |
``os.chdir(os.path.join(ROOT, "a"))``, written with double quotes, then
edited to ``"b"`` and run: ``ROOT`` was run again inside ``a/`` and the
edited cell failed.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]


@pytest.fixture
def var(tmp_path):
    """A name of its own: the warm kernel keeps its environment between tests."""
    return f"CASH_TEST_MAGIC_{abs(hash(str(tmp_path))) % 10**8}"


@pytest.mark.parametrize("alone", [True, False], ids=["magic_cell", "magic_among_statements"])
def test_env_magic_below_the_reader(nb_runner, var, alone):
    rest = "" if alone else "\nz = 1"
    nb_runner.create_notebook(
        ["import os", f"%env {var}=a{rest}", f"x = os.environ['{var}'] * 2", f"%env {var}=b{rest}", "print('X', x)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X aa" in nb_runner.get_output(5), nb_runner.get_raw_output(5)


def test_cd_magic_below_the_reader(nb_runner, tmp_path):
    (tmp_path / "sub").mkdir()
    nb_runner.create_notebook(
        ["import os", "here = os.path.basename(os.getcwd())", "%cd sub", "print('IN SUB', here == 'sub')"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "IN SUB False" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_editing_a_chdir_written_with_double_quotes(nb_runner):
    nb_runner.create_notebook(
        [
            'import os\nROOT = os.getcwd()\nfor d in "ab":\n    os.makedirs(d, exist_ok=True)\n'
            '    open(f"{d}/cfg.txt", "w").write(d + "-dir")',
            'os.chdir(os.path.join(ROOT, "a"))',
            'x = open("cfg.txt").read()\nprint("X", x)',
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X a-dir" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.set_cell_source(2, 'os.chdir(os.path.join(ROOT, "b"))')
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "X b-dir" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
