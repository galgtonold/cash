"""CAS-84: os.chdir + relative-path reads must re-resolve against the live cwd.

A relative-path read (``pd.read_csv('data.csv')``) recorded only the realpath
resolved at first execution, frozen to that cwd. After editing an ``os.chdir``
cell to point at a different directory holding a DIFFERENT file with the same
relative name, even run_all served the old directory's data (the frozen realpath
still existed and was unmodified). Tracking the relative path too lets the
freshness check re-resolve it against the current cwd and catch the collision.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]


def test_chdir_relative_path_different_file_invalidates(nb_runner, tmp_path):
    dira = tmp_path / "dira"
    dirb = tmp_path / "dirb"
    dira.mkdir()
    dirb.mkdir()
    (dira / "data.csv").write_text("v\n1\n2\n3\n")
    (dirb / "data.csv").write_text("v\n100\n200\n300\n400\n")
    pa = str(dira).replace("\\", "/")
    pb = str(dirb).replace("\\", "/")

    nb_runner.create_notebook([
        "import os\nimport pandas as pd",
        f"os.chdir('{pa}')",
        "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(3)

    # Point the chdir cell at a DIFFERENT directory whose data.csv differs.
    nb_runner.set_cell_source(2, f"os.chdir('{pb}')")
    nb_runner.run_all()
    assert "vals = [100, 200, 300, 400]" in nb_runner.get_output(3), (
        f"reader served the OLD directory's data after chdir edit: "
        f"{nb_runner.get_output(3)!r}"
    )


def test_relative_read_same_cwd_stays_cached(nb_runner, tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "data.csv").write_text("v\n7\n8\n")
    p = str(d).replace("\\", "/")
    nb_runner.create_notebook([
        "import os\nimport pandas as pd",
        f"os.chdir('{p}')",
        "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [7, 8]" in nb_runner.get_output(3)

    # No chdir change, no file change → second run restores from cache.
    nb_runner.run_all()
    assert "vals = [7, 8]" in nb_runner.get_output(3)
