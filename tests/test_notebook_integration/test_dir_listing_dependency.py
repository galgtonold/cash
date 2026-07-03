"""CAS-85: a directory listing (glob/os.listdir) is a tracked dependency.

A cell that enumerates a directory and reads the matches used to get file-deps
only for the files READ on the first run, so a NEW matching file was invisible
even to run_all. Tracking the enumerated directory (its mtime bumps on
add/remove) invalidates the reader when membership changes, while an unchanged
directory keeps the cache hit.
"""
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]


def test_new_file_in_globbed_dir_invalidates(nb_runner, tmp_path):
    gdir = tmp_path / "gdir"
    gdir.mkdir()
    (gdir / "d1.num").write_text("1")
    (gdir / "d2.num").write_text("2")
    gp = str(gdir).replace("\\", "/")
    nb_runner.create_notebook([
        "import glob\n"
        f"vals = [int(open(fp).read()) for fp in sorted(glob.glob('{gp}/*.num'))]\n"
        "print('vals =', vals)"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "vals = [1, 2]" in nb_runner.get_output(1)

    time.sleep(1.1)  # rule out mtime-granularity timing (CAS-10)
    (gdir / "d3.num").write_text("30")
    nb_runner.run_all()
    assert "vals = [1, 2, 30]" in nb_runner.get_output(1), nb_runner.get_output(1)


def test_unchanged_globbed_dir_stays_cached(nb_runner, tmp_path):
    gdir = tmp_path / "gdir2"
    gdir.mkdir()
    (gdir / "a.num").write_text("5")
    (gdir / "b.num").write_text("6")
    gp = str(gdir).replace("\\", "/")
    nb_runner.create_notebook([
        "import glob\n"
        f"vals = [int(open(fp).read()) for fp in sorted(glob.glob('{gp}/*.num'))]\n"
        "print('vals =', vals)"
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [5, 6]" in nb_runner.get_output(1)

    # No membership change → the second run must restore from cache, not recompute.
    time.sleep(1.1)
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(1)
    assert "vals = [5, 6]" in nb_runner.get_output(1)
    assert "RESTORED" in raw or "CACHE_HIT" in raw or "cache" in raw.lower(), (
        f"unchanged directory should have restored from cache; raw: {raw[-300:]!r}"
    )


def test_os_listdir_new_file_invalidates(nb_runner, tmp_path):
    ldir = tmp_path / "ldir"
    ldir.mkdir()
    (ldir / "x.txt").write_text("x")
    lp = str(ldir).replace("\\", "/")
    nb_runner.create_notebook([
        "import os\n"
        f"names = sorted(os.listdir('{lp}'))\n"
        "print('names =', names)"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "names = ['x.txt']" in nb_runner.get_output(1)

    time.sleep(1.1)
    (ldir / "y.txt").write_text("y")
    nb_runner.run_all()
    assert "names = ['x.txt', 'y.txt']" in nb_runner.get_output(1), nb_runner.get_output(1)
