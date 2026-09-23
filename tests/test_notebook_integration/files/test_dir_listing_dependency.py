"""A directory listing (glob/os.listdir) is a tracked dependency.

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
    (gdir / "d1.num").write_text("1", encoding="utf-8")
    (gdir / "d2.num").write_text("2", encoding="utf-8")
    gp = str(gdir).replace("\\", "/")
    nb_runner.create_notebook(
        [f"import glob\nvals = [int(open(fp).read()) for fp in sorted(glob.glob('{gp}/*.num'))]\nprint('vals =', vals)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "vals = [1, 2]" in nb_runner.get_output(1)

    time.sleep(1.1)  # rule out mtime-granularity timing
    (gdir / "d3.num").write_text("30", encoding="utf-8")
    nb_runner.run_all()
    assert "vals = [1, 2, 30]" in nb_runner.get_output(1), nb_runner.get_output(1)


def test_unchanged_globbed_dir_stays_cached(nb_runner, tmp_path):
    gdir = tmp_path / "gdir2"
    gdir.mkdir()
    (gdir / "a.num").write_text("5", encoding="utf-8")
    (gdir / "b.num").write_text("6", encoding="utf-8")
    gp = str(gdir).replace("\\", "/")
    nb_runner.create_notebook(
        [f"import glob\nvals = [int(open(fp).read()) for fp in sorted(glob.glob('{gp}/*.num'))]\nprint('vals =', vals)"]
    )
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


@pytest.mark.parametrize("listing", ["glob('*.num')", "rglob('*.num')", "iterdir()"])
def test_new_file_in_pathlib_listed_dir_invalidates(nb_runner, tmp_path, listing):
    """``Path.glob`` lists through a captured ``os.scandir`` on 3.13
    (and pathlib's accessor on 3.10) -- a new month's file was never seen."""
    pdir = tmp_path / "pdir"
    pdir.mkdir()
    (pdir / "d1.num").write_text("1", encoding="utf-8")
    (pdir / "d2.num").write_text("2", encoding="utf-8")
    pp = str(pdir).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "from pathlib import Path\n"
            f"vals = sorted(int(p.read_text()) for p in Path('{pp}').{listing})\n"
            "print('vals =', vals)"
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "vals = [1, 2]" in nb_runner.get_output(1)

    time.sleep(1.1)
    (pdir / "d3.num").write_text("30", encoding="utf-8")
    nb_runner.run_all()
    assert "vals = [1, 2, 30]" in nb_runner.get_output(1), nb_runner.get_output(1)


def test_os_listdir_new_file_invalidates(nb_runner, tmp_path):
    ldir = tmp_path / "ldir"
    ldir.mkdir()
    (ldir / "x.txt").write_text("x", encoding="utf-8")
    lp = str(ldir).replace("\\", "/")
    nb_runner.create_notebook([f"import os\nnames = sorted(os.listdir('{lp}'))\nprint('names =', names)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "names = ['x.txt']" in nb_runner.get_output(1)

    time.sleep(1.1)
    (ldir / "y.txt").write_text("y", encoding="utf-8")
    nb_runner.run_all()
    assert "names = ['x.txt', 'y.txt']" in nb_runner.get_output(1), nb_runner.get_output(1)
