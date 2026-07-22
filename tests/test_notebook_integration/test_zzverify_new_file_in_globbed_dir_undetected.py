"""Minimal verifier for probe finding 'new-file-in-globbed-dir-undetected'.

A cell that enumerates a directory (glob.glob) and reads every matched file
is cached with file-deps covering only the files READ on the first run.
When a NEW matching file appears in the directory, no tracked file changed,
so run_all serves the stale list — the directory listing itself is not a
tracked dependency.
"""

import time

import pytest

pytestmark = [pytest.mark.timeout(90)]


def test_new_file_in_globbed_dir_invalidates(nb_runner, tmp_path):
    gdir = tmp_path / "gdir"
    gdir.mkdir()
    (gdir / "d1.num").write_text("1")
    (gdir / "d2.num").write_text("2")
    gp = str(gdir).replace("\\", "/")
    nb_runner.create_notebook([
        (
            "import glob\n"
            f"vals = [int(open(fp).read()) for fp in sorted(glob.glob('{gp}/*.num'))]\n"
            "print('vals =', vals)"
        ),
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [1, 2]" in nb_runner.get_output(1)

    # Generous gap rules out any CAS-10-style mtime-granularity timing excuse.
    time.sleep(1.1)
    (gdir / "d3.num").write_text("30")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    raw = nb_runner.get_raw_output(1)
    assert "vals = [1, 2, 30]" in out, (
        f"NEW file in the globbed directory was NOT detected - the directory "
        f"listing is not a tracked dependency, so the cell served the stale "
        f"list. Got: {out!r}\nRaw (last 400): {raw[-400:]!r}"
    )
