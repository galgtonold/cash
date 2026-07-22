"""Adversarial probes: file I/O dependency tracking.

Attack surface probed (one test per distinct mechanism):

1.  writer-cell edit -> run_all: reader cell must see the newly written file
    (open('w') write channel, open('r') read channel).
2.  write + read in the SAME cell: edit the written payload -> re-run must echo
    the new payload; unchanged re-runs stay idempotent.
3.  pandas to_csv writer edit -> ISOLATED run of a downstream cell only:
    upstream simulation must re-execute the writer and the reader must see the
    new file (pre-write-mtime staleness decision attack).
4.  pathlib Path.read_text channel: external modification detected?
5.  numpy np.load channel: external .npy rewrite detected?
6.  file DELETED between runs: graceful FileNotFoundError, not stale restore.
7.  touch-only (mtime bumped, content+size identical): should stay cached
    (effectiveness candidate if it recomputes).
8.  os.chdir + relative path: after pointing chdir at a directory holding a
    DIFFERENT data.csv, the reader must not serve the old directory's data.
9.  glob comprehension over many files: (a) one file rewritten -> new values;
    (b) a NEW file appears in the globbed dir -> new values (directory listing
    as an untracked dependency).
10. two cells read the same file: external modification refreshes both.
11. StringIO read_csv (no path): upstream string edit invalidates via lineage.
12. pickle round-trip writer edit -> isolated downstream run sees new payload.
"""

import os
import time

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(90)]


def _p(path) -> str:
    """Forward-slash path string for embedding in cell source."""
    return str(path).replace("\\", "/")


_HIT_MARKERS = ("[CACHE_HIT_DEBUG]", "Cache hit", "RESTORED", "Restored", "SKIPPED", "Skipped")


# ---------------------------------------------------------------------------
# 1. writer cell edited -> run_all -> reader must see new bytes
# ---------------------------------------------------------------------------

def test_writer_edit_run_all_reader_fresh(nb_runner, tmp_path):
    data = tmp_path / "data.txt"
    p = _p(data)
    writer_v1 = (
        "rows = 'a\\n1\\n2\\n'\n"
        f"with open('{p}', 'w') as f:\n"
        "    f.write(rows)\n"
        "print('wrote', len(rows))"
    )
    reader = (
        f"with open('{p}') as f:\n"
        "    body = f.read()\n"
        "print('body =', body.strip().replace('\\n', '|'))"
    )
    nb_runner.create_notebook([writer_v1, reader])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "body = a|1|2" in nb_runner.get_output(2)

    # Edit the writer: different rows, DIFFERENT byte size (avoids CAS-10).
    writer_v2 = (
        "rows = 'a\\n7\\n8\\n9\\n'\n"
        f"with open('{p}', 'w') as f:\n"
        "    f.write(rows)\n"
        "print('wrote', len(rows))"
    )
    nb_runner.set_cell_source(1, writer_v2)
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "body = a|7|8|9" in out, (
        f"Reader cell served STALE file content after the writer cell was "
        f"edited and re-run in the same run_all. Got: {out!r}"
    )


# ---------------------------------------------------------------------------
# 2. write + read in the SAME cell
# ---------------------------------------------------------------------------

def test_write_and_read_same_cell(nb_runner, tmp_path):
    data = tmp_path / "roundtrip.txt"
    p = _p(data)
    cell_v1 = (
        "payload = 'x=1'\n"
        f"with open('{p}', 'w') as f:\n"
        "    f.write(payload)\n"
        f"with open('{p}') as f:\n"
        "    echo = f.read()\n"
        "print('echo =', echo)"
    )
    nb_runner.create_notebook([cell_v1, "print('len =', len(echo))"])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "echo = x=1" in nb_runner.get_output(1)
    assert "len = 3" in nb_runner.get_output(2)

    # Unchanged re-run: idempotent, and the cell must not be flagged as changed.
    nb_runner.run_all()
    assert "echo = x=1" in nb_runner.get_output(1)
    raw1 = nb_runner.get_raw_output(1)
    assert "[CELL_CHANGED]" not in raw1

    # Edit the payload (different size) -> fresh echo.
    cell_v2 = cell_v1.replace("'x=1'", "'x=42-longer'")
    nb_runner.set_cell_source(1, cell_v2)
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert "echo = x=42-longer" in out, (
        f"Same-cell write+read served stale echo after payload edit. Got: {out!r}"
    )
    assert "len = 11" in nb_runner.get_output(2)


# ---------------------------------------------------------------------------
# 3. to_csv writer edited -> run ONLY the last cell (isolated downstream run)
# ---------------------------------------------------------------------------

def test_tocsv_writer_edit_isolated_downstream_run(nb_runner, tmp_path):
    csv = tmp_path / "made.csv"
    p = _p(csv)
    writer_v1 = (
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [1, 2, 3]}}).to_csv('{p}', index=False)\n"
        "print('wrote v1')"
    )
    reader = (
        "import pandas as pd\n"
        f"df = pd.read_csv('{p}')\n"
        "print('vals =', df['v'].tolist())"
    )
    consumer = "print('total =', int(df['v'].sum()))"
    nb_runner.create_notebook([writer_v1, reader, consumer])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "total = 6" in nb_runner.get_output(3)

    # Edit the writer (different rows AND row count -> different size).
    writer_v2 = (
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [10, 20, 30, 40]}}).to_csv('{p}', index=False)\n"
        "print('wrote v2')"
    )
    nb_runner.set_cell_source(1, writer_v2)

    # Isolated run of ONLY the downstream consumer: upstream simulation must
    # re-execute the edited writer, and the reader's file-dep check must see
    # the POST-write mtime, not the pre-write one.
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "total = 100" in out, (
        f"Isolated downstream run after editing the to_csv writer cell served "
        f"a stale total (writer not re-executed, or reader's file-dep check "
        f"used the pre-write mtime). Got: {out!r}"
    )


# ---------------------------------------------------------------------------
# 3a. BOUNDARY: writer edited -> isolated re-run of the READER cell itself
# ---------------------------------------------------------------------------

def test_writer_edit_isolated_rerun_of_reader_itself(nb_runner, tmp_path):
    csv = tmp_path / "made3.csv"
    p = _p(csv)
    writer_v1 = (
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [1, 2, 3]}}).to_csv('{p}', index=False)\n"
        "print('wrote v1')"
    )
    reader = (
        "import pandas as pd\n"
        f"df = pd.read_csv('{p}')\n"
        "print('vals =', df['v'].tolist())"
    )
    nb_runner.create_notebook([writer_v1, reader])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "vals = [1, 2, 3]" in nb_runner.get_output(2)

    writer_v2 = (
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [10, 20, 30, 40]}}).to_csv('{p}', index=False)\n"
        "print('wrote v2')"
    )
    nb_runner.set_cell_source(1, writer_v2)
    # Re-run the READER cell directly (what a user does after editing the
    # writer, if they only care about the read result).
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "vals = [10, 20, 30, 40]" in out, (
        f"Isolated re-run of the READER cell after editing the writer cell "
        f"served stale file content. Got: {out!r}"
    )


# ---------------------------------------------------------------------------
# 3c. BOUNDARY: consumer needs BOTH the writer's var and the reader's var, so
#     upstream sim MUST re-execute the edited writer. Does the reader then see
#     the post-write file, or was its staleness decided on the pre-write mtime?
# ---------------------------------------------------------------------------

def test_writer_reexecuted_midrun_reader_must_see_new_file(nb_runner, tmp_path):
    pkl = tmp_path / "both.pkl"
    p = _p(pkl)
    writer_v1 = (
        "import pickle\n"
        "payload = {'nums': [1, 2, 3]}\n"
        f"with open('{p}', 'wb') as f:\n"
        "    pickle.dump(payload, f)\n"
        "print('dumped', len(payload['nums']))"
    )
    reader = (
        "import pickle\n"
        f"with open('{p}', 'rb') as f:\n"
        "    loaded = pickle.load(f)\n"
        "print('loaded =', loaded['nums'])"
    )
    # Consumer references BOTH payload (writer var) and loaded (reader var):
    # the upstream simulation is forced to re-execute the edited writer.
    consumer = "print('combo =', sum(loaded['nums']), len(payload['nums']))"
    nb_runner.create_notebook([writer_v1, reader, consumer])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "combo = 6 3" in nb_runner.get_output(3)

    writer_v2 = writer_v1.replace("[1, 2, 3]", "[10, 20, 30, 40]")
    nb_runner.set_cell_source(1, writer_v2)
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "combo = 100 4" in out, (
        f"Writer WAS re-executed during upstream sim (payload needed), but the "
        f"reader's file-dep staleness was decided against the PRE-write state "
        f"-> stale 'loaded'. Got: {out!r}"
    )


# ---------------------------------------------------------------------------
# 3b. BOUNDARY: same to_csv writer edit, but via run_all (does the full-run
#     path recover where the isolated-downstream-run path serves stale?)
# ---------------------------------------------------------------------------

def test_tocsv_writer_edit_run_all_boundary(nb_runner, tmp_path):
    csv = tmp_path / "made2.csv"
    p = _p(csv)
    writer_v1 = (
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [1, 2, 3]}}).to_csv('{p}', index=False)\n"
        "print('wrote v1')"
    )
    reader = (
        "import pandas as pd\n"
        f"df = pd.read_csv('{p}')\n"
        "print('vals =', df['v'].tolist())"
    )
    consumer = "print('total =', int(df['v'].sum()))"
    nb_runner.create_notebook([writer_v1, reader, consumer])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "total = 6" in nb_runner.get_output(3)

    writer_v2 = (
        "import pandas as pd\n"
        f"pd.DataFrame({{'v': [10, 20, 30, 40]}}).to_csv('{p}', index=False)\n"
        "print('wrote v2')"
    )
    nb_runner.set_cell_source(1, writer_v2)
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "total = 100" in out, (
        f"Even run_all after editing the to_csv writer served a stale total. "
        f"Got: {out!r}"
    )


# ---------------------------------------------------------------------------
# 4. pathlib read_text channel
# ---------------------------------------------------------------------------

def test_pathlib_read_text_external_modification(nb_runner, tmp_path):
    data = tmp_path / "plib.txt"
    data.write_text("alpha")
    p = _p(data)
    nb_runner.create_notebook([
        f"from pathlib import Path\ntxt = Path('{p}').read_text()",
        "print('txt =', txt)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "txt = alpha" in nb_runner.get_output(2)

    time.sleep(1.1)
    data.write_text("bravo-longer")  # different size
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "txt = bravo-longer" in out, (
        f"pathlib Path.read_text() modification NOT detected (pathlib read "
        f"channel untracked?). Got: {out!r}"
    )


# ---------------------------------------------------------------------------
# 5. numpy np.load channel
# ---------------------------------------------------------------------------

def test_numpy_load_external_npy_rewrite(nb_runner, tmp_path):
    import numpy as np
    npy = tmp_path / "arr.npy"
    np.save(npy, np.array([1, 2, 3]))
    p = _p(npy)
    nb_runner.create_notebook([
        f"import numpy as np\narr = np.load('{p}')\nprint('n =', arr.shape[0])",
        "print('s =', int(arr.sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "n = 3" in nb_runner.get_output(1)
    assert "s = 6" in nb_runner.get_output(2)

    time.sleep(1.1)
    np.save(npy, np.array([10, 20, 30, 40]))  # different length -> different size
    nb_runner.run_all()
    out1 = nb_runner.get_output(1)
    out2 = nb_runner.get_output(2)
    assert "n = 4" in out1 and "s = 100" in out2, (
        f"np.load .npy rewrite NOT detected. Got cell1: {out1!r}, cell2: {out2!r}"
    )


# ---------------------------------------------------------------------------
# 6. file deleted between runs -> graceful error, not stale restore
# ---------------------------------------------------------------------------

def test_deleted_file_errors_not_stale(nb_runner, tmp_path):
    data = tmp_path / "gone.txt"
    data.write_text("still-here")
    p = _p(data)
    nb_runner.create_notebook([
        f"txt = open('{p}').read()\nprint('txt =', txt)",
        "print('len =', len(txt))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "txt = still-here" in nb_runner.get_output(1)

    os.remove(data)
    errored = False
    err_text = ""
    try:
        nb_runner.run_all()
    except CellExecutionError as e:
        errored = True
        err_text = str(e)

    if errored:
        # The cell must carry the natural FileNotFoundError as its error
        # output (nbclient's CellExecutionError str() only embeds stdout, so
        # check the error outputs directly).
        enames = [
            o.get("ename")
            for o in nb_runner.get_cell(1).outputs
            if o.get("output_type") == "error"
        ]
        assert "FileNotFoundError" in enames, (
            f"Re-run after file deletion errored, but not with the natural "
            f"FileNotFoundError a fresh run raises. Error outputs: {enames}. "
            f"FULL ERROR (last 800):\n{err_text[-800:]}"
        )
    else:
        # No error: cash must NOT have silently served the stale content.
        out1 = nb_runner.get_output(1)
        raw1 = nb_runner.get_raw_output(1)
        assert "FileNotFoundError" in raw1 or "txt = still-here" not in out1, (
            f"File was DELETED but the reader cell silently served the stale "
            f"cached content instead of erroring. Got: {out1!r}"
        )


# ---------------------------------------------------------------------------
# 7. touch-only (mtime changed, content+size identical) -> should stay cached
# ---------------------------------------------------------------------------

def test_touch_only_mtime_should_stay_cached(nb_runner, tmp_path):
    data = tmp_path / "touched.txt"
    data.write_text("constant-content")
    p = _p(data)
    nb_runner.create_notebook([
        f"blob = open('{p}').read()",
        "print('blen =', len(blob))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "blen = 16" in nb_runner.get_output(2)

    # Sanity: an unchanged isolated re-run serves from cache (no fresh miss).
    nb_runner.run_cell(1)
    raw_sanity = nb_runner.get_raw_output(1)
    assert "Executing (cache miss)" not in raw_sanity, (
        f"HARNESS SANITY: unchanged re-run of the read cell was a cache miss; "
        f"marker idiom wrong or read statement uncacheable. Raw: {raw_sanity[:500]}"
    )

    # Bump ONLY the mtime; content and size unchanged.
    st = os.stat(data)
    os.utime(data, (st.st_atime + 30, st.st_mtime + 30))

    nb_runner.run_cell(1)
    raw2 = nb_runner.get_raw_output(1)
    assert (
        "Executing (cache miss)" not in raw2
        and "file changed" not in raw2
        and "File dependency mtime changed" not in raw2
    ), (
        f"EFFECTIVENESS: touch-only mtime bump (content+size identical) caused "
        f"a recompute; mtime-based invalidation with no content check. "
        f"Raw: {raw2[:600]}"
    )
    assert "blen = 16" in nb_runner.get_output(2)


# ---------------------------------------------------------------------------
# 8. chdir + relative path: different file with the same relative name
# ---------------------------------------------------------------------------

def test_chdir_relative_path_different_file(nb_runner, tmp_path):
    import pandas as pd
    dira = tmp_path / "dira"
    dirb = tmp_path / "dirb"
    dira.mkdir()
    dirb.mkdir()
    pd.DataFrame({"v": [1, 2, 3]}).to_csv(dira / "data.csv", index=False)
    pd.DataFrame({"v": [100, 200, 300, 400]}).to_csv(dirb / "data.csv", index=False)

    nb_runner.create_notebook([
        "import os\nimport pandas as pd",
        f"os.chdir(r'{_p(dira)}')",
        "df = pd.read_csv('data.csv')\nprint('vals =', df['v'].tolist())",
        "print('total =', int(df['v'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "total = 6" in nb_runner.get_output(4)

    # Point the chdir cell at dirb, which holds a DIFFERENT data.csv.
    nb_runner.set_cell_source(2, f"os.chdir(r'{_p(dirb)}')")
    nb_runner.run_all()
    out3 = nb_runner.get_output(3)
    out4 = nb_runner.get_output(4)
    assert "total = 1000" in out4 and "100" in out3, (
        f"After chdir to a directory with a DIFFERENT data.csv, the reader "
        f"served the OLD directory's data (path-collision stale cache). "
        f"Got cell3: {out3!r}, cell4: {out4!r}"
    )


# ---------------------------------------------------------------------------
# 9. glob comprehension: one file rewritten, then a NEW file appears
# ---------------------------------------------------------------------------

def test_glob_many_files_change_and_new_file(nb_runner, tmp_path):
    gdir = tmp_path / "gdir"
    gdir.mkdir()
    (gdir / "d1.num").write_text("1")
    (gdir / "d2.num").write_text("2")
    (gdir / "d3.num").write_text("3")
    gp = _p(gdir)
    nb_runner.create_notebook([
        (
            "import glob\n"
            f"vals = [int(open(fp).read()) for fp in sorted(glob.glob('{gp}/*.num'))]\n"
            "print('vals =', vals)"
        ),
        "print('total =', sum(vals))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "total = 6" in nb_runner.get_output(2)

    # Phase A: one existing file changes (content AND size).
    time.sleep(1.1)
    (gdir / "d2.num").write_text("222")
    nb_runner.run_all()
    out1 = nb_runner.get_output(1)
    out2 = nb_runner.get_output(2)
    assert "vals = [1, 222, 3]" in out1 and "total = 226" in out2, (
        f"Rewriting ONE of the globbed files was not detected. "
        f"Got cell1: {out1!r}, cell2: {out2!r}"
    )

    # Phase B: a NEW file appears in the globbed directory.
    time.sleep(1.1)
    (gdir / "d4.num").write_text("40")
    nb_runner.run_all()
    out1b = nb_runner.get_output(1)
    out2b = nb_runner.get_output(2)
    assert "vals = [1, 222, 3, 40]" in out1b and "total = 266" in out2b, (
        f"A NEW file appearing in the globbed directory was NOT detected — "
        f"directory listing is not a tracked dependency, cell served stale "
        f"list. Got cell1: {out1b!r}, cell2: {out2b!r}"
    )


# ---------------------------------------------------------------------------
# 10. two cells read the same file; external modification refreshes both
# ---------------------------------------------------------------------------

def test_two_cells_read_same_file_both_refresh(nb_runner, tmp_path):
    data = tmp_path / "shared.txt"
    data.write_text("aaa")
    p = _p(data)
    nb_runner.create_notebook([
        f"s1 = open('{p}').read()\nprint('s1 =', s1)",
        f"s2 = open('{p}').read()\nprint('s2 =', s2)",
        "print('combo =', s1, s2)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "combo = aaa aaa" in nb_runner.get_output(3)

    time.sleep(1.1)
    data.write_text("bbbbbb")  # different size
    nb_runner.run_all()
    out1 = nb_runner.get_output(1)
    out2 = nb_runner.get_output(2)
    out3 = nb_runner.get_output(3)
    assert "s1 = bbbbbb" in out1 and "s2 = bbbbbb" in out2 and "combo = bbbbbb bbbbbb" in out3, (
        f"One or both readers of the SAME modified file served stale content. "
        f"Got: {out1!r} / {out2!r} / {out3!r}"
    )


# ---------------------------------------------------------------------------
# 11. StringIO read_csv (no file path involved)
# ---------------------------------------------------------------------------

def test_stringio_read_csv_lineage(nb_runner):
    nb_runner.create_notebook([
        "csv_text = 'v\\n1\\n2\\n'",
        (
            "import pandas as pd\n"
            "from io import StringIO\n"
            "df = pd.read_csv(StringIO(csv_text))\n"
            "print('vals =', df['v'].tolist())"
        ),
        "print('total =', int(df['v'].sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "total = 3" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, "csv_text = 'v\\n5\\n6\\n7\\n'")
    nb_runner.run_all()
    out2 = nb_runner.get_output(2)
    out3 = nb_runner.get_output(3)
    assert "vals = [5, 6, 7]" in out2 and "total = 18" in out3, (
        f"StringIO-based read_csv did not refresh after upstream string edit. "
        f"Got: {out2!r} / {out3!r}"
    )


# ---------------------------------------------------------------------------
# 12. pickle round-trip: writer edited -> isolated run of last cell
# ---------------------------------------------------------------------------

def test_pickle_roundtrip_writer_edit_isolated_run(nb_runner, tmp_path):
    pkl = tmp_path / "payload.pkl"
    p = _p(pkl)
    writer_v1 = (
        "import pickle\n"
        "payload = {'nums': [1, 2, 3]}\n"
        f"with open('{p}', 'wb') as f:\n"
        "    pickle.dump(payload, f)\n"
        "print('dumped', len(payload['nums']))"
    )
    reader = (
        "import pickle\n"
        f"with open('{p}', 'rb') as f:\n"
        "    loaded = pickle.load(f)\n"
        "print('loaded =', loaded['nums'])"
    )
    consumer = "print('lsum =', sum(loaded['nums']))"
    nb_runner.create_notebook([writer_v1, reader, consumer])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "lsum = 6" in nb_runner.get_output(3)

    writer_v2 = writer_v1.replace("[1, 2, 3]", "[10, 20, 30, 40]")
    nb_runner.set_cell_source(1, writer_v2)
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "lsum = 100" in out, (
        f"Isolated downstream run after editing the pickle writer served a "
        f"stale sum. Got: {out!r}"
    )
