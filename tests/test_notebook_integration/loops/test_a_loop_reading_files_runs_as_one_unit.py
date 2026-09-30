"""A long loop that reads a folder of files runs as ONE cached unit.

`for f in files: d = pd.read_csv(f); ...` over hundreds of files used to be
refused the single-unit fast path because its body calls a file reader, so
each iteration's statements went through the per-statement machinery: about
3.5 ms per statement per iteration, 11 s cold and 4.6 s warm for a loop that
takes 1.8 s with cash off.

A single unit runs through the same statement path as a comprehension
(`[pd.read_csv(f) for f in files]`): one file tracker around the whole loop,
so the unit's entry depends on every file it read and on every folder it
listed. These tests pin that the entry follows those files: an edit, a new
file in the globbed folder and a deleted file each give a miss and the
result a plain kernel gives. A loop that WRITES, moves or removes files keeps
the per-iteration path, which records each write where it happens.
"""

import os
import time

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.loops, pytest.mark.files, pytest.mark.timeout(120)]

# Enough iterations and statements to clear the single-unit thresholds:
# more than 50 iterations, and 60 x 3 x 8 ms = 1.4 s estimated overhead.
N_FILES = 60
SINGLE_UNIT = "Fast-loop: executing as single unit"
UNIT_RESTORED = "[CONTROL] Completed: 1 iterations, 1 cached, 0 computed"
UNIT_COMPUTED = "[CONTROL] Completed: 1 iterations, 0 cached, 1 computed"


def _make_folder(tmp_path, n=N_FILES):
    folder = tmp_path / "inputs"
    folder.mkdir()
    day_ago = time.time() - 86_400
    for i in range(n):
        p = folder / f"part_{i:03d}.num"
        p.write_text(str(i), encoding="utf-8")
        # Aged, like real exports: cash treats a file written moments ago
        # differently from one untouched for a day.
        os.utime(p, (day_ago, day_ago))
    return folder


def _loop_cell(folder):
    gp = str(folder).replace("\\", "/")
    return (
        "import glob\n"
        f"files = sorted(glob.glob('{gp}/*.num'))\n"
        "total = 0\n"
        "for f in files:\n"
        "    text = open(f).read()\n"
        "    value = int(text)\n"
        "    total = total + value\n"
        "print('total =', total, 'from', len(files))"
    )


def _expected(n=N_FILES):
    return sum(range(n))


def _start(nb_runner, cells):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()


def _assert_rerun_as_one_unit(nb_runner, cell=1):
    """The re-run took the single-unit path and computed the loop again."""
    raw = nb_runner.get_raw_output(cell)
    assert SINGLE_UNIT in raw, "the loop should still run as one unit"
    assert UNIT_COMPUTED in raw, "a changed input file should be a miss for the unit"


def _bump(path, text):
    """Rewrite *path* so both its content and its mtime move."""
    path.write_text(text, encoding="utf-8")
    later = time.time() + 5
    os.utime(path, (later, later))


def test_a_file_reading_loop_runs_as_one_unit_and_hits(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    _start(nb_runner, [_loop_cell(folder)])
    raw = nb_runner.get_raw_output(1)
    assert f"total = {_expected()} from {N_FILES}" in nb_runner.get_output(1)
    assert SINGLE_UNIT in raw, "a loop that only reads files should run as one unit"

    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected()} from {N_FILES}" in out
    raw = nb_runner.get_raw_output(1)
    assert SINGLE_UNIT in raw
    assert UNIT_RESTORED in raw, "an unchanged folder should restore the loop"


def test_editing_one_file_reruns_the_loop(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    _start(nb_runner, [_loop_cell(folder)])
    assert SINGLE_UNIT in nb_runner.get_raw_output(1)

    _bump(folder / "part_007.num", "1007")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected() + 1000} from {N_FILES}" in out, out
    _assert_rerun_as_one_unit(nb_runner)


def test_a_same_size_edit_reruns_the_loop(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    _start(nb_runner, [_loop_cell(folder)])

    # "13" -> "93": the same size, so only the content and mtime tell.
    _bump(folder / "part_013.num", "93")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected() + 80} from {N_FILES}" in out, out
    _assert_rerun_as_one_unit(nb_runner)


def test_a_new_file_in_the_globbed_folder_reruns_the_loop(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    _start(nb_runner, [_loop_cell(folder)])
    assert f"total = {_expected()} from {N_FILES}" in nb_runner.get_output(1)

    time.sleep(1.1)  # rule out folder-mtime granularity
    (folder / "part_999.num").write_text("500", encoding="utf-8")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected() + 500} from {N_FILES + 1}" in out, out
    _assert_rerun_as_one_unit(nb_runner)


def test_a_deleted_file_reruns_the_loop(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    _start(nb_runner, [_loop_cell(folder)])

    time.sleep(1.1)
    (folder / "part_010.num").unlink()
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected() - 10} from {N_FILES - 1}" in out, out
    _assert_rerun_as_one_unit(nb_runner)


def test_a_file_deleted_from_a_fixed_list_raises_again(nb_runner, tmp_path):
    """The file list does not move (it is a literal), so only the unit's own
    file dependency can tell the file is gone: the loop must run again and
    raise, not restore the old total."""
    folder = _make_folder(tmp_path)
    paths = [str(folder / f"part_{i:03d}.num").replace("\\", "/") for i in range(N_FILES)]
    nb_runner.create_notebook(
        [
            f"files = {paths!r}",
            "total = 0\nfor f in files:\n    text = open(f).read()\n    value = int(text)\n    total = total + value\n"
            "print('total =', total)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert f"total = {_expected()}" in nb_runner.get_output(2)

    (folder / "part_020.num").unlink()
    nb_runner.run_cell(1)
    with pytest.raises(CellExecutionError) as err:
        nb_runner.run_cell(2)
    assert err.value.ename == "FileNotFoundError", err.value.ename
    assert f"total = {_expected()}" not in nb_runner.get_output(2)
    assert SINGLE_UNIT in nb_runner.get_raw_output(2)


@pytest.mark.fresh_kernel
def test_a_restart_restores_the_loop_and_still_sees_an_edit(nb_runner, tmp_path):
    folder = _make_folder(tmp_path)
    nb_runner.create_notebook([_loop_cell(folder)])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert f"total = {_expected()} from {N_FILES}" in nb_runner.get_output(1)

    nb_runner.restart()
    nb_runner._init_cash()  # a restart leaves a kernel without cash
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert f"total = {_expected()} from {N_FILES}" in nb_runner.get_output(1)
    assert UNIT_RESTORED in nb_runner.get_raw_output(1), "the unit should come back from disk"

    _bump(folder / "part_030.num", "3030")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected() + 3000} from {N_FILES}" in out, out
    _assert_rerun_as_one_unit(nb_runner)


def test_a_frame_built_from_the_loop_follows_an_edited_file(nb_runner, tmp_path):
    """The benchmark shape: `parts.append(d)` in the loop, `pd.concat` after.
    The loop mutates `parts` in place, so the unit itself is not cached; the
    statement that concatenates must still see an edited file, not restore
    the old frame."""
    folder = tmp_path / "csvs"
    folder.mkdir()
    day_ago = time.time() - 86_400
    for i in range(N_FILES):
        p = folder / f"day_{i:03d}.csv"
        p.write_text(f"qty\n{i}\n", encoding="utf-8")
        os.utime(p, (day_ago, day_ago))
    gp = str(folder).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import glob, os\nimport pandas as pd\n"
            f"files = sorted(glob.glob('{gp}/*.csv'))\n"
            "parts = []\n"
            "for f in files:\n"
            "    d = pd.read_csv(f)\n"
            "    d['source_file'] = os.path.basename(f)\n"
            "    parts.append(d)\n"
            "raw = pd.concat(parts, ignore_index=True)",
            "print('qty =', int(raw['qty'].sum()), 'rows =', len(raw))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert SINGLE_UNIT in nb_runner.get_raw_output(1)
    assert f"qty = {_expected()} rows = {N_FILES}" in nb_runner.get_output(2)

    _bump(folder / "day_005.csv", "qty\n1005\n")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert f"qty = {_expected() + 1000} rows = {N_FILES}" in out, out


def test_a_file_read_through_a_helper_is_a_dependency_of_the_unit(nb_runner, tmp_path):
    """No reader call is written in the loop body: the helper opens the file.
    The unit's file tracker still sees the read, so an edit is a miss."""
    folder = _make_folder(tmp_path)
    gp = str(folder).replace("\\", "/")
    _start(
        nb_runner,
        [
            "def load(path):\n    with open(path) as fh:\n        return int(fh.read())",
            "import glob\n"
            f"files = sorted(glob.glob('{gp}/*.num'))\n"
            "total = 0\n"
            "for f in files:\n"
            "    value = load(f)\n"
            "    doubled = value * 2\n"
            "    total = total + doubled\n"
            "print('total =', total)",
        ],
    )
    assert f"total = {2 * _expected()}" in nb_runner.get_output(2)

    _bump(folder / "part_050.num", "150")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert f"total = {2 * (_expected() + 100)}" in out, out
    _assert_rerun_as_one_unit(nb_runner, cell=2)


def test_a_folder_listed_in_the_body_is_a_dependency_of_the_unit(nb_runner, tmp_path):
    """The body lists a folder per iteration: a file added to one of them
    changes no path the loop opened, only a listing, and is still a miss."""
    root = tmp_path / "days"
    root.mkdir()
    for i in range(N_FILES):
        (root / f"d{i:03d}").mkdir()
        (root / f"d{i:03d}" / "a.txt").write_text("x", encoding="utf-8")
    rp = str(root).replace("\\", "/")
    _start(
        nb_runner,
        [
            "import os\n"
            f"days = sorted(os.listdir('{rp}'))\n"
            "count = 0\n"
            "for day in days:\n"
            f"    names = os.listdir('{rp}/' + day)\n"
            f"    first = open('{rp}/' + day + '/a.txt').read()\n"
            "    count = count + len(names)\n"
            "print('count =', count)",
        ],
    )
    assert f"count = {N_FILES}" in nb_runner.get_output(1)

    time.sleep(1.1)  # rule out folder-mtime granularity
    (root / "d017" / "b.txt").write_text("y", encoding="utf-8")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"count = {N_FILES + 1}" in out, out
    _assert_rerun_as_one_unit(nb_runner)


def test_a_loop_stopped_by_a_bad_file_runs_again_once_it_is_fixed(nb_runner, tmp_path):
    """An error part-way through is not cached: the unit stores nothing, and
    after the bad file is fixed the whole loop runs and gives the full total."""
    folder = _make_folder(tmp_path)
    (folder / "part_040.num").write_text("not a number", encoding="utf-8")
    nb_runner.create_notebook([_loop_cell(folder)])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    with pytest.raises(CellExecutionError) as err:
        nb_runner.run_all()
    assert err.value.ename == "ValueError", err.value.ename
    assert SINGLE_UNIT in nb_runner.get_raw_output(1)

    _bump(folder / "part_040.num", "40")
    nb_runner.run_all()
    out = nb_runner.get_output(1)
    assert f"total = {_expected()} from {N_FILES}" in out, out
    _assert_rerun_as_one_unit(nb_runner)
