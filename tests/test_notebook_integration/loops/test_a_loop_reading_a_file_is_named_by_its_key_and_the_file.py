"""A loop run as one unit that reads a file names what it fills by its key and the file.

``for line in open(path): rows.append(parse(line))`` filled a list of 1.5M
dicts in 9.7 s, and reading the list back for its lineage took 7-9 s more:
a loop that read a file was named by its values. It is now named by its key
and the files it read, as a statement is: an edit to the file, even one
that keeps its size, is a new name, so a cell built on the list misses.
"""

import pytest

from .test_a_loop_filling_lists_is_named_by_its_inputs import COUNT, READ_BACK, SETUP, UNCOUNT

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LOOP = (
    "rows = []\n"
    "for line in open('log.txt'):\n"
    "    level, _, msg = line.rstrip('\\n').partition(' ')\n"
    "    rows.append({'level': level, 'msg': msg})"
)
REPORT = "print('ROWS', len(rows), sum('X' in r['msg'] for r in rows))"


def _log(marked: bool = False) -> str:
    return "".join(f"INFO request {'X' if marked and i == 7 else 'i'}{i:04d}\n" for i in range(2_000))


def test_the_rows_are_named_by_the_key_and_follow_the_file(nb_runner):
    log = nb_runner.work_dir / "log.txt"
    log.write_text(_log(), encoding="utf-8")
    nb_runner.create_notebook([SETUP, LOOP, REPORT])
    nb_runner.start_kernel()
    nb_runner.run_cells([1])
    nb_runner.peek(f"exec({COUNT!r})")
    try:
        nb_runner.run_cells([2, 3])
        assert "ROWS 2000 0" in nb_runner.get_output(3)
        assert nb_runner.peek(f"len({READ_BACK})") == "0"

        log.write_text(_log(marked=True), encoding="utf-8")  # the same size
        nb_runner.run_cells([2, 3])
        assert "ROWS 2000 1" in nb_runner.get_output(3)

        log.write_text(_log(), encoding="utf-8")
        nb_runner.run_cells([2, 3])
        assert "ROWS 2000 0" in nb_runner.get_output(3)
    finally:
        nb_runner.peek(f"exec({UNCOUNT!r})")


@pytest.mark.fresh_kernel
def test_after_a_restart_an_edited_file_is_read_again(nb_runner):
    log = nb_runner.work_dir / "log.txt"
    log.write_text(_log(), encoding="utf-8")
    nb_runner.create_notebook([SETUP, LOOP, REPORT])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])
    assert "ROWS 2000 0" in nb_runner.get_output(3)

    log.write_text(_log(marked=True), encoding="utf-8")
    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    assert "ROWS 2000 1" in nb_runner.get_output(3)

    log.write_text(_log(), encoding="utf-8")
    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    assert "ROWS 2000 0" in nb_runner.get_output(3)
