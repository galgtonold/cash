"""What is built from ``f = open(path)`` restores after a restart.

``f = open(path)`` then ``lines = f.readlines()``: the handle is a new object
on every run, so its lineage was new on every run, and every cell built on
``lines`` re-ran though the file had not changed (``with open(path) as f`` did
not have the problem). The handle is now named by its file and position, so
the parse below restores; an edit to the file, or reading from a different
position, still re-runs it, and a look at the handle is never served stale.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

SETUP = "import cash\n%cash_on\n%cash_badge print"
OPEN = "f = open('rows.txt')"
READ = "lines = f.readlines()"
PARSE = "import time\nparsed = [tuple(line.split(',')) for line in lines]\ntime.sleep(0.3)"
SHOW = "print('ROWS', len(parsed), parsed[-1])"


def _write(nb_runner, n):
    (nb_runner.work_dir / "rows.txt").write_text(
        "".join(f"{i},{i * 3}\n" for i in range(n)), encoding="utf-8", newline="\n"
    )


def _parse_line(output):
    return next(line for line in output.splitlines() if "parsed = " in line or "time.sleep" in line)


def test_the_parse_restores_after_a_restart_and_follows_an_edit(nb_runner):
    _write(nb_runner, 2000)
    nb_runner.create_notebook([SETUP, OPEN, READ, PARSE, SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "ROWS 2000 ('1999', '5997\\n')" in nb_runner.get_output(5)

    nb_runner.restart()
    nb_runner.run_all()
    assert "ROWS 2000 ('1999', '5997\\n')" in nb_runner.get_output(5)
    assert "CACHED" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert "COMPUTED: parsed" not in nb_runner.get_output(4)

    _write(nb_runner, 1500)
    nb_runner.run_all()
    assert "ROWS 1500 ('1499', '4497\\n')" in nb_runner.get_output(5), nb_runner.get_output(5)


def test_a_look_at_the_handle_is_never_served_stale(nb_runner):
    _write(nb_runner, 10)
    nb_runner.create_notebook(
        [SETUP, OPEN, "pos = f.tell()\nprint('POS', pos)", "first = f.readline()", "print('AT', f.tell())"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "POS 0" in nb_runner.get_output(3)
    assert "AT 4" in nb_runner.get_output(5)
    nb_runner.run_cell(3)
    assert "POS 4" in nb_runner.get_output(3), nb_runner.get_output(3)


@pytest.mark.parametrize(
    "opener, read",
    [
        ("f = open('data.csv')\nrd = f.readline", "header = rd()"),
        ("f = open('data.csv')\ndef rd(fh=f):\n    return fh.readline()", "header = rd()"),
        ("f = open('data.csv')\ndef rd():\n    return fh.readline()\nfh = f", "header = rd()"),
        ("files = {'data': open('data.csv')}\nf = files['data']", "header = files['data'].readline()"),
    ],
    ids=["bound-method", "default", "global", "dict-entry"],
)
def test_a_read_through_another_name_moves_the_file_on_every_run(nb_runner, opener, read):
    """A read through ``f.readline``, a function holding the handle, or a
    dict of handles moves the file on as ``f.readline()`` does: served from
    the cache, it left the next read starting at the header again."""
    (nb_runner.work_dir / "data.csv").write_text("id,v\n1,2\n3,4\n", encoding="utf-8", newline="\n")
    setup = SETUP + "\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)"
    nb_runner.create_notebook(
        [setup, opener, read + "\nprint('HEADER', repr(header))", "print('REST', repr(f.read()))"]
    )
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert "HEADER 'id,v\\n'" in nb_runner.get_output(3)
        assert "REST '1,2\\n3,4\\n'" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.restart()
    nb_runner.run_all()
    assert "REST '1,2\\n3,4\\n'" in nb_runner.get_output(4), nb_runner.get_output(4)
