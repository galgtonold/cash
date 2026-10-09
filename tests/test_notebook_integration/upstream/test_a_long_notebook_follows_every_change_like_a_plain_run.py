"""A cell at the end of a long notebook gets what Restart & Run All gives,
whatever changed above it.

The upstream check before each cell remembers what the text of a cell or a
statement decides (its digests, whether it writes files, what it binds and
reads), so a cell far down a notebook costs no more to check than one near
the top. What depends on the kernel -- the values, the files, the
environment, the modules -- is still asked on every check. Each change
below is one that could make a remembered answer wrong: the result is
compared with the plain top-to-bottom value every time.
"""

import json
import textwrap
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.fresh_kernel, pytest.mark.timeout(600)]

#: Chained cells between the setup and the result.
N = 30


def _chain(i: int, step: int) -> str:
    return f"s{i} = s{i - 1} + {step}\nt{i} = [s{i}] * 2"


def _cells(tmp: str, scale: str, steps: dict[int, int], module: str = "helper") -> list[str]:
    cells = [
        f"import sys, os, json\nsys.path.insert(0, {tmp!r})\nimport {module} as helper\n"
        f"os.environ['CASH_TEST_LONG_SCALE'] = '{scale}'",
        f"with open({tmp + '/data.json'!r}) as f:\n    data = json.load(f)\ns0 = data['start']",
    ]
    cells += [_chain(i, steps.get(i, i)) for i in range(1, N + 1)]
    cells += [
        "scale = int(os.environ['CASH_TEST_LONG_SCALE'])",
        f"scaled = helper.bump(s{N}) * scale",
        f"print('R', scaled, t{N}[0])",
    ]
    return cells


def _expected(start: int, scale: int, steps: dict[int, int], k: int) -> str:
    s = start + sum(steps.get(i, i) for i in range(1, N + 1))
    return f"R {(s + k) * scale} {s}"


def _write_helper(tmp_path, k: int, module: str = "helper") -> None:
    (tmp_path / f"{module}.py").write_text(textwrap.dedent(f"""\
        K = {k}
        def bump(x):
            return x + K
        """), encoding="utf-8")


def _set_start(tmp_path, start: int) -> None:
    (tmp_path / "data.json").write_text(json.dumps({"start": start}), encoding="utf-8")


def test_the_last_cell_follows_every_change_above(nb_runner, tmp_path):
    tmp = str(tmp_path).replace("\\", "/")
    _write_helper(tmp_path, 1)
    _set_start(tmp_path, 10)
    steps: dict[int, int] = {}
    cells = _cells(tmp, "3", steps)
    last = len(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()

    def result() -> str:
        return nb_runner.get_output(last)

    nb_runner.run_all()
    assert _expected(10, 3, steps, 1) in result(), nb_runner.get_raw_output(last)

    # Unchanged: the same answer.
    nb_runner.run_all()
    assert _expected(10, 3, steps, 1) in result(), nb_runner.get_raw_output(last)

    # An edit near the top, then Run All.
    steps[3] = 100
    nb_runner.set_cell_source(2 + 3, _chain(3, 100))
    nb_runner.run_all()
    assert _expected(10, 3, steps, 1) in result(), nb_runner.get_raw_output(last)

    # An edit in the middle, and only the last cell run (out of order).
    steps[17] = -5
    nb_runner.set_cell_source(2 + 17, _chain(17, -5))
    nb_runner.run_cell(last)
    assert _expected(10, 3, steps, 1) in result(), nb_runner.get_raw_output(last)

    # A cell above edited and run on its own, then the last one: its lineage
    # changed in the kernel, not only in the file.
    steps[25] = 7
    nb_runner.set_cell_source(2 + 25, _chain(25, 7))
    nb_runner.run_cell(2 + 25)
    nb_runner.run_cell(last)
    assert _expected(10, 3, steps, 1) in result(), nb_runner.get_raw_output(last)

    # The file the second cell reads changed, and Run All.
    time.sleep(1.1)  # a new mtime on file systems that keep whole seconds
    _set_start(tmp_path, 1000)
    nb_runner.run_all()
    assert _expected(1000, 3, steps, 1) in result(), nb_runner.get_raw_output(last)

    # The environment setting edited in the first cell, and Run All. (An
    # edited setting that has not run reaches no reader, as in a plain
    # kernel: ``ReexecutionPlanner.unrun_process_writers``.)
    nb_runner.set_cell_source(1, _cells(tmp, "5", steps)[0])
    nb_runner.run_all()
    assert _expected(1000, 5, steps, 1) in result(), nb_runner.get_raw_output(last)

    # The imported module edited.
    time.sleep(1.1)  # a new mtime on file systems that keep whole seconds
    _write_helper(tmp_path, 40)
    nb_runner.run_cell(last)
    assert _expected(1000, 5, steps, 40) in result(), nb_runner.get_raw_output(last)

    # Another module imported under the same name, and Run All.
    _write_helper(tmp_path, 7, "helper_two")
    nb_runner.set_cell_source(1, _cells(tmp, "5", steps, "helper_two")[0])
    nb_runner.run_all()
    assert _expected(1000, 5, steps, 7) in result(), nb_runner.get_raw_output(last)

    # A new kernel, then only the last cell.
    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.run_cell(last)
    assert _expected(1000, 5, steps, 7) in result(), nb_runner.get_raw_output(last)

    # And Run All after it all.
    nb_runner.run_all()
    assert _expected(1000, 5, steps, 7) in result(), nb_runner.get_raw_output(last)
