"""A value whose rebuild would change a file is never served, and the file
keeps what it held.

``s = {'sink': Sink(path), 'n': slow()}`` holds an object whose
``__setstate__`` reopens its file with ``"w"``. The store's own copy was
checked, but the copy each hit makes, and the copy the notebook takes to
see whether the entry holds a variable's object itself, were not: the file
the cell above had just written was emptied on every Run All after the
first, where plain Python leaves it as it is.

Checked over Run All x3, a restart, and Run All x2, with the floors at their
defaults and at 0.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETUP = (
    "import time\n"
    "def slow(n=1):\n"
    "    time.sleep(0.3)\n"
    "    return n\n"
    "class Sink:\n"
    "    def __init__(self, path):\n"
    "        self.path, self.f = path, None\n"
    "    def __getstate__(self):\n"
    "        return {'path': self.path}\n"
    "    def __setstate__(self, st):\n"
    "        self.path = st['path']\n"
    "        self.f = open(self.path, 'w')\n"
)

FLOORS = {
    "default": "",
    "zero": "\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)",
}


@pytest.mark.parametrize("floors", list(FLOORS))
def test_the_file_keeps_what_the_cell_above_wrote(nb_runner, tmp_path, floors):
    sink = tmp_path / "sink.txt"
    cells = [
        "import cash\n%cash_on" + FLOORS[floors],
        SETUP,
        f"open({str(sink)!r}, 'w').write('keep me')",
        f"s = {{'sink': Sink({str(sink)!r}), 'n': slow()}}",
        f"print('R', repr(open({str(sink)!r}).read()))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    seen = []
    for run in range(5):
        if run == 3:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append([line for line in nb_runner.get_output(len(cells)).splitlines() if line.startswith("R ")])
    assert seen == [["R 'keep me'"]] * 5
