"""A notebook warning names the cell it is about, not IPython's machinery.

Raised from cash's own check of a statement, the nearest frame outside cash
is ipykernel's or contextlib's: NOTEBOOK-UNSEEN-WRITES, NOTEBOOK-WORK-RUNNING
and NOTEBOOK-THREAD-ALIVE read ``ipkernel.py:460:`` or ``contextlib.py:144:``,
which tells a reader nothing. They name the cell (``<cash>:N``) as cash's
other notebook warnings do.
"""

import re

import pytest

from cash.decorator import code_trace

pytestmark = [pytest.mark.integration, pytest.mark.fresh_kernel, pytest.mark.timeout(240)]

CELLS = [
    "import threading, time\ncalls = []\nclass Vec:\n    def __add__(self, x):\n        calls.append(1)\n"
    "        return 0\nstop = threading.Event()\ndef fill():\n    stop.wait(30)",
    "y = (time.sleep(0.3), 1)[1]",
    "t = threading.Thread(target=fill, name='filler')\nt.start()",
    "stop.set()\nt.join()\nstop.clear()",
    "z = sum([1], Vec())",
]


def _blamed(out: str, code: str) -> list[str]:
    """The ``file:line`` each *code* warning in *out* names."""
    return re.findall(rf"(\S+:\d+): \w+Warning: \[{code}\]", out)


def test_each_warning_names_its_cell(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    found = {"NOTEBOOK-THREAD-ALIVE": (3, _blamed(nb_runner.get_raw_output(3), "NOTEBOOK-THREAD-ALIVE"))}
    # Run while the thread runs, the stored cell 2 is run rather than served.
    nb_runner.run_cell(3)
    nb_runner.run_cell(2)
    found["NOTEBOOK-WORK-RUNNING"] = (2, _blamed(nb_runner.get_raw_output(2), "NOTEBOOK-WORK-RUNNING"))
    nb_runner.run_cell(4)
    if code_trace.on():
        found["NOTEBOOK-UNSEEN-WRITES"] = (5, _blamed(nb_runner.get_raw_output(5), "NOTEBOOK-UNSEEN-WRITES"))
    for code, (cell, blamed) in found.items():
        assert blamed, (code, [nb_runner.get_raw_output(i) for i in range(1, 6)])
        assert all(place.startswith("<cash>:") for place in blamed), (code, cell, blamed)
