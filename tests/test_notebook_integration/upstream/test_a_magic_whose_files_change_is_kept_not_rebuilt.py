"""A magic statement whose line reads a file or folder in Python keeps its
value when that changes, with ``NOTEBOOK-MAGIC-STALE``: never rebuilt from
the Python above it.

``p = load()`` then ``p = !echo {len(os.listdir('d')) * 0}`` while another
process changes ``d/``: the listing was taken as ``p``'s own file, keyed
``p = load()`` on it, and a cell below reading ``CALLS`` rebuilt ``p`` as
``load()``'s list. A folder of the Python installation is no file of the
user's, nor is ``/proc`` where there is one, though it changes as
processes come and go: listed, neither is a dependency, and nothing is
stale.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

HELPERS = "import os\nCALLS = []\ndef load():\n    CALLS.append(1)\n    return [3, 1, None, 2]"
CHURN = """import os, sys, time
d = sys.argv[1]
i = 0
while True:
    open(os.path.join(d, str(i % 50)), 'w').close()
    gone = os.path.join(d, str((i + 25) % 50))
    if os.path.exists(gone):
        os.remove(gone)
    i += 1
    time.sleep(0.005)
"""


@pytest.mark.parametrize(
    "magic",
    [
        "p = !echo {len(os.listdir('d')) * 0}",
        "p = !echo {len(os.listdir(os.path.dirname(os.__file__))) * 0}",
        pytest.param(
            "p = !echo {len(os.listdir('/proc')) * 0}",
            marks=pytest.mark.skipif(not os.path.isdir("/proc"), reason="no /proc here"),
        ),
    ],
    ids=["a-changing-folder", "an-installed-folder", "the-proc-folder"],
)
def test_a_magic_reading_a_changing_folder_keeps_its_value(nb_runner, magic):
    d = nb_runner.work_dir / "d"
    d.mkdir()
    churn = subprocess.Popen([sys.executable, "-c", CHURN, str(d)])
    try:
        nb_runner.create_notebook(
            [HELPERS, "x = 1", f"p = load()\n{magic}\nq = len(p)", "print('C', p, q, len(CALLS))"]
        )
        nb_runner.start_kernel()
        for _ in range(2):
            nb_runner.run_all()
            out = nb_runner.get_raw_output(4)
            assert "C ['0'] 1 1" in out, out
            # The folder changes every few milliseconds: the magic is stale.
            assert ("NOTEBOOK-MAGIC-STALE" in out) is ("'d'" in magic), out
    finally:
        churn.kill()
        churn.wait()
