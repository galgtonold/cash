"""A helper that writes a step marker to stderr does not make its callers file writers.

One user marked each step with ``os.write(2, f"FDRUN {step} ...")``
inside a ``mark()`` helper, called from the loader, the feature builder and
every fit. After a restart, a sanity-check cell at the bottom that reads only
the loaded frame ran 70 statements, every fit among them, in 19 s where the
cells it needs take 4 s: each statement reaching ``mark`` counted as writing
a file of unknown kind, had no record of its files after the restart, and was
re-fired.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.files]

MARKERS = {
    "os_write": "os.write(2, f'RUN {step}\\n'.encode())",
    "sys_stderr": "sys.stderr.write(f'RUN {step}\\n')",
}


def _cells(marker):
    return [
        "import cash\n%cash_on\n%cash_badge print",
        "import os, sys, time\nimport pandas as pd\ndef mark(step):\n    " + marker,
        "def load():\n    mark('load')\n    return pd.DataFrame({'m': [1, 1, 2], 'v': [3.0, 4.0, 5.0]})\nraw = load()",
        "def fit(frame):\n    mark('fit')\n    time.sleep(0.2)\n    return float(frame['v'].sum())\n"
        "model = fit(raw)\nprint('MODEL', model)",
        "print('BY MONTH', raw.groupby('m').size().to_dict())",
    ]


@pytest.mark.parametrize("marker", list(MARKERS), ids=list(MARKERS))
def test_after_a_restart_a_cell_needing_only_the_frame_does_not_refit(nb_runner, marker):
    nb_runner.create_notebook(_cells(MARKERS[marker]))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "MODEL 12.0" in nb_runner.get_output(4)

    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)

    assert "BY MONTH {1: 2, 2: 1}" in nb_runner.get_output(5), nb_runner.get_output(5)
    # Counted from the badge: the marker has no file to count in, which is the point.
    raw = nb_runner.get_raw_output(5)
    assert "model = fit(raw)" not in raw, raw
