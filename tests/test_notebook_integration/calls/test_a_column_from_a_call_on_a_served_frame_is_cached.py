"""``df['a'] = feat(df)`` is served once ``df = load()`` was served too.

Load in one cell, add an expensive feature column with a notebook function in
the next: once the load cell had been served from the cache, the values cash
read back for it were still held by a dict of its own, the share check
counted that dict as another holder of ``df``, and the feature statement was
refused, so ``feat`` ran on every Run All and after every restart.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"


def _cells(log):
    return [
        SETUP,
        "import os, time\nimport numpy as np, pandas as pd\n"
        f"LOG = {str(log)!r}\n"
        "def load():\n    time.sleep(0.3)\n    return pd.DataFrame({'x': np.arange(100_000.0)})\n"
        "def feat(d):\n"
        "    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
        "    os.write(fd, b'f'); os.close(fd)\n"
        "    time.sleep(0.3)\n"
        "    return d.x * 3",
        "df = load()",
        "df['a'] = feat(df)",
        "print('A', df.a.iloc[:3].tolist(), len(df))",
    ]


def test_the_feature_column_is_served_on_later_run_alls_and_after_a_restart(nb_runner, tmp_path):
    log = tmp_path / "feat.log"
    nb_runner.create_notebook(_cells(log))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "A [0.0, 3.0, 6.0] 100000" in nb_runner.get_output(5)
    assert log.read_text(encoding="utf-8") == "f"

    nb_runner.run_all()
    nb_runner.run_all()
    assert "A [0.0, 3.0, 6.0] 100000" in nb_runner.get_output(5)
    assert "holds an object another variable" not in nb_runner.get_output(4), nb_runner.get_output(4)
    assert log.read_text(encoding="utf-8") == "f", f"feat ran {len(log.read_text(encoding="utf-8"))} times"

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_all()
    assert "A [0.0, 3.0, 6.0] 100000" in nb_runner.get_output(5)
    assert log.read_text(encoding="utf-8") == "f", f"feat ran {len(log.read_text(encoding="utf-8"))} times with the restart"
