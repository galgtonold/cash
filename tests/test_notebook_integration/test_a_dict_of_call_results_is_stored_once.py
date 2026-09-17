"""A dict gathered from cached calls is stored once, and restores after a restart.

Round 25: ``holdout_models = {key: fit_series(g, ...) for ...}`` stored ~120 MiB
per version on top of the same fits under their call keys (r25s5), and every
project's cache grew 1.5 GB a day. The statement's entry now refers to the call
entries. It must still restore on its own after a restart -- that entry is what
spares rebuilding the calls' arguments.
"""
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
CELLS = [
    SETUP,
    "import time\ndef fit(k):\n    time.sleep(0.3)\n    return bytes([k]) * 3_000_000",
    "raw = list(range(3))",
    "models = {k: fit(k) for k in raw}",
    "print('M', sorted((k, len(v), v[0]) for k, v in models.items()))",
]
EXPECTED = "M [(0, 3000000, 0), (1, 3000000, 1), (2, 3000000, 2)]"


def _cache_bytes(nb_runner):
    root = os.path.join(str(nb_runner.work_dir), ".cash")
    return sum(e.stat().st_size for e in os.scandir(root) if e.name.endswith(".entry"))


def test_the_dict_restores_after_a_restart_without_a_second_copy(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert EXPECTED in nb_runner.get_output(5)
    nb_runner.peek("__import__('cash').get_default_cash().backend.flush() if hasattr(__import__('cash').get_default_cash().backend, 'flush') else None")
    size = _cache_bytes(nb_runner)
    # Three 3 MB results, stored under their call keys, and not again in the dict.
    assert size < 3 * 3_000_000 * 1.5, f"{size / 1e6:.1f} MB on disk"

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)
    out = nb_runner.get_output(5)
    assert EXPECTED in out, out
    assert "CACHED: models = " in out, out
