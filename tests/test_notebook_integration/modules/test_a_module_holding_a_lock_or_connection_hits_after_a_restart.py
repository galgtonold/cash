"""A cell calling a module function that uses a module-level lock or database connection hits after a restart.

The module data a statement reaches is keyed by value; a ``threading.Lock``
or a ``sqlite3`` connection hashes by identity, so every new kernel made a
new key and ``x = mylib.slow(3)`` ran again after each restart.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIBS = {
    "lock": "import time, threading\n_LOCK = threading.Lock()\ndef slow(n):\n    with _LOCK:\n"
    "        time.sleep(0.3)\n    return n * 2\n",
    "sqlite": "import time, sqlite3\n_DB = sqlite3.connect(':memory:')\ndef slow(n):\n"
    "    _DB.execute('select 1')\n    time.sleep(0.3)\n    return n * 2\n",
}


@pytest.mark.parametrize("kind", sorted(LIBS))
def test_hits_after_a_restart(nb_runner, tmp_path, kind):
    (tmp_path / "handlelib.py").write_text(LIBS[kind], encoding="utf-8")
    nb_runner.create_notebook(["%cash_badge print\nimport handlelib", "x = handlelib.slow(3)\nprint('X', x)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X 6" in nb_runner.get_output(2)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert "CACHED: x = handlelib.slow(3)" in raw, raw
