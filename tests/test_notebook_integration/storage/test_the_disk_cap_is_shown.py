"""A notebook is told how big its disk cache may grow, and when it first evicts.

``%cash_on`` names the cache folder and its cap; the cell whose writes first
take the cache over the cap says what was removed, and later evictions stay
quiet. The unit arms are ``tests/test_backends/test_the_disk_cap_is_said_out_loud.py``
and ``tests/test_notebook/test_cash_on_shows_the_disk_cap.py``.

The cache folder is a fresh one set in the first cell: the kernel's own
start-up ``%cash_on`` has already shown the cap of its default folder.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

# ~400 KB values that took 0.3 s: worth their disk, so each reaches the 3 MB
# disk tier; eight of them do not fit.
SETUP = "import time\ndef slow(n):\n    time.sleep(0.3)\n    return bytes([n]) * 400_000"
N_VALUES = 12


def _cells():
    return [
        "import cash\ncash.configure(cache_dir='capped_cache', max_cache_size='3MB')\n%cash_on",
        SETUP,
        *[f"v{i} = slow({i})" for i in range(N_VALUES)],
    ]


def test_cash_on_shows_the_cap_and_the_first_eviction_is_said_once(nb_runner):
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()

    first = nb_runner.get_output(1)
    assert "Caching in " in first and "capped_cache, up to 3 MB (set by max_cache_size)." in first, first

    outputs = [nb_runner.get_output(3 + i) for i in range(N_VALUES)]
    said = [i for i, out in enumerate(outputs) if "reached its 3 MB cap, so cash removed" in out]
    assert len(said) == 1, f"the first eviction should be said in exactly one cell: {outputs!r}"
    assert said[0] >= 5, f"said before the cache could have been full: cell {said[0]}"
    assert nb_runner.peek("v11[:1]") == "b'\\x0b'"
