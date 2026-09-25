"""A statement whose value the disk cap evicted says so when it recomputes.

After a restart, a statement whose entry the cap removed misses like any
other. The disk tier notes what its cap evicts, so the miss is attributed to
the eviction, and a recompute that takes seconds warns once
(``CACHE-EVICTED-RECOMPUTE``) with the cap and how to raise it. The unit arms
are ``tests/test_backends/test_evicted_results_are_noted.py`` and
``tests/test_notebook/test_evicted_statement_is_attributed.py``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240), pytest.mark.fresh_kernel]

# 1.2 MB values that took 2.1 s -- over the 2 s "worth telling" floor -- in a
# 3 MB disk tier: the third one evicts the first.
SETUP = "import time\ndef slow(n):\n    time.sleep(2.1)\n    return bytes([n]) * 1_200_000"
CELLS = [
    "import cash\ncash.configure(cache_dir='capped_cache', max_cache_size='3MB')\n%cash_on",
    SETUP,
    "a = slow(1)",
    "b = slow(2)",
    "c = slow(3)",
    "print(a[:1], b[:1], c[:1])",
]


def test_a_restarted_notebook_says_an_evicted_value_was_recomputed(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = [nb_runner.get_raw_output(i) for i in range(1, 7)]
    assert "reached its 3 MB cap" in "".join(first), "nothing was evicted, so nothing is under test"
    assert "CACHE-EVICTED-RECOMPUTE" not in "".join(first)

    nb_runner.restart()
    nb_runner.run_all()
    outputs = {i: nb_runner.get_raw_output(i) for i in (3, 4, 5)}
    warned = [i for i, out in outputs.items() if "[CACHE-EVICTED-RECOMPUTE]" in out]
    # Which value the cap chose is GDSF's business; that the one it chose says
    # so, once, with the cap and the fix, is this test's.
    assert len(warned) == 1, outputs
    out = outputs[warned[0]]
    assert f"the value of `{CELLS[warned[0] - 1]}` had been evicted from the disk cache to make room" in out, out
    assert "3 MB cap" in out and "raise max_cache_size above 3 MB" in out, out
    assert "b'\\x01' b'\\x02' b'\\x03'" in nb_runner.get_output(6)
