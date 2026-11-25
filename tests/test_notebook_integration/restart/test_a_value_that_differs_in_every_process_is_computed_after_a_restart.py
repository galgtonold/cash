"""After a restart, a statement reading the process's id or the hash of
text gives the new kernel's value.

``hash()`` of a string is salted per process and ``os.getpid()`` is the
process's own. A statement that took long enough to be stored was restored
after a restart with the old kernel's number, which no longer matched the
same expression written out in the next cell. The value is part of the
statement's key, as an environment variable's is: the new kernel computes
its own, and the same kernel is still served.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import os, time\ndef bucket(user):\n    time.sleep(0.4)\n    return hash(user) % 100000",
    "shard = [time.sleep(0.4), hash('alice') % 100000][1]",
    "worker = [time.sleep(0.4), os.getpid()][1]",
    "group = bucket('bob')",
    "stable = [time.sleep(0.4), hash((3, 4)) % 1000][1]",
]

AGREES = (
    "(shard, worker, group, stable) == (hash('alice') % 100000, os.getpid(), hash('bob') % 100000, hash((3, 4)) % 1000)"
)


def test_the_new_kernel_computes_its_own_value(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek(AGREES) == "True"

    # The same kernel: its values hold, and the statements are served.
    nb_runner.run_all()
    assert nb_runner.peek(AGREES) == "True"
    for number in (3, 4, 6):
        assert shows_cached(nb_runner.get_output(number)), nb_runner.get_output(number)

    nb_runner.restart()
    nb_runner.run_all()

    assert nb_runner.peek(AGREES) == "True"
    # The hash of numbers is the same in every process: served across the restart.
    assert shows_cached(nb_runner.get_output(6)), nb_runner.get_output(6)
