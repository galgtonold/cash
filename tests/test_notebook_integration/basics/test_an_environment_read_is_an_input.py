"""A statement reading an environment variable is keyed on its value.

``tenant = os.getenv("TENANT")`` was cached with nothing of the variable in
its key: a new tenant got the first tenant's value, and so did everything
built on it. The key now folds a digest of the value, and so does the
lineage of what the statement binds, in the runtime and in the upstream
simulation -- so running only the cell at the bottom after the variable
changed rebuilds what depends on it, and going back to a value seen before
restores its entries.

The variable is set with ``peek``, outside the notebook's cells, as a shell
or a launcher would set it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\nimport os, time",
    "tenant = os.getenv('CASH_IT_TENANT')",
    "def shout(t):\n    time.sleep(1.0)\n    return t.upper()\nloud = shout(tenant)",
    "print('OUT', loud)",
]


def _setenv(nb_runner, value):
    nb_runner.peek(f"__import__('os').environ.__setitem__('CASH_IT_TENANT', {value!r})")


def test_what_is_built_on_the_read_follows_the_variable(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    _setenv(nb_runner, "acme")
    nb_runner.run_all()
    assert "OUT ACME" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    # Only the last cell: the upstream check has to see that `tenant` moved.
    _setenv(nb_runner, "globex")
    nb_runner.run_cell(4)
    assert "OUT GLOBEX" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
    assert nb_runner.peek("loud") == "'GLOBEX'"
    assert "loud = shout" in nb_runner.get_raw_output(4), nb_runner.get_raw_output(4)

    # Nothing moved since: the simulation must agree with what ran, or
    # everything above re-runs every time.
    nb_runner.run_cell(4)
    raw = nb_runner.get_raw_output(4)
    assert "OUT GLOBEX" in raw and "loud = shout" not in raw, raw

    # A new kernel, and only the last cell: the simulation rebuilds the
    # lineages from the code, and must arrive at the ones the entries were
    # stored under, or the slow step runs again.
    nb_runner.restart()
    _setenv(nb_runner, "acme")
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)
    raw = nb_runner.get_raw_output(4)
    assert "OUT ACME" in raw, raw
    assert "EXECUTED: loud = shout" not in raw, "the entry made for this value was not restored:\n" + raw
