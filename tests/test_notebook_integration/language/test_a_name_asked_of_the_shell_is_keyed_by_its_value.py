"""A statement that asks IPython's shell for a name by a string
(``get_ipython().ev('k')``) runs every time, and what it binds is keyed by
its value: a statement below that reads it, cached on the first run, is not
served again once ``k`` changed (a printed ``v=1`` when ``v`` is 2 now).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

# The sleep makes the print cost enough to be stored, as a loaded machine does.
ASK = "v = get_ipython().ev('k')\n_ = (time.sleep(0.05), print(f'v={v}'))"


def test_a_reader_of_what_the_shell_found_is_not_served_its_old_output(nb_runner):
    nb_runner.create_notebook(["%cash_badge print\nimport time\nk = 1", ASK, "k = 2"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "v=1" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    again = " ".join(nb_runner.get_raw_output(2).split())
    assert "v=2" in again, again
    assert "NOT CACHED: v = get_ipython().ev('k')" in again, again
