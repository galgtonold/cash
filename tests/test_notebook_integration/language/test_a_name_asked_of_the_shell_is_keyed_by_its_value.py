"""A statement that asks IPython's shell for a name by a string
(``get_ipython().ev('k')``) runs every time, and what it binds is keyed by
its value: a statement below that reads it, cached on the first run, is not
served again once ``k`` changed (a printed ``v=1`` when ``v`` is 2 now).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

#: How the statement asks the shell: the call itself, a name holding the
#: shell, the shell reached through the module.
ASKS = {
    "get_ipython()": "v = get_ipython().ev('k')",
    "a name holding the shell": "v = ip.ev('k')",
    "IPython.get_ipython()": "v = IPython.get_ipython().ev('k')",
}


@pytest.mark.parametrize("how", list(ASKS))
def test_a_reader_of_what_the_shell_found_is_not_served_its_old_output(nb_runner, how):
    # The sleep makes the print cost enough to be stored, as a loaded machine does.
    ask = ASKS[how] + "\n_ = (time.sleep(0.05), print(f'v={v}'))"
    setup = "%cash_badge print\nimport time\nimport IPython\nip = get_ipython()\nk = 1"
    nb_runner.create_notebook([setup, ask, "k = 2"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "v=1" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    again = " ".join(nb_runner.get_raw_output(2).split())
    assert "v=2" in again, again
    assert f"NOT CACHED: {ASKS[how]}" in again, again
