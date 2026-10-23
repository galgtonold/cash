"""A statement that reads a value work left running in a thread can change
runs every time; what it binds is keyed by its value, so a statement below
that reads it, cached on the first run, is not served again once the
thread changed it (a printed ``snap=0`` when ``snap`` is 3 now).
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

START = textwrap.dedent("""\
    %cash_badge print
    import threading, time
    data = []
    ev = threading.Event()
    def worker():
        ev.wait()
        data.extend([1, 2, 3])
    t = threading.Thread(target=worker, daemon=True)
    t.start()
""")
# The sleep makes the print cost enough to be stored, as a loaded machine does.
READ = "snap = len(data)\n_ = (time.sleep(0.05), print(f'snap={snap}'))"
RELEASE = "ev.set()\nt.join()\nprint(f'n={len(data)}')"


def test_a_reader_of_what_the_thread_changed_is_not_served_its_old_output(nb_runner):
    nb_runner.create_notebook([START, READ, RELEASE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "snap=0" in nb_runner.get_output(2)
    assert "n=3" in nb_runner.get_output(3)

    nb_runner.run_cell(2)
    again = " ".join(nb_runner.get_raw_output(2).split())
    assert "snap=3" in again, again
    assert "snap=0" not in again, again


def test_a_reader_of_an_unchanged_value_is_still_served(nb_runner):
    # The work ends leaving ``data`` as it was: nothing is served while it
    # runs (NOTEBOOK-WORK-RUNNING), and after, the reader of the same value is.
    nb_runner.create_notebook([START.replace("data.extend([1, 2, 3])", "pass"), READ, RELEASE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    lines = [line.strip() for line in nb_runner.get_raw_output(2).splitlines()]
    assert "snap=0" in lines, lines
    assert any(line.startswith("CACHED: _ = (time.sleep(0.05)") for line in lines), lines
