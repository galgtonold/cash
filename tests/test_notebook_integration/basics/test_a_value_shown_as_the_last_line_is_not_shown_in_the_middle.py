"""An expression stored as a cell's last line does not replay its display when it comes back in the middle.

``slow(5)`` as a cell's last line shows its value, and the stored entry held
that display. The same statement as the first line of a longer cell shows
nothing in a plain run, but was served that entry and its display (found
replaying a student's notebook of the JuNE dataset, 'student_9' steps 69-70).
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

SLOW = f"import time\ndef slow(n):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n    return list(range(n))"


def test_a_display_is_not_replayed_for_a_statement_that_is_not_the_last(nb_runner):
    nb_runner.create_notebook([SLOW, "slow(5)", "slow(5)\nx = 1"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "[0, 1, 2, 3, 4]" in nb_runner.get_output(2)
    assert "[0, 1, 2, 3, 4]" not in nb_runner.get_output(3), nb_runner.get_output(3)
