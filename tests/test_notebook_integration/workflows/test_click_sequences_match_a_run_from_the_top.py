"""Click sequences on small notebooks print what a plain run from the top prints.

Run All, edit, restart, run one cell further down: after every run the
cell's output is compared with a plain kernel running the notebook, as it
stands, from the top (``tests/_nbharness/click_sequences.py`` holds the
sequences, ``session_harness.Player`` the check). The notebooks are those
where a cell above changes something no name below spells: a list a helper
fills, a counter under ``global``, a bound method kept in a variable, a
closure that counts, a setting of the process, a variable read by ``eval``.
"""

from __future__ import annotations

import pytest

from tests._nbharness.click_sequences import EACH_CELL_ALONE, SEQUENCES
from tests._nbharness.session_harness import Player

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.fresh_kernel, pytest.mark.timeout(300)]


@pytest.mark.parametrize("session", SEQUENCES, ids=[s.name.replace(" ", "_") for s in SEQUENCES])
def test_every_run_of_the_sequence_matches_a_plain_run(session, nb_runner):
    player = Player(session, nb_runner)
    failures = player.play()
    assert not failures, player.report()


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("session", EACH_CELL_ALONE, ids=[s.name.replace(" ", "_") for s in EACH_CELL_ALONE])
def test_every_cell_run_alone_after_a_restart_matches_a_plain_run(session, nb_runner):
    player = Player(session, nb_runner)
    failures = player.play()
    assert not failures, player.report()
