"""Replay real user sessions, step by step, against a plain run.

See ``session_harness`` for the steps and what is checked after each, and
``recorded_sessions`` for the sessions. Each session runs twice: with cash's
defaults, and with every result persisted -- several bugs appeared
only once a value was big or slow enough to reach the disk, which a small
dataset alone never makes happen.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from tests._nbharness.recorded_sessions import SESSIONS  # noqa: E402
from tests._nbharness.session_harness import Player  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(900)]


@pytest.mark.parametrize("persist", [False, True], ids=["default", "persist_all"])
@pytest.mark.parametrize("session", SESSIONS, ids=[s.name.split()[0] for s in SESSIONS])
def test_session_matches_a_plain_run_at_every_step(session, persist, nb_runner):
    player = Player(session, nb_runner, persist=persist)
    failures = player.play()
    assert not failures, player.report()
