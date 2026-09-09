"""``use_locking=True`` must not weaken what a cache hit means.

Round-16 gate finding (WRONG ANSWER, 5/5 against 0/5 without the flag). With
``Cash(use_locking=True)``, a file the cached function reads could be edited and
the next process would serve the answer computed from the old contents -- no
recompute, no warning. The tester found it while wrapping a paid API, which is
the pairing that makes it expensive: ``use_locking=True`` is the documented cure
for a concurrent burst turning into a burst of billing, so the people who turn
it on are the ones with money on the line, and it silently switched off the
README's headline file-awareness promise.

The mechanism is a double-check that had drifted from the first check. The
wrapper's unlocked read runs ``_try_get_cached``, which validates the TTL, the
recorded file dependencies and the chunk manifest. It correctly reported a miss.
``_compute_with_lock`` then re-read the entry inside the lock through its own
hand-rolled sequence -- which asked about the TTL and the chunks, and never
about the files -- and handed the entry back anyway.

This is the second omission of that kind in the same block: ``_chunks_are_intact``
was added to it after a chunked manifest served a SHORT iterator, 3 of 10 items,
silently. Twice means the duplication is the defect, so the locked path now
calls the same function the unlocked one does, and these tests pin that the two
agree.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

from cash import Cash


_CHILD = """
import json, os, sys, time
import cash
from cash import Cash

mode = sys.argv[1]
engine = Cash(use_locking=(mode == "lock")) if mode != "default" else cash
cfg_path = os.environ["PROBE_CONFIG"]


def load_config():
    with open(cfg_path, encoding="utf-8") as fh:
        return json.load(fh)


@engine.cache(assume_safe=True)
def answer(prompt):
    print("RAN", file=sys.stderr, flush=True)
    time.sleep(0.3)
    return f"{load_config()['model']}:{prompt}"


print("RESULT", answer("q1"))
"""


@pytest.fixture
def project(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(_CHILD), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"model": "MODEL-A"}', encoding="utf-8")
    return script, config


def _run(script, mode, tmp_path, config):
    env = dict(os.environ,
               CASH_CACHE_DIR=str(tmp_path / f"cache_{mode}"),
               PROBE_CONFIG=str(config))
    proc = subprocess.run([sys.executable, str(script), mode], env=env,
                          cwd=str(tmp_path), capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip(), proc.stderr.count("RAN")


@pytest.mark.parametrize("mode", ["lock", "nolock", "default"])
def test_editing_a_tracked_file_invalidates_in_every_mode(project, tmp_path, mode):
    """The reported sequence: cold, an unchanged control, then the edit.

    Parametrised across the three ways to get a cached function, because the
    finding was precisely that they disagreed -- and the control run is what
    makes the third arm mean anything: without a HIT there, "it recomputed"
    proves nothing.
    """
    script, config = project

    assert _run(script, mode, tmp_path, config) == ("RESULT MODEL-A:q1", 1)
    assert _run(script, mode, tmp_path, config) == ("RESULT MODEL-A:q1", 0), (
        "control: the unchanged second run must be a hit"
    )

    config.write_text('{"model": "MODEL-B"}', encoding="utf-8")

    result, ran = _run(script, mode, tmp_path, config)
    assert result == "RESULT MODEL-B:q1", (
        f"mode={mode} served the answer computed from the old file"
    )
    assert ran == 1


def test_the_locked_path_still_serves_a_valid_entry(tmp_path):
    """The control for the fix: validating more must not stop it caching.

    The locked re-read is the single-flight payoff -- the follower has to be
    served the leader's value, not compute its own.
    """
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False,
             use_locking=True)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def work(n):
        runs.append(n)
        time.sleep(0.2)
        return n * 2

    assert work(21) == 42
    assert work(21) == 42
    assert len(runs) == 1, "the locked path stopped serving cached values"


def test_an_expired_entry_still_recomputes_under_the_lock(tmp_path):
    """The TTL check the locked path already had must survive the refactor."""
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False,
             use_locking=True)
    runs: list[int] = []

    @c.cache(ttl=1, assume_safe=True)
    def work(n):
        runs.append(n)
        return n * 2

    assert work(21) == 42
    time.sleep(1.2)
    assert work(21) == 42
    assert len(runs) == 2, "the TTL stopped being honoured"
