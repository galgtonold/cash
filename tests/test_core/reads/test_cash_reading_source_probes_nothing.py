"""Cash looking up a function's source file is nobody's existence probe.

``inspect.getsourcefile`` asks ``os.path.exists`` of the file a function's
code names, and the tracker recorded the answer as the running statement's
absent file. networkx compiles each ``argmap`` decorator under a filename
numbered by a process-wide counter -- ``<class
'networkx.utils.decorators.argmap'> compilation 36`` -- so a graph built
after a kernel restart probed ``... compilation 4`` instead: its lineage
changed, and every statement built on it re-ran.

The control arm: the same probe made by the user's code stays tracked.
"""

from __future__ import annotations

import os

import pytest

from cash import source_reading
from cash.tracking.file_tracker import FileAccessTracker

pytestmark = [pytest.mark.core]

PSEUDO = "<class 'fakelib.argmap'> compilation 7"


def _compiled_away_from_any_file():
    namespace: dict = {}
    exec(compile("def wrapped(x):\n    return x\n", PSEUDO, "exec"), namespace)
    return namespace["wrapped"]


def _ignoring_oserror(read, *args):
    try:
        read(*args)
    except OSError:
        pass  # no source to read: the probe is what is under test


@pytest.mark.parametrize(
    "read",
    [
        source_reading.getsourcelines,
        source_reading.getsource,
        source_reading.own_source,
        source_reading.settled_source_version,
        source_reading.code_is_where_it_says,
        source_reading.defining_file,
    ],
    ids=lambda f: f.__name__,
)
def test_reading_source_of_code_compiled_under_a_made_up_name_probes_nothing(read):
    fn = _compiled_away_from_any_file()
    target = fn.__code__ if read is source_reading.code_is_where_it_says else fn
    tracker = FileAccessTracker()
    with tracker:
        _ignoring_oserror(read, target)
    assert not tracker.get_absent_files(), tracker.get_absent_files()
    assert not tracker.get_present_files(), tracker.get_present_files()


def test_the_same_probe_made_by_the_users_code_is_tracked():
    tracker = FileAccessTracker()
    with tracker:
        assert not os.path.exists(PSEUDO)
    assert any("compilation 7" in p for p in tracker.get_absent_files()), tracker.get_absent_files()
