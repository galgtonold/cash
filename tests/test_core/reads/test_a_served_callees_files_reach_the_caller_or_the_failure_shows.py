"""Handing a served callee's files to its caller does not fail in silence.

When a cached function is served from its entry inside another cached
function, the files that entry depends on are handed to the caller's
tracker, so the caller's own entry depends on them too. The hand-over
began with a ``try`` around reading which tracker is active, and gave up
without a word when that raised: the caller's entry would then be stored
without the callee's files and served after one of them changed. Reading
the active tracker cannot fail in ordinary use; if it ever does, the call
must show it, not store an entry that is missing dependencies.
"""

from __future__ import annotations

import pytest

from cash.decorator import file_deps


class _Broken:
    def get(self):
        raise RuntimeError("no tracker can be read")


class _Entry:
    auto_file_deps = {"/data/input.csv": {"hash": "abc"}}


def test_a_hit_does_not_drop_its_files_when_the_tracker_cannot_be_read(monkeypatch):
    monkeypatch.setattr(file_deps, "active_tracker", _Broken())
    with pytest.raises(RuntimeError, match="no tracker can be read"):
        file_deps.propagate_file_deps_to_active_tracker(_Entry(), "load")


def test_an_unentered_body_is_not_dropped_either(monkeypatch):
    monkeypatch.setattr(file_deps, "active_tracker", _Broken())
    with pytest.raises(RuntimeError, match="no tracker can be read"):
        file_deps.note_unentered_body("load")


def test_with_no_tracker_active_there_is_nothing_to_hand_over():
    file_deps.propagate_file_deps_to_active_tracker(_Entry(), "load")
    file_deps.note_unentered_body("load")
