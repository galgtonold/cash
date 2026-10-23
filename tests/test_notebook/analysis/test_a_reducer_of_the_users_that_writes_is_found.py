"""Which reducers of the user's write (`reducer_writes`): one that only
reads stays cached; one that changes a global, or hands something to a
function of the user's that changes it, writes; one that changes the
object it pickles writes where no reach can be read.
"""

from __future__ import annotations

from cash.notebook.statement import reducer_writes

LOG: list = []
TABLE: dict = {}


def _record(target):
    target.append(1)


def _tidy(state):
    return {k: v for k, v in state.items() if k != "cache"}


class OnlyReads:
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("cache", None)
        return _tidy(state)


class WritesAGlobal:
    def __reduce__(self):
        TABLE["seen"] = True
        return (WritesAGlobal, ())


class HandsAGlobalOn:
    def __reduce__(self):
        _record(LOG)
        return (HandsAGlobalOn, ())


class WritesItself:
    def __getstate__(self):
        self.pickled = True
        return self.__dict__


def test_one_that_only_reads_is_no_writer():
    assert reducer_writes._decide(OnlyReads) is None


def test_one_that_changes_a_global_writes_and_reaches_it():
    writer = reducer_writes._decide(WritesAGlobal)
    assert writer is not None and writer.name == "WritesAGlobal"
    assert reducer_writes.reach([writer], {"TABLE": TABLE, "LOG": LOG}) == {"TABLE"}


def test_one_that_hands_a_global_to_a_function_that_changes_it_writes():
    writer = reducer_writes._decide(HandsAGlobalOn)
    assert writer is not None
    assert "LOG" in reducer_writes.reach([writer], {"TABLE": TABLE, "LOG": LOG})


def test_one_that_changes_what_it_pickles_reaches_every_variable():
    writer = reducer_writes._decide(WritesItself)
    assert writer is not None and reducer_writes.reach([writer], {"x": 1}) is None


def test_one_whose_source_cannot_be_read_reaches_every_variable():
    namespace: dict = {}
    exec("class Hidden:\n    def __reduce__(self):\n        return (Hidden, ())\n", namespace)
    hidden = namespace["Hidden"]
    hidden.__module__ = __name__
    hidden.__reduce__.__module__ = __name__
    writer = reducer_writes._decide(hidden)
    assert writer is not None and reducer_writes.reach([writer], {"x": 1}) is None
