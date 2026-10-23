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


SKIP = ("cache",)


def _drop(d, keys):
    for k in keys:
        d.pop(k, None)
    return d


def _drop_inside(d):
    d["rows"].append(0)
    return d


class HandsAFreshCopy:
    def __getstate__(self):
        return _drop(self.__dict__.copy(), SKIP)


class HandsAFreshLocal:
    def __getstate__(self):
        state = dict(self.__dict__)
        return _drop(state, SKIP)


class HandsAShallowCopyChangedInside:
    def __getstate__(self):
        return _drop_inside(self.__dict__.copy())


class HandsADeepCopyChangedInside:
    def __getstate__(self):
        import copy

        return _drop_inside(copy.deepcopy(self.__dict__))


def test_one_handing_a_fresh_copy_to_a_helper_that_edits_it_is_no_writer():
    assert reducer_writes._decide(HandsAFreshCopy) is None
    assert reducer_writes._decide(HandsAFreshLocal) is None
    assert reducer_writes._decide(HandsADeepCopyChangedInside) is None


def test_one_whose_helper_changes_what_a_shallow_copy_shares_writes():
    assert reducer_writes._decide(HandsAShallowCopyChangedInside) is not None


def test_what_a_writer_reaches_is_found_by_object_not_by_name():
    writer = reducer_writes._decide(WritesAGlobal)
    # The notebook's own ``TABLE`` is another object: it is not reached.
    assert reducer_writes.reach([writer], {"TABLE": {}}) == set()
    # The module's table, bound to another name, is.
    assert reducer_writes.reach([writer], {"table": TABLE}) == {"table"}


def test_one_whose_source_cannot_be_read_is_refused_not_run(monkeypatch):
    """Its code unread, what it changes cannot be told: not run, it changes
    nothing, and only the value holding it is not stored. Run, every
    variable moved after it."""
    import sys
    import types

    module = types.ModuleType("_unread_reducer")
    module.__file__ = None
    exec("class Chosen:\n    def __getstate__(self):\n        return {'k': 1}\n", module.__dict__)
    module.Chosen.__module__ = module.__name__
    monkeypatch.setitem(sys.modules, module.__name__, module)
    reducer_writes.watch({"a": []})
    try:
        assert reducer_writes._writer(module.Chosen).functions is None
        assert reducer_writes.refuses(module.Chosen)
    finally:
        reducer_writes.unwatch()


VERSION = 2


class State:
    def __init__(self, d):
        self.d = d


class BuildsOne:
    def __reduce__(self):
        return (list, (State(self.rows), VERSION))


class StateChangingWhatItIsHanded:
    def __init__(self, d):
        self.d = d
        self.d.append(1)


class BuildsOneThatChangesItsArgument:
    def __reduce__(self):
        return (list, (StateChangingWhatItIsHanded(self.rows),))


_clean = lambda d: {k: v for k, v in d.items() if k != "cache"}


class ReadsThroughALambda:
    def __getstate__(self):
        return _clean(self.__dict__)


COUNT = 0


class CountsInAGlobalNumber:
    def __reduce__(self):
        global COUNT
        COUNT += 1
        return (CountsInAGlobalNumber, ())


def test_building_a_new_object_of_the_users_class_is_no_write():
    assert reducer_writes._decide(BuildsOne) is None


def test_building_one_whose_init_changes_its_argument_writes():
    assert reducer_writes._decide(BuildsOneThatChangesItsArgument) is not None


def test_a_lambda_that_only_reads_is_read():
    assert reducer_writes._decide(ReadsThroughALambda) is None


def test_an_equal_number_is_not_the_number_a_writer_changes():
    """``n = 1`` shares its object with any other 1: matched by id, a
    writer of ``COUNT`` reached it. Only the name it rebinds is reached,
    and only in its own namespace."""
    writer = reducer_writes._decide(CountsInAGlobalNumber)
    assert writer is not None
    assert reducer_writes.reach([writer], {"n": COUNT, "m": 2}) == set()
    assert reducer_writes.reach([writer], globals()) >= {"COUNT"}


def test_a_module_whose_list_it_changes_is_reached_through_the_module():
    """``len(rmod.SEEN)`` reads the list through the module: the notebook
    holds ``rmod``, not ``SEEN``. So does a function of the module."""
    import sys

    module = sys.modules[__name__]
    writer = reducer_writes._decide(HandsAGlobalOn)
    assert reducer_writes.reach([writer], {"rmod": module}) == {"rmod"}
    assert reducer_writes.reach([writer], {"record": _record}) == {"record"}


ARR: list = [0.0]


class HandsAGlobalToALibrary:
    def __getstate__(self):
        import operator

        operator.setitem(ARR, 0, 5.0)
        return {}


def test_a_global_handed_to_a_library_call_counts_as_written():
    writer = reducer_writes._decide(HandsAGlobalToALibrary)
    assert writer is not None and "ARR" in writer.handed
    assert reducer_writes.reach([writer], {"arr": ARR}) == {"arr"}


def test_cash_off_runs_every_reducer_again():
    reducer_writes.watch({"table": TABLE})
    try:
        assert reducer_writes.refuses(WritesAGlobal)
    finally:
        reducer_writes.unwatch()
    reducer_writes.runs_now()
    assert not reducer_writes.refuses(WritesAGlobal)
