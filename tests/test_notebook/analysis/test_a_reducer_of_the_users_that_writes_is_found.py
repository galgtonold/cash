"""Which reducers of the user's write (`reducer_writes`): one that only
reads stays cached; one that changes a global, or hands something to a
function of the user's that changes it, writes; one that changes the
object it pickles writes where no reach can be read.
"""

from __future__ import annotations

import pytest

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
    # Read as written by the mutation analysis, or else as handed on.
    assert writer is not None and "ARR" in writer.written | writer.handed
    assert reducer_writes.reach([writer], {"arr": ARR}) == {"arr"}


def test_cash_off_runs_every_reducer_again():
    reducer_writes.watch({"table": TABLE})
    try:
        assert reducer_writes.refuses(WritesAGlobal)
    finally:
        reducer_writes.unwatch()
    reducer_writes.runs_now()
    assert not reducer_writes.refuses(WritesAGlobal)


KEY = "a"
VER = 2
INDENT = 2
SEP = ","
SCALE = 2.5
DT = float
PARTS = ["a", "b"]
LST = [1, 2, 3]
DEFAULTS = {"k": [1]}


class OnlyReadsWhatItIsHanded:
    def __getstate__(self):
        import copy
        import functools
        import itertools
        import json
        import math
        import operator

        return {
            "a": self.__dict__.get(KEY),
            "b": "v{}".format(VER),
            "c": json.dumps(self.__dict__, indent=INDENT),
            "d": "a,b".split(SEP),
            "e": math.floor(SCALE),
            "f": DT(SCALE),
            "g": "-".join(PARTS),
            "h": list(map(str, LST)) + list(filter(None, LST)) + list(iter(LST)) + list(reversed(LST)),
            "i": list(itertools.chain(LST)),
            "j": functools.reduce(operator.add, LST),
            "k": copy.deepcopy(DEFAULTS),
            "l": SEP.join(PARTS),
        }


def test_a_reducer_that_hands_globals_only_to_calls_that_read_is_no_writer():
    assert reducer_writes._decide(OnlyReadsWhatItIsHanded) is None


CFG = {"k": 1}


class WritesAGlobalDict:
    def __reduce__(self):
        CFG["k"] += 1
        return (WritesAGlobalDict, ())


def test_a_cached_function_returning_one_is_not_stored_and_the_reducer_does_not_run(disk_cash):
    """With the default backend, RAM and disk tiers alike."""
    import warnings

    from cash.backends.tiered_backend import TieredBackend
    from cash.decorator.store import REDUCER_NOT_RUN_FIX

    calls = []

    @disk_cash.cache
    def make(i):
        calls.append(i)
        return WritesAGlobalDict()

    reducer_writes.watch({"cfg": CFG})
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            make(1)
            make(1)
    finally:
        reducer_writes.unwatch()
    assert isinstance(disk_cash.backend, TieredBackend)
    assert CFG == {"k": 1}
    assert calls == [1, 1]
    assert any(REDUCER_NOT_RUN_FIX.split(":")[0] in str(w.message) for w in seen)


class HandsAGlobalToAMethodOfABuiltinType:
    def __getstate__(self):
        list.append(LST, 4)
        dict.update(DEFAULTS, k=2)
        return {}


def test_a_global_handed_to_a_method_of_a_builtin_type_counts_as_written():
    writer = reducer_writes._decide(HandsAGlobalToAMethodOfABuiltinType)
    assert writer is not None and {"LST", "DEFAULTS"} <= writer.written | writer.handed


def test_a_cached_result_whose_reducer_cannot_be_read_is_not_stored_and_says_why_once(disk_cash):
    import warnings

    ns: dict = {"__name__": __name__}
    exec("class Unread:\n    def __reduce__(self):\n        return (Unread, ())\n", ns)
    calls = []

    @disk_cash.cache
    def make(i):
        calls.append(i)
        return ns["Unread"]()

    reducer_writes.watch({})
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            for i in (1, 1, 2):
                make(i)
    finally:
        reducer_writes.unwatch()
    assert calls == [1, 1, 2]
    said = [str(w.message) for w in seen if "STORE-FAILED" in str(w.message)]
    assert len(said) == 1 and "its source cannot be read" in said[0], said


ROWS_CFG = {"rows": []}
MEMO: dict = {}
ACC: list = []
NESTED = [[1], [2]]


class ConfigObject:
    def __init__(self):
        self.seen = []


CFG_OBJECT = ConfigObject()


class ChangesAShallowCopyOfAGlobal:
    def __getstate__(self):
        import copy

        c = copy.copy(ROWS_CFG)
        c["rows"].append(1)
        return {}


class ChangesAShallowCopyOfAGlobalObject:
    def __getstate__(self):
        import copy

        c = copy.copy(CFG_OBJECT)
        c.seen.append(len(c.seen))
        return {}


class ChangesADictOfAGlobal:
    def __getstate__(self):
        c = dict(ROWS_CFG)
        c["rows"].append(1)
        return {}


class ChangesACopyMethodsCopyOfAGlobal:
    def __getstate__(self):
        rows = ROWS_CFG.copy()["rows"]
        c = ROWS_CFG.copy()
        c["rows"].append(len(rows))
        return {}


class ChangesASliceOfAGlobalsAlias:
    def __getstate__(self):
        alias = ACC
        c = alias[:]
        c.append(1)
        return {}


class DumpsIntoAGlobalFile:
    def __getstate__(self):
        import json

        json.dump(self.__dict__, LOG)
        return {}


class FillsAGlobalMemo:
    def __getstate__(self):
        import copy

        return copy.deepcopy(self.__dict__, MEMO)


class ReducesIntoAGlobal:
    def __getstate__(self):
        import functools
        import operator

        functools.reduce(operator.iadd, [[1]], ACC)
        return {}


class ReducesInPlaceOverAGlobal:
    def __getstate__(self):
        import functools
        import operator

        return {"s": functools.reduce(operator.iadd, NESTED)}


@pytest.mark.parametrize(
    "cls, name",
    [
        (ChangesAShallowCopyOfAGlobal, "ROWS_CFG"),
        (ChangesAShallowCopyOfAGlobalObject, "CFG_OBJECT"),
        (ChangesADictOfAGlobal, "ROWS_CFG"),
        (ChangesACopyMethodsCopyOfAGlobal, "ROWS_CFG"),
        (ChangesASliceOfAGlobalsAlias, "ACC"),
        (DumpsIntoAGlobalFile, "LOG"),
        (FillsAGlobalMemo, "MEMO"),
        (ReducesIntoAGlobal, "ACC"),
        (ReducesInPlaceOverAGlobal, "NESTED"),
    ],
)
def test_a_global_changed_through_a_copy_or_a_library_function_that_writes_counts(cls, name):
    """``copy.copy`` shares what it copies, ``json.dump`` writes its file,
    ``copy.deepcopy`` its memo, ``functools.reduce(operator.iadd, ..., ACC)``
    its initial value: only ``json.dumps``/``loads``, ``copy.deepcopy`` with
    no memo and a ``reduce`` of a function that changes nothing read."""
    writer = reducer_writes._decide(cls)
    assert writer is not None and name in writer.written | writer.handed


class CopiesAGlobalAndChangesNothing:
    def __getstate__(self):
        import json

        c = dict(ROWS_CFG)
        return {"n": len(c["rows"]), "j": json.loads(json.dumps(DEFAULTS))}


def test_a_copy_of_a_global_that_is_only_read_is_no_writer():
    assert reducer_writes._decide(CopiesAGlobalAndChangesNothing) is None


_BODIES = {
    "an_item": "c = ALIASED['rows']\nc.append(1)",
    "get": "c = ALIASED.get('rows')\nc.append(1)",
    "an_attribute": "c = ALIASED_OBJECT.seen\nc.append(1)",
    "getattr": "c = getattr(ALIASED_OBJECT, 'seen')\nc.append(1)",
    "a_walrus": "len(c := ALIASED['rows'])\nc.append(1)",
    "a_dict_unpacked": "c = {**ALIASED}\nc['rows'].append(1)",
    "a_list_of_values": "c = list(ALIASED.values())\nc[0].append(1)",
    "a_loop_target": "for c in [ALIASED]:\n    c['rows'].append(1)",
    "an_unpacked_target": "c, k = ALIASED, 1\nc['rows'].append(1)",
    "a_with_target": "import contextlib\nwith contextlib.nullcontext(ALIASED) as c:\n    c['rows'].append(1)",
    "next_of_iter": "c = next(iter([ALIASED]))\nc['rows'].append(1)",
    "a_conditional": "c = ALIASED if self else None\nc['rows'].append(1)",
    "sorted_items_that_change": "c = sorted(NESTED_ROWS)\nc[0].append(1)",
    "a_comprehension_of_items_that_change": "c = [r for r in NESTED_ROWS]\nc[0].append(1)",
    "a_loop_over_items_that_change": "for c in NESTED_ROWS:\n    c.append(1)",
    "a_class_attribute": "c = K.seen\nc.append(1)",
    "a_module_attribute": "c = M.SEEN\nc.append(1)",
    "its_own_class_by_type": "c = type(self).seen\nc.append(1)",
    "its_own_class_by_dunder": "c = self.__class__.seen\nc.append(1)",
    "what_a_function_returns": "c = holder().seen\nc.append(1)",
}
#: The variable a body changes, where the body does not name it.
_HELD_IN = {
    "a_class_attribute": "K",
    "a_module_attribute": "M",
    "its_own_class_by_type": "Made",
    "its_own_class_by_dunder": "Made",
    "what_a_function_returns": "ALIASED_OBJECT",
}


@pytest.mark.parametrize("body", list(_BODIES))
def test_a_change_through_any_local_bound_from_a_global_counts(tmp_path, monkeypatch, body):
    """``c = CFG['rows']; c.append(1)``, ``for c in [CFG]``, ``c, k = CFG,
    1``: only a local bound to the global or to a recognised copy of it was
    followed, and the reducer, run on every pickle, appended where plain
    Python never pickles. So were ``K.seen`` (a class counted as never
    changed), ``M.SEEN`` (a module), ``type(self).seen`` and ``holder().seen`` (no global named)."""
    module = _module_with(tmp_path, monkeypatch, _BODIES[body])
    writer = reducer_writes._decide(module.Made)
    assert writer is not None, "not a writer"
    named = _HELD_IN.get(body) or next(n for n in ("ALIASED_OBJECT", "NESTED_ROWS", "ALIASED") if n in _BODIES[body])
    holder = getattr(module, named)
    assert reducer_writes.reach([writer], {"cfg": holder}) == {"cfg"}


def test_a_default_bound_to_a_global_counts(tmp_path, monkeypatch):
    module = _module_with(tmp_path, monkeypatch, "_c['rows'].append(1)", params=", _c=ALIASED")
    writer = reducer_writes._decide(module.Made)
    assert writer is not None and reducer_writes.reach([writer], {"cfg": module.ALIASED}) == {"cfg"}


@pytest.mark.parametrize("body", ["globals()['ALIASED']['rows'].append(1)", "vars(ALIASED_OBJECT)['seen'].append(1)"])
def test_one_reading_a_namespace_reaches_every_variable(tmp_path, monkeypatch, body):
    module = _module_with(tmp_path, monkeypatch, body)
    writer = reducer_writes._decide(module.Made)
    assert writer is not None and reducer_writes.reach([writer], {"x": 1}) is None


@pytest.mark.parametrize("body", ["acc.append(1)", "c = acc\nc.append(1)"], ids=["directly", "through_a_local"])
def test_a_closure_cell_the_notebook_holds_is_reached(tmp_path, monkeypatch, body):
    module = _module_with(tmp_path, monkeypatch, body, closure=True)
    writer = reducer_writes._decide(module.Made)
    assert writer is not None and reducer_writes.reach([writer], {"held": module.ACC}) == {"held"}


@pytest.mark.parametrize("body", ["c = copy.deepcopy(ALIASED)\nc['rows'].append(1)", "c = len(ALIASED)\nc += 1"])
def test_a_local_holding_nothing_of_a_global_is_no_writer(tmp_path, monkeypatch, body):
    module = _module_with(tmp_path, monkeypatch, body)
    assert reducer_writes._decide(module.Made) is None


_BODIES_THAT_CHANGE_ONLY_A_NEW_OBJECT = {
    "a_comprehension_filtered_by_a_global": (
        "state = {k: v for k, v in self.__dict__.items() if k not in SKIP}\nstate['version'] = 2"
    ),
    "a_loop_over_names": "state = {}\nfor name in NAMES:\n    state[name] = getattr(self, name, None)",
    "sorted_names": "order = sorted(NAMES)\norder.append('x')",
    "a_merge_of_dicts_of_atoms": "state = {**self.__dict__, **EXTRA}\nstate['v'] = 1",
    "a_list_of_names": "c = list(NAMES)\nc.append('z')",
    "a_comprehension_of_names": "rows = [r for r in NAMES]\nrows.append('x')",
    "a_dict_of_atoms_updated": "out = dict(DEFAULTS)\nout.update(self.__dict__)",
}


@pytest.mark.parametrize("body", list(_BODIES_THAT_CHANGE_ONLY_A_NEW_OBJECT))
def test_a_change_to_a_new_object_built_from_a_global_is_no_writer(tmp_path, monkeypatch, body):
    """A global read in a test (``k not in SKIP``), or numbers and strings
    taken out of one (``for name in NAMES``, ``sorted(NAMES)``, ``{**EXTRA}``):
    the local holds nothing a change can reach in the global, yet any global
    read counted, and the statement building the object never cached."""
    module = _module_with(tmp_path, monkeypatch, _BODIES_THAT_CHANGE_ONLY_A_NEW_OBJECT[body])
    assert reducer_writes._decide(module.Made) is None


def _module_with(tmp_path, monkeypatch, body: str, params: str = "", closure: bool = False):
    """A module file whose class ``Made`` has a ``__getstate__`` running
    *body*: its source is read from the file, as a module of the user's is."""
    import importlib
    import sys
    import textwrap

    name = f"_reducer_alias_{abs(hash((body, params, closure)))}"
    method = f"    def __getstate__(self{params}):\n" + textwrap.indent(body, " " * 8) + "\n        return {}\n"
    if closure:
        source = (
            "ACC = []\n"
            "def _make(acc):\n"
            "    class Made:\n" + textwrap.indent(method, "    ") + "    return Made\n"
            "Made = _make(ACC)\n"
        )
    else:
        source = "import copy\nclass Cfg:\n    def __init__(self):\n        self.seen = []\n"
        source += "ALIASED = {'rows': []}\nALIASED_OBJECT = Cfg()\nNESTED_ROWS = [[1]]\n"
        source += "SKIP = {'_lock'}\nNAMES = ['a', 'b']\nEXTRA = {'v': 0}\nDEFAULTS = {'x': 1}\n"
        source += "class K:\n    seen = []\ndef holder():\n    return ALIASED_OBJECT\n"
        source += "import types\nM = types.ModuleType('m')\nM.SEEN = []\n"
        source += "class Made:\n" + method
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


_BODIES_THAT_WRITE_ONCE_THE_GLOBAL_HOLDS_A_LIST = {
    "a_loop_over_it": ("for row in NAMES:\n    if isinstance(row, list):\n        row.append(1)", "NAMES"),
    "sorted": ("for row in sorted(NAMES, key=str):\n    if isinstance(row, list):\n        row.append(1)", "NAMES"),
    "a_merge": ("d = {**EXTRA}\nif 'rows' in d:\n    d['rows'].append(1)", "EXTRA"),
}


@pytest.mark.parametrize("body", list(_BODIES_THAT_WRITE_ONCE_THE_GLOBAL_HOLDS_A_LIST))
def test_a_global_that_gains_a_list_is_decided_again(tmp_path, monkeypatch, body):
    """``for row in NAMES: row.append(1)`` was decided writing nothing while
    ``NAMES`` held only strings, and the answer kept for the class: after
    ``NAMES.append([])`` the reducer changed it, unseen."""
    text, name = _BODIES_THAT_WRITE_ONCE_THE_GLOBAL_HOLDS_A_LIST[body]
    module = _module_with(tmp_path, monkeypatch, text)
    assert reducer_writes._writer(module.Made) is None
    held = getattr(module, name)
    if type(held) is dict:
        held["rows"] = []
    else:
        held.append([])
    writer = reducer_writes._writer(module.Made)
    assert writer is not None and name in reducer_writes.reach([writer], vars(module))
