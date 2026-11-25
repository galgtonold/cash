"""How long each module-level memo of the notebook path stays true.

Every dict, list, set or memo object a ``cash.notebook`` module keeps at its
top level is listed in `LIFETIMES` with its scope (`cash._memo_registry`
says what each scope means), and registered there, so ``cash.reset_session()``
clears the ones a session owns. A test fails for a container that is not
listed (``tests/test_notebook/internals/test_memo_lifetimes.py``): a new
memo has to say how long it lives.

The containers are found by name when a reset runs, in the modules loaded
by then: this module imports none of them.
"""

from __future__ import annotations

import sys

from cash._memo_registry import CLEARED_SCOPES, register_memo

__all__ = ["LIFETIMES"]

_P, _S, _C, _ST, _B = "process", "session", "cell", "statement", "block"

#: ``module -> {name: scope}``. Process: keyed by code objects, types,
#: modules or live objects (each guarded against a reused ``id``), or a
#: registration made once. Session: keyed by lineages or cache keys, or
#: what a session was warned of once.
LIFETIMES: dict[str, dict[str, str]] = {
    "cash.notebook.borrowed_memory": {"_FAMILIES": _P, "_UNREADABLE": _P},
    "cash.notebook.cache_key": {
        "_GLOBAL_NAMES_MEMO": _P,
        "_IMPORTED_BY": _P,
        "_BODY_IMPORTS": _S,
        "_DRAW_KEYS": _S,
    },
    "cash.notebook.call_key": {
        "_GLOBAL_MUTATION_CACHE": _P,
        "_CODE_NAMES": _P,
        "_PARAM_ONLY_READ": _P,
        "_OWN_DEFS": _P,
        "_NUMPY": _P,
        "_USERS_TYPES": _P,
        "_CLASS_MEMOS": _P,
    },
    "cash.notebook.call_unit": {"_WARNING_REGISTRIES": _S},
    "cash.notebook.callee_reach": {
        "_WALKS": _B,
        "_SET_ON_LIBRARIES": _P,
        "_UNCHECKED_WARNED": _S,
        "_FOREIGN_C_TYPES": _P,
        "_NOTEBOOK": _B,
    },
    "cash.notebook.compiled_source": {"_PLACE": _B},
    "cash.notebook.file_records": {"_OWN": _S, "_MERGED": _S},
    "cash.notebook.held_functions": {
        "_MEMO": _S,
        "_BY_LINEAGE": _S,
        "_SHARED": _ST,
        "_CASH_MEMOS": _P,
        "_OBJECTS_SEEN": _P,
        "_USERS_MODULES": _P,
        "_USERS_MODULE": _P,
        "_USERS_CLASS": _P,
        "_NO_FUNCTIONS": _P,
    },
    "cash.notebook.held_modules": {"_IDENTITY": _P},
    # Each keyed weakly by a memo of the user's, and read again when the
    # memo is not as it was left.
    "cash.notebook.helper_memos": {
        "_owner_memos": _P,
        "_digests": _P,
        "_answers_seen": _P,
        "_ns_reached": _P,
        "_hidden": _P,
        "_key_digest_memo": _P,
        "_MARK": _P,
    },
    "cash.notebook.installed_code": {"_STATEMENT_ATTRIBUTES": _P, "_USER_CODE": _P},
    "cash.notebook.ipython.autoreload_hook": {"_LOCAL": _P, "_BY_KEY": _P},
    "cash.notebook.lineage_formula": {
        "_SETTING_RUNS": _S,
        # Read again from the new session's store (`use_lineage_alias_store`).
        "_LINEAGE_ALIASES": _S,
        "_FORWARD_HELPERS": _S,
        "_FORWARD_NAMES": _S,
        "_FUNCTION_IDENTITIES": _S,
        "_UNREAD_ROUTINES_WARNED": _S,
    },
    "cash.notebook.recorded_reads": {"_READING": _B, "_IMMUTABLE_DIGESTS": _P},
    "cash.notebook.run_memo": {
        "_FRESH_ENTRY_VERDICTS": _C,
        "_FILE_STATE_THIS_RUN": _C,
        "_COVERED_SNAPSHOTS": _S,
        "_HELD_KEYS": _S,
        "_REFUSED_KEYS": _S,
        "_SEEN_COMPONENTS": _S,
        "_COVERED_COMPONENTS": _S,
    },
    "cash.notebook.shared_objects": {
        "_OWN_NAMES_BY_TYPE": _P,
        "_SCALAR_HOOKS": _P,
        "_EXACT_WITH_LIBRARIES": _P,
        "_LEAVES_WITH_LIBRARIES": _P,
        "_ESTIMATOR_CLASSES": _P,
        "_LOCAL_MODULE_CLASSES": _P,
        "_SLOT_MEMBERS": _P,
        "_LIBRARY_KINDS": _P,
        "_REPAIR_SNAPSHOTS": _B,
        "_DEADLINE": _B,
    },
    "cash.notebook.statement.derivation_edges": {"_EDGE_LOGS": _B},
    "cash.notebook.statement.held_generators": {"_METHOD_CHAINS": _P},
    "cash.notebook.statement.identity_holders": {
        "_FLAT_READS": _C,
        # Each entry is checked against a witness of the object it was read
        # from, and goes with that object.
        "_SHAPE_READS": _P,
        "_ARROW_READS": _P,
        "_ARROW_SIGNED": _P,
        "_MAPPINGS": _P,
    },
    "cash.notebook.statement.in_place": {"_REFUSED": _P, "_GC_READS": _P},
    "cash.notebook.statement.processor": {"_LOGGED_ONCE": _S},
    "cash.notebook.statement.randomness": {"_MINTS": _P},
    "cash.notebook.statement.reducer_writes": {
        "_NAMESPACE": _B,
        "_REFUSES": _ST,
        "_REFUSED": _P,
        "_WRITERS": _P,
        "_CLEARED": _B,
    },
    # The differences a verify run found: the tests read and empty it.
    "cash.notebook.upstream.virtual_lineage": {"VERIFY_FAILURES": _P},
    "cash.notebook.vscode_backup": {"_answers": _P, "_headers": _P},
    "cash.notebook.write_observer": {"_frame_verdicts": _P},
}


def _clearing(module: str, name: str):
    def clear() -> None:
        loaded = sys.modules.get(module)
        memo = None if loaded is None else vars(loaded).get(name)
        if memo is not None:
            memo.clear()

    return clear


for _module, _names in LIFETIMES.items():
    for _name, _scope in _names.items():
        register_memo(f"{_module}.{_name}", _clearing(_module, _name) if _scope in CLEARED_SCOPES else None, _scope)
