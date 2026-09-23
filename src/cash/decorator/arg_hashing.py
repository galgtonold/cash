"""How a call's arguments become the args segment of its key."""

from __future__ import annotations

import functools
import hashlib
import threading
import types
from typing import Any

from .. import _plain_data
from ..value_types import PLAIN_SEQS

#: A census taken while one cache key is built, shared by the code fold and the
#: argument hash so a big argument is looked at once (`plain_census`). None
#: outside a key build: after the body has run, an argument may have changed.
PLAIN_CENSUS = threading.local()


def plain_census(value: Any) -> tuple[str, Any] | None:
    """What kind of plain data *value* is, memoized for the key build in progress.

    ``("plain", value)`` for lists and tuples of primitives (`_plain_data.is_plain`),
    ``("dict_rows", (keys, rows))`` for a list of dicts sharing their keys
    (`_plain_data.dict_rows`), None for anything else.
    """
    memo = getattr(PLAIN_CENSUS, "memo", None)
    if memo is not None:
        hit = memo.get(id(value))
        if hit is not None and hit[0] is value:
            return hit[1]
    found: tuple[str, Any] | None = None
    if _plain_data.is_plain(value):
        found = ("plain", value)
    else:
        rows = _plain_data.dict_rows(value)
        if rows is not None:
            found = ("dict_rows", rows)
    if memo is not None:
        memo[id(value)] = (value, found)
    return found


def plain_key_part(value: Any) -> Any:
    """*value*, or -- for plain data -- a marker holding the digest of its content.

    Each plain argument is keyed by its content on its own, pickled without the
    memo (`_plain_data.pickle_unshared`). It used to take the fast path only
    when EVERY argument did: one small dict beside two million rows sent the
    whole call down the general path, 8x the cost (round 20).
    """
    if type(value) not in PLAIN_SEQS:
        return value
    census = plain_census(value)
    if census is None:
        return value
    kind, data = census
    return (f"__cash_{kind}__", hashlib.sha256(_plain_data.pickle_unshared(data)).hexdigest())


#: Values whose identity is code plus what it captures. A hasher registered for
#: one of these types covers every such value in the process, and the obvious
#: one -- by name -- gives every closure one factory makes the same identity.
#: `Cash._first_unhashable_arg` found only built-in-typed arguments.
NO_SUSPECT = object()


CODE_VALUE_TYPES = (types.FunctionType, types.MethodType, functools.partial)


#: The fix for an unhashable code value. It must NOT suggest
#: `register_hasher(function, ...)`: following that advice is how a second
#: closure got the first one's result.
CODE_ARG_FIX = (
    "pass a module-level function in its place, and give the values it "
    "captures to the cached function as plain arguments, where they reach the "
    "key. Do not register a hasher for function: every closure one factory "
    "makes shares a name, so a hasher keyed on it hands one closure's result "
    "to another. See known-limitations.md, 'A closure or lambda passed as an "
    "argument'."
)


def unhashable_arg_fix(value: Any, type_name: str) -> str:
    """The fix line for an argument of *type_name* that could not be hashed."""
    if isinstance(value, CODE_VALUE_TYPES):
        return CODE_ARG_FIX
    return f"register a hasher with cash.register_hasher({type_name}, ...), or pass the argument by a hashable value."


#: Who wrote a value's ``_cash_lineage_hash``, in ``_cash_lineage_src``. Only the
#: notebook's statement layer keeps the tag current as the value changes, so
#: only its tag stands in for the value's content (see `_hash_arg_payload`).
LINEAGE_SRC_STATEMENT = "statement"
LINEAGE_SRC_DECORATOR = "decorator"
#: Written for a function decorated ``frozen=True``: the user's promise that the
#: result is not modified afterwards, trusted like the statement layer's tag and
#: audited now and then (`_audit_frozen`).
LINEAGE_SRC_FROZEN = "frozen"


_COW_PANDAS: bool | None = None


#: The costliest argument of the key most recently hashed on this thread:
#: ``(label, seconds, type name, producer, pandas without copy-on-write)``.
#: A description, never the value: a reference here would keep a large
#: argument alive after its caller dropped it.
ARG_COST = threading.local()


def is_cow_pandas(value: Any) -> bool:
    """Is *value* a pandas DataFrame/Series under copy-on-write?

    Copy-on-write is the only mode in pandas 3 and opt-in before. Checked
    without importing pandas: a pandas object means it is already loaded.
    """
    global _COW_PANDAS
    t = type(value)
    if t.__name__ not in ("DataFrame", "Series") or not (t.__module__ or "").startswith("pandas"):
        return False
    if _COW_PANDAS is None:
        try:
            import pandas as pd

            major = int(pd.__version__.split(".", 1)[0])
            _COW_PANDAS = major >= 3 or pd.options.mode.copy_on_write is True
        except Exception:  # noqa: BLE001 - unknown pandas: no memo, hash every time
            _COW_PANDAS = False
    return _COW_PANDAS
