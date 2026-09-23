"""A statement's entry points at call results already in the cache.

Without this, a project's cache held its expensive results twice. A statement
gathering per-call cached fits -- ``models = {k: fit(g) for k, g in groups}``,
``results[name] = evaluate(...)`` in a loop -- stored the whole dict, and each
fit was stored again under its call's key (about 1.2 of 2.7 GiB in one
project, 1.3 of 4 GiB in another). The statement's entry is still what lets a
later cell, or a kernel restart, get the value without rebuilding the call's
arguments; only its copy of the call results is redundant.

So when a statement is stored, a value that IS a call result this cell stored
or was served -- directly, or as an item of a plain dict, list or tuple --
is written as a :class:`CallRef` to that call's entry. Restoring the statement
reads the call entry back. Two things make that sound:

* The value must still be what the call returned. The SHA-256 of its pickle,
  taken when the call was stored, is compared with the live value's when the
  statement is: ``m = fit(g).set_params(...)`` stores the value, not a
  reference.
* The call entry must still hold that value. The hash rides in the call
  entry's metadata; a reference whose entry is gone or holds something else
  makes the statement a miss, which recomputes -- never a wrong value.

A version of the statement weighs the bytes it refers to when versions are
pruned, and the call entries only a pruned version referred to go with it.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Metadata fields on a call entry: the content hash of its value, and its
#: pickled size.
DIGEST_FIELD = "value_digest"
SIZE_FIELD = "value_bytes"
#: Set when ``SIZE_FIELD`` is an estimate and ``DIGEST_FIELD`` a one-off token,
#: the result not having been pickled to be digested: it is too big to be worth
#: its bytes, or it is the plain value of its statement (``a, b = build()``),
#: whose reference needs no digest. Such an entry is judged for disk on its
#: own (``TieredBackend``), and a statement refers to it only where nothing can
#: have changed the value since the call returned it (``with_call_refs``'s
#: *trusted*).
ESTIMATED_FIELD = "value_bytes_estimated"
UNHASHED_PREFIX = "unhashed:"

#: How deep into plain dicts, lists and tuples a reference is looked for.
_DEPTH = 2


@dataclass(frozen=True)
class CallRef:
    """A call entry's value, stored by reference in a statement's entry."""

    key: str
    digest: str
    #: The position in the call's returned tuple, for a name bound by
    #: unpacking it (``a, b = f()``); ``None`` for the whole value. A default,
    #: so a reference pickled before this field existed still loads.
    item: int | None = None


class _Missing(Exception):
    """A referenced call entry is gone or holds something else."""


def digest_of(value: Any) -> str | None:
    """SHA-256 of *value*'s pickle, or ``None`` when it does not pickle
    (see :func:`digest_and_size`)."""
    found = digest_and_size(value)
    return found[0] if found else None


def digest_and_size(value: Any) -> tuple[str, int] | None:
    """``(SHA-256 of value's pickle, pickled bytes)``, or ``None`` when it does
    not pickle.

    Every byte, since this is what decides a restore; over pickle protocol 5
    with its buffers hashed in place, which for a 64 MB frame is 24 ms --
    ``compute_hash_full`` took 115 ms, and a call's store pays this.
    """
    try:
        buffers: list = []
        head = pickle.dumps(value, protocol=5, buffer_callback=buffers.append)
        digest = hashlib.sha256(head)
        size = len(head)
        for buffer in buffers:
            raw = buffer.raw()
            digest.update(raw)
            size += raw.nbytes
        return digest.hexdigest(), size
    except Exception:  # noqa: BLE001 - no digest means no reference, never a failure
        return None


#: Statement metadata: the call keys its entry refers to, and their bytes --
#: what pruning one of its versions weighs, and what it may free.
REFS_FIELD = "call_refs"
REF_BYTES_FIELD = "call_ref_bytes"


def with_call_refs(
    variables: dict[str, Any],
    held: dict[int, tuple[Any, str, str, int]],
    referenced: dict[str, int] | None = None,
    *,
    trusted: tuple[str, int] | None = None,
    unpacked: dict[str, int] | None = None,
) -> dict[str, Any]:
    """*variables* with each value that is an unchanged held call result
    replaced by a :class:`CallRef`. *held* maps ``id(result)`` to
    ``(result, call key, digest, pickled bytes)``; *referenced*, when given,
    collects ``{call key: bytes}`` for the references made.

    "Unchanged" is proved by digesting the value again -- a second pickle of
    all of it -- except for *trusted*, ``(call key, id(result))`` of a result
    nothing can have changed since the call returned it: the statement is
    ``names = call(...)`` and that call returned last. Then the reference is
    made without a digest, and *unpacked* (``{name: position}``, for
    ``a, b = call(...)``) refers each name to its item of the result. A result
    whose digest is a one-off token (`UNHASHED_PREFIX`) can only be referred
    to that way.
    """
    if not held:
        return variables
    verified: dict[int, bool] = {}

    def ref_for(value):
        entry = held.get(id(value))
        if entry is None or entry[0] is not value:
            return None
        if trusted != (entry[1], id(value)):
            if str(entry[2]).startswith(UNHASHED_PREFIX):
                return None
            ok = verified.get(id(value))
            if ok is None:
                ok = verified[id(value)] = digest_of(value) == entry[2]
            if not ok:
                return None
        if referenced is not None:
            referenced[entry[1]] = entry[3]
        return CallRef(entry[1], entry[2])

    if trusted and unpacked:
        entry = held.get(trusted[1])
        result = entry[0] if entry is not None and entry[1] == trusted[0] else None
        if type(result) in (tuple, list):
            refs = {}
            for name, position in unpacked.items():
                if name in variables and position < len(result) and variables[name] is result[position]:
                    refs[name] = CallRef(entry[1], entry[2], item=position)
            if refs:
                if referenced is not None:
                    referenced[entry[1]] = entry[3]
                variables = {**variables, **refs}

    def swap(value, depth):
        ref = ref_for(value)
        if ref is not None:
            return ref
        if depth <= 0:
            return value
        kind = type(value)
        if kind is dict:
            items = {k: swap(v, depth - 1) for k, v in value.items()}
            return items if any(items[k] is not value[k] for k in value) else value
        if kind in (list, tuple):
            items = [swap(v, depth - 1) for v in value]
            return kind(items) if any(a is not b for a, b in zip(items, value, strict=True)) else value
        return value

    try:
        return {name: swap(value, _DEPTH) for name, value in variables.items()}
    except Exception:  # noqa: BLE001 - a reference is an optimisation; store the values
        logger.debug("call refs: could not build references", exc_info=True)
        if referenced is not None:
            referenced.clear()
        return variables


def has_call_refs(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("variables"), dict):
        return False

    def found(value, depth):
        if isinstance(value, CallRef):
            return True
        if depth <= 0:
            return False
        if type(value) is dict:
            return any(found(v, depth - 1) for v in value.values())
        if type(value) in (list, tuple):
            return any(found(v, depth - 1) for v in value)
        return False

    return any(found(v, _DEPTH) for v in payload["variables"].values())


def resolve_call_refs(payload: Any, backend: Any) -> Any:
    """*payload* with its references read back from *backend*, or ``None`` when
    one cannot be -- the statement is then a miss."""
    if not has_call_refs(payload):
        return payload
    loaded: dict[str, Any] = {}

    def load(ref: CallRef):
        if ref.key not in loaded:
            try:
                metadata, value = backend.get(ref.key)
            except Exception as exc:  # noqa: BLE001
                raise _Missing(ref.key) from exc
            if not isinstance(metadata, dict) or metadata.get(DIGEST_FIELD) != ref.digest:
                raise _Missing(ref.key)
            loaded[ref.key] = value
        return loaded[ref.key]

    def swap(value, depth):
        if isinstance(value, CallRef):
            loaded_value = load(value)
            item = getattr(value, "item", None)
            if item is None:
                return loaded_value
            try:
                return loaded_value[item]
            except (TypeError, IndexError, KeyError) as exc:
                raise _Missing(value.key) from exc
        if depth <= 0:
            return value
        if type(value) is dict:
            return {k: swap(v, depth - 1) for k, v in value.items()}
        if type(value) in (list, tuple):
            return type(value)(swap(v, depth - 1) for v in value)
        return value

    try:
        variables = {name: swap(value, _DEPTH) for name, value in payload["variables"].items()}
    except _Missing as missing:
        logger.debug("call refs: entry %s is gone; the statement recomputes", missing)
        return None
    return {**payload, "variables": variables}
