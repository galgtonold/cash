"""The ``functools`` memos of the user's that a statement used, checked before a hit.

A helper under ``@functools.cache`` / ``@functools.lru_cache`` keeps what it
returned. A statement served from the cache does not run, so the memo stays
as it was: empty after a kernel restart. A run leaves it filled, so a call of
the helper below gets the kept answer and its body does not run. Served, the
body runs there instead, and what it does besides return happens where the
run never did it: ``random.seed(n)`` inside it resets the stream a draw
below reads, so the draw differs from the run's.

So an entry records, for each such memo the statement called, how many
answers it held after the run and the digests of their keys (`watched`,
`used`), and a lookup serves it only while the memo holds those keys again
(`unfilled`); otherwise the statement runs. A memo filled by a module dict
of the user's is a write to that dict, which the in-place check already sees.

What is looked at is bounded: memos bound in the namespace and at the top of
the user's modules (`_LRU_WRAPPER` only: the pure-Python fallback is a plain
function), their ``cache_info()`` counters, and the keys of a memo holding at
most `MAX_KEYED` answers. A larger one is checked by its count alone.
"""

from __future__ import annotations

import functools
import gc
import logging
import sys
import weakref
from collections.abc import Mapping
from typing import Any

from cash.install_paths import is_user_code_file, is_user_module
from cash.value_hash import HASH_ERRORS, compute_hash

__all__ = ["MAX_KEYED", "unfilled", "used", "watched"]

logger = logging.getLogger(__name__)

#: The type ``functools.cache`` and ``functools.lru_cache`` return.
_LRU_WRAPPER = getattr(functools, "_lru_cache_wrapper", None)

#: A memo holding more answers than this is checked by its count, not its keys.
MAX_KEYED = 256

#: ``module -> (len of its namespace, its memo names)``: a module's top
#: level is read again only when names were added to or taken from it.
_module_memos: weakref.WeakKeyDictionary[Any, tuple[int, tuple[str, ...]]] = weakref.WeakKeyDictionary()
#: ``(len(sys.modules), the user's modules)``: judged again only when a
#: module was imported or dropped.
_user_modules: tuple[int, tuple[Any, ...]] = (-1, ())
#: ``memo -> ((answers held, misses), key digests)``: every new answer is
#: a miss, so a memo whose two counters stand still holds the same keys.
_digests: weakref.WeakKeyDictionary[Any, tuple[tuple[int, int], list[str] | None]] = weakref.WeakKeyDictionary()


def _is_user_memo(value: Any) -> bool:
    if type(value) is not _LRU_WRAPPER:
        return False
    code = getattr(getattr(value, "__wrapped__", None), "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


def _memos_of_module(mod: Any) -> list[tuple[str, Any]]:
    try:
        namespace = vars(mod)
    except TypeError:
        return []
    try:
        known = _module_memos.get(mod)
    except TypeError:  # a module object that takes no weak reference
        known = None
    if known is None or known[0] != len(namespace):
        names = tuple(name for name, value in list(namespace.items()) if _is_user_memo(value))
        known = (len(namespace), names)
        try:
            _module_memos[mod] = known
        except TypeError:
            pass
    return [(name, namespace[name]) for name in known[1] if name in namespace]


def _where(ns: Mapping[str, Any]) -> list[tuple[list[str], Any]]:
    """``(where, memo)`` for every memo of the user's the statement can reach:
    ``["ns", name]`` or ``["module", module, name]``."""
    found = [(["ns", name], value) for name, value in list(ns.items()) if _is_user_memo(value)]
    for mod in _modules_of_user():
        found.extend((["module", mod.__name__, attr], value) for attr, value in _memos_of_module(mod))
    return found


def _modules_of_user() -> tuple[Any, ...]:
    global _user_modules
    if _user_modules[0] == len(sys.modules):
        return _user_modules[1]
    mods = []
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", None)
        if not isinstance(name, str) or name == "__main__":
            continue  # the notebook's own namespace is looked at as such
        try:
            if is_user_module(mod):
                mods.append(mod)
        except Exception:  # noqa: BLE001 - an odd entry in sys.modules
            continue
    _user_modules = (len(sys.modules), tuple(mods))
    return _user_modules[1]


def watched(ns: Mapping[str, Any]) -> list[tuple[list[str], Any, int]] | None:
    """Before a statement runs: each memo it can reach with its call count."""
    if _LRU_WRAPPER is None:
        return None
    out = []
    for where, memo in _where(ns):
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        out.append((where, memo, info.hits + info.misses))
    return out or None


def _cache_dict(memo: Any) -> dict | None:
    """The dict a C ``lru_cache`` keeps its answers in, which only the
    collector's view of it reaches."""
    own = getattr(memo, "__dict__", None)
    for ref in gc.get_referents(memo):
        if type(ref) is dict and ref is not own:
            return ref
    return None


def _key_digest(key: Any) -> str:
    """A digest of one memo key, the same in every process: the marker
    ``functools`` puts between positional and keyword arguments is a bare
    ``object()``, keyed by what it is, not where it lives."""
    if type(key) is not int and type(key) is not str and isinstance(key, (list, tuple)):
        key = tuple("<kwd_mark>" if type(part) is object else part for part in key)
    return compute_hash(key)


def _key_digests(memo: Any) -> list[str] | None:
    info = memo.cache_info()
    stamp = (info.currsize, info.misses)
    known = _digests.get(memo)
    if known is not None and known[0] == stamp:
        return known[1]
    digests = None
    cache = _cache_dict(memo) if info.currsize <= MAX_KEYED else None
    if cache is not None and len(cache) <= MAX_KEYED:
        try:
            digests = sorted({_key_digest(key) for key in list(cache)})
        except HASH_ERRORS:
            digests = None
    _digests[memo] = (stamp, digests)
    return digests


def used(before: list[tuple[list[str], Any, int]] | None) -> list[dict] | None:
    """After a statement ran: each memo it called, with what it held then."""
    if not before:
        return None
    out = []
    for where, memo, calls in before:
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001
            continue
        if info.hits + info.misses == calls:
            continue
        record: dict[str, Any] = {"where": where, "size": info.currsize}
        keys = _key_digests(memo)
        if keys is not None:
            record["keys"] = keys
        out.append(record)
    return out or None


def _resolve(where: list[str], ns: Mapping[str, Any]) -> Any:
    if where[0] == "ns":
        return ns.get(where[1])
    mod = sys.modules.get(where[1])
    return getattr(mod, where[2], None) if mod is not None else None


def unfilled(records: list[dict] | None, ns: Mapping[str, Any]) -> str | None:
    """Why an entry cannot be served over the memos it used, or None.

    Each memo must hold at least the answers it held after the run, and the
    same keys where they were recorded."""
    if not records:
        return None
    for record in records:
        where = record.get("where") or []
        memo = _resolve(where, ns) if where else None
        label = where[-1] if where else "?"
        if not _is_user_memo(memo):
            return f"the helper memo {label} it used is not bound"
        try:
            size = memo.cache_info().currsize
        except Exception:  # noqa: BLE001
            return f"the helper memo {label} cannot be read"
        if size < int(record.get("size") or 0):
            return f"the helper memo {label} holds fewer answers than the run left"
        keys = record.get("keys")
        if keys:
            now = _key_digests(memo)
            if now is None or not set(keys) <= set(now):
                return f"the helper memo {label} does not hold the answers the run left"
    return None
