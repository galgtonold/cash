"""The ``functools`` memos of the user's that a statement used, checked before a hit.

A helper under ``@functools.cache`` / ``@functools.lru_cache`` keeps what it
returned. A statement served from the cache does not run, so the memo stays
as it was: empty after a kernel restart. A run leaves it filled, so a call of
the helper below gets the kept answer and its body does not run. Served, the
body runs there instead, and what it does besides return happens where the
run never did it: ``random.seed(n)`` inside it resets the stream a draw
below reads, so the draw differs from the run's. And the other way round: an
entry stored when the helper's body ran, served where the memo already holds
the answer, replays what the body printed where a run prints nothing.

So a statement whose run made a memo compute (its miss count moved) changes
the memo the way it changes a list it appends to: it is not stored
(`used` says so), and it runs every time. One that only read answers already
there records, for each memo it called, the digest of every key and answer
the memo held (`used`), and a lookup serves it only while the memo holds
them all again (`unfilled`). A memo whose answers cannot be read here (more
than `MAX_KEYED` of them, a layout this module does not recognise, a key or
answer with no digest) is recorded as such and never lets the entry be
served. A memo filled in a module dict of the user's is a write to that
dict, which the in-place check already sees.

The memos looked for: ``functools``' C wrapper (`_LRU_WRAPPER`; the
pure-Python fallback, used only without the ``_functools`` module, is a plain
function this cannot read), bound in the namespace or at the top of one of
the user's modules, under another decorator (followed through
``__wrapped__``), or held by one of the user's classes (a ``staticmethod``,
a ``classmethod`` or a plain attribute).
"""

from __future__ import annotations

import functools
import gc
import inspect
import logging
import sys
import weakref
from collections.abc import Iterator, Mapping
from typing import Any

from cash.install_paths import is_user_code_file, is_user_module
from cash.value_hash import compute_hash

__all__ = ["FILLED", "MAX_KEYED", "unfilled", "used", "watched"]

logger = logging.getLogger(__name__)

#: The type ``functools.cache`` and ``functools.lru_cache`` return.
_LRU_WRAPPER = getattr(functools, "_lru_cache_wrapper", None)

#: A memo holding more answers than this is not read: an entry over it is
#: never served.
MAX_KEYED = 256

#: How many ``__wrapped__`` links are followed to find a memo under other
#: decorators.
_MAX_WRAPPED = 4

#: What `used` returns, as the first item, for a run that made a memo compute.
FILLED = "filled"

#: ``(len(sys.modules), the last module's name)`` and the user's modules then:
#: judged again when a module was imported or dropped (an import puts its
#: name last, so one dropped and one imported still change the stamp).
_user_modules: tuple[Any, tuple[Any, ...]] = (None, ())
#: ``owner -> (len of its namespace, [(path, memo)])``: an owner's names are
#: read again only when names were added to or taken from it.
_owner_memos: weakref.WeakKeyDictionary[Any, tuple[int, list[tuple[list[str], Any]]]] = weakref.WeakKeyDictionary()
#: ``memo -> (the keys and answers it held, their digests)``: digested again
#: only when one of them is not the very object it was.
_digests: weakref.WeakKeyDictionary[Any, tuple[list[Any], list[str] | None]] = weakref.WeakKeyDictionary()


def _is_user_memo(value: Any) -> bool:
    if _LRU_WRAPPER is None or type(value) is not _LRU_WRAPPER:
        return False
    code = getattr(getattr(value, "__wrapped__", None), "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


def _memos_in(value: Any, path: list[str], seen: set[int]) -> Iterator[tuple[list[str], Any]]:
    """``(path, memo)`` for the memos *value* is or holds: itself, the one
    under its decorators (``~`` steps), or the ones its class body holds
    (``.name`` steps)."""
    for _ in range(_MAX_WRAPPED + 1):
        if _is_user_memo(value):
            yield path, value
            return
        inner = getattr(value, "__wrapped__", None) if inspect.isfunction(value) else None
        if inner is None:
            break
        value, path = inner, [*path, "~"]
    if isinstance(value, type) and id(value) not in seen:
        seen.add(id(value))
        for steps, memo in _class_memos(value, seen):
            yield [*path, *steps], memo


def _class_memos(cls: type, seen: set[int]) -> list[tuple[list[str], Any]]:
    """The memos the body of a class holds, remembered per class until names
    are added to or taken from it. Any class: one made in a cell names the
    notebook as its module, which `is_user_module` does not know, and a
    memo is the user's by its own code (`_is_user_memo`)."""
    namespace = vars(cls)
    try:
        known = _owner_memos.get(cls)
    except TypeError:
        known = None
    if known is not None and known[0] == len(namespace):
        return known[1]
    found: list[tuple[list[str], Any]] = []
    for attr, raw in list(namespace.items()):
        held = getattr(raw, "__func__", raw) if isinstance(raw, (staticmethod, classmethod)) else raw
        if inspect.isfunction(held) or _is_user_memo(held):
            found.extend(_memos_in(held, ["." + attr], seen))
    try:
        _owner_memos[cls] = (len(namespace), found)
    except TypeError:
        pass
    return found


def _owner_memo_paths(owner: Any, namespace: Mapping[str, Any]) -> list[tuple[list[str], Any]]:
    """The memos at the top of *namespace* (a module's), remembered per owner."""
    try:
        known = _owner_memos.get(owner)
    except TypeError:  # an owner that takes no weak reference
        known = None
    if known is not None and known[0] == len(namespace):
        return known[1]
    seen: set[int] = set()
    found = [hit for name, value in list(namespace.items()) for hit in _memos_in(value, [name], seen)]
    try:
        _owner_memos[owner] = (len(namespace), found)
    except TypeError:
        pass
    return found


def _modules_of_user() -> tuple[Any, ...]:
    global _user_modules
    stamp = (len(sys.modules), next(reversed(sys.modules), None))
    if _user_modules[0] == stamp:
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
    _user_modules = (stamp, tuple(mods))
    return _user_modules[1]


def _where(ns: Mapping[str, Any]) -> list[tuple[list[str], Any]]:
    """``(where, memo)`` for every memo of the user's a statement can reach:
    ``["ns", name, *steps]`` or ``["module", module, name, *steps]``."""
    seen: set[int] = set()
    found = [(["ns", *path], memo) for name, value in list(ns.items()) for path, memo in _memos_in(value, [name], seen)]
    for mod in _modules_of_user():
        try:
            namespace = vars(mod)
        except TypeError:
            continue
        found.extend((["module", mod.__name__, *path], memo) for path, memo in _owner_memo_paths(mod, namespace))
    return found


def watched(ns: Mapping[str, Any]) -> list[tuple[list[str], Any, int, int]] | None:
    """Before a statement runs: each memo it can reach, with its counters."""
    if _LRU_WRAPPER is None:
        return None
    out = []
    for where, memo in _where(ns):
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        out.append((where, memo, info.hits, info.misses))
    return out or None


def _items(memo: Any) -> list[tuple[Any, Any]] | None:
    """The ``(key, answer)`` pairs a C ``lru_cache`` holds, which only the
    collector's view of it reaches, or None when its layout is not the one
    this expects.

    The wrapper visits its own type first (3.12 on). Unbounded, the answers
    dict comes next. Bounded (``maxsize=N``), it first visits each node of
    its list in order -- the node's key, its answer, its type (3.12 on) --
    then the dict mapping each key to its node, which is taken only when it
    holds exactly the keys visited.
    """
    try:
        size = memo.cache_info().currsize
        maxsize = memo.cache_parameters().get("maxsize")
        refs = gc.get_referents(memo)
    except Exception:  # noqa: BLE001 - a wrapper that cannot be read
        return None
    if refs and refs[0] is type(memo):
        refs = refs[1:]
    if maxsize is None:
        cache = refs[0] if refs else None
        if type(cache) is not dict or len(cache) != size:
            return None
        return list(cache.items())
    typed = len(refs) > 2 and isinstance(refs[2], type) and refs[2].__name__ == "_lru_list_elem"
    step = 3 if typed else 2
    nodes = refs[: step * size]
    cache = refs[step * size] if len(refs) > step * size else None
    if type(cache) is not dict or len(cache) != size or len(nodes) != step * size:
        return None
    keys, answers = nodes[0::step], nodes[1::step]
    try:
        if any(type(node).__name__ != "_lru_list_elem" for node in cache.values()):
            return None
        if {id(k) for k in keys} != {id(k) for k in cache}:
            return None
    except Exception:  # noqa: BLE001 - a key whose comparison raises
        return None
    return list(zip(keys, answers))


def _key_digest(key: Any) -> str:
    """A digest of one memo key, the same in every process: the marker
    ``functools`` puts between positional and keyword arguments is a bare
    ``object()``, keyed by what it is, not where it lives."""
    if isinstance(key, (list, tuple)):
        key = tuple("<kwd_mark>" if type(part) is object else part for part in key)
    return compute_hash(key)


def _item_digests(memo: Any) -> list[str] | None:
    """``key-digest:answer-digest`` for every answer *memo* holds, or None
    when they cannot be read here."""
    items = _items(memo)
    if items is None or len(items) > MAX_KEYED:
        logger.debug("[MEMO] the answers of %r are not read: never served over it", memo)
        return None
    flat = [part for item in items for part in item]
    known = _digests.get(memo)
    if known is not None and len(known[0]) == len(flat) and all(a is b for a, b in zip(known[0], flat)):
        return known[1]
    try:
        digests: list[str] | None = sorted({f"{_key_digest(k)}:{compute_hash(v)}" for k, v in items})
    except Exception:  # noqa: BLE001 - a key or answer with no digest: unreadable
        digests = None
    _digests[memo] = (flat, digests)
    return digests


def used(before: list[tuple[list[str], Any, int, int]] | None) -> tuple[str, str] | list[dict] | None:
    """After a statement ran: ``(FILLED, name)`` when it made a memo compute,
    else each memo it called with the answers it held then."""
    if not before:
        return None
    out = []
    for where, memo, hits, misses in before:
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        if info.misses != misses:
            return FILLED, ".".join(step for step in where[1:] if step != "~")
        if info.hits == hits:
            continue
        digests = _item_digests(memo)
        out.append({"where": where, "items": digests} if digests is not None else {"where": where, "unread": True})
    return out or None


def _resolve(where: list[str], ns: Mapping[str, Any]) -> Any:
    if where[0] == "ns":
        value, steps = ns.get(where[1]), where[2:]
    else:
        mod = sys.modules.get(where[1])
        value = getattr(mod, where[2], None) if mod is not None else None
        steps = where[3:]
    for step in steps:
        if value is None:
            return None
        if step == "~":
            value = getattr(value, "__wrapped__", None)
            continue
        raw = vars(value).get(step[1:]) if isinstance(value, type) else getattr(value, step[1:], None)
        value = getattr(raw, "__func__", raw) if isinstance(raw, (staticmethod, classmethod)) else raw
    return value


def unfilled(records: list[dict] | None, ns: Mapping[str, Any]) -> str | None:
    """Why an entry cannot be served over the memos it read, or None: each
    must hold every key and answer it held when the statement ran."""
    if not records:
        return None
    for record in records:
        where = record.get("where") or []
        label = ".".join(step for step in where[1:] if step != "~") or "?"
        memo = _resolve(where, ns) if where else None
        if not _is_user_memo(memo):
            return f"the helper memo {label} it read is not bound"
        if record.get("unread"):
            return f"the helper memo {label} it read holds more than cash checks"
        now = _item_digests(memo)
        if now is None or not set(record.get("items") or ()) <= set(now):
            return f"the helper memo {label} does not hold the answers the statement read"
    return None
