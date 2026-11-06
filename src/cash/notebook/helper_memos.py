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

import contextlib
import functools
import hashlib
import itertools
import logging
import pickle
import sys
import types
import weakref
from collections.abc import Iterator, Mapping
from typing import Any

from cash._referents import get_referents
from cash._user_attrs import attribute_or, instance_of
from cash.install_paths import is_user_code_file, is_user_module
from cash.value_hash import compute_hash

__all__ = [
    "FILLED",
    "MAX_KEYED",
    "Clears",
    "bodies_run",
    "carries_changes",
    "change_answers",
    "change_holders",
    "class_memo_record",
    "held_by_cash",
    "lifted",
    "misses_before",
    "new_statement",
    "put_back",
    "reads_counters",
    "unfilled",
    "used",
    "watched",
    "watching_clears",
]

logger = logging.getLogger(__name__)

#: The type ``functools.cache`` and ``functools.lru_cache`` return.
_LRU_WRAPPER = attribute_or(functools, "_lru_cache_wrapper", None)

#: A memo holding more answers than this is not read: an entry over it is
#: never served, and the statement says so once (``NOTEBOOK-MEMO-UNREAD``).
#: Reading one costs about 55 us an answer the first time (0.2 s at the
#: limit), 2 ms when its answers are unchanged, and 32 characters an answer
#: in each entry that read it.
MAX_KEYED = 4096

#: How many ``__wrapped__`` links are followed to find a memo under other
#: decorators.
_MAX_WRAPPED = 4

#: What `used` returns, as the first item, for a run that made a memo compute.
FILLED = "filled"

#: ``(len(sys.modules), the last module's name)`` and the user's modules then:
#: judged again when a module was imported or dropped (an import puts its
#: name last, so one dropped and one imported still change the stamp).
_user_modules: tuple[Any, tuple[Any, ...]] = (None, ())
#: ``owner -> (the ids of what its namespace binds that can hold a memo,
#: [(path, memo)])``: an owner's names are read again when one of those was
#: bound, rebound or dropped (``importlib.reload``, ``C.f = cache(g)``).
_owner_memos: weakref.WeakKeyDictionary[Any, tuple[tuple[int, ...], list[tuple[list[str], Any]]]] = (
    weakref.WeakKeyDictionary()
)
#: ``memo -> (the keys and answers it held, their digests)``: digested again
#: when one of them is not the very object it was, or is one that can be
#: changed in place (`_fixed`).
_digests: weakref.WeakKeyDictionary[Any, tuple[list[Any], list[str] | None]] = weakref.WeakKeyDictionary()
#: ``memo -> {id(answer): (answer, digest)}`` for the answers that can be
#: changed in place, as the last statement that called the memo left them:
#: what tells a statement that changed one in place (``f(1).append(5)``)
#: from one that only read it.
_answers_seen: weakref.WeakKeyDictionary[Any, dict[int, tuple[Any, str | bytes]]] = weakref.WeakKeyDictionary()

#: The types whose values nothing can change in place.
_FIXED_TYPES = (int, float, complex, str, bytes, bool, type(None), range)


def _is_user_memo(value: Any) -> bool:
    if _LRU_WRAPPER is None or type(value) is not _LRU_WRAPPER:
        return False
    code = attribute_or(attribute_or(value, "__wrapped__", None), "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


def _memos_in(value: Any, path: list[str], seen: set[int]) -> Iterator[tuple[list[str], Any]]:
    """``(path, memo)`` for the memos *value* is or holds: itself, the one
    under its decorators (``~`` steps), or the ones its class body holds
    (``.name`` steps)."""
    for _ in range(_MAX_WRAPPED + 1):
        if _is_user_memo(value):
            yield path, value
            return
        inner = attribute_or(value, "__wrapped__", None) if instance_of(value, types.FunctionType) else None
        if inner is None:
            break
        value, path = inner, [*path, "~"]
    if instance_of(value, type) and id(value) not in seen:
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
    holders = _holders(namespace)
    if known is not None and known[0] == holders:
        return known[1]
    found: list[tuple[list[str], Any]] = []
    for attr, raw in list(namespace.items()):
        held = attribute_or(raw, "__func__", raw) if instance_of(raw, (staticmethod, classmethod)) else raw
        if instance_of(held, types.FunctionType) or _is_user_memo(held):
            found.extend(_memos_in(held, ["." + attr], seen))
    try:
        _owner_memos[cls] = (holders, found)
    except TypeError:
        pass
    return found


def _owner_memo_paths(owner: Any, namespace: Mapping[str, Any]) -> list[tuple[list[str], Any]]:
    """The memos at the top of *namespace* (a module's), remembered per owner."""
    try:
        known = _owner_memos.get(owner)
    except TypeError:  # an owner that takes no weak reference
        known = None
    holders = _holders(namespace)
    if known is not None and known[0] == holders:
        return known[1]
    seen: set[int] = set()
    found = [hit for name, value in list(namespace.items()) for hit in _memos_in(value, [name], seen)]
    try:
        _owner_memos[owner] = (holders, found)
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
        name = attribute_or(mod, "__name__", None)
        if not isinstance(name, str) or name == "__main__":
            continue  # the notebook's own namespace is looked at as such
        try:
            if is_user_module(mod):
                mods.append(mod)
        except Exception:  # noqa: BLE001 - an odd entry in sys.modules
            continue
    _user_modules = (stamp, tuple(mods))
    return _user_modules[1]


def _ns_memos(ns: Mapping[str, Any]) -> tuple[list[tuple[list[str], Any]], list[tuple[str, type]]]:
    """The memos bound in *ns* outside classes, and the classes bound there
    (whose bodies `_class_part` reads each time, as a class can be given a
    memo without the namespace changing)."""
    seen: set[int] = set()
    kinds = _HOLDER_TYPES
    found: list[tuple[list[str], Any]] = []
    classes: list[tuple[str, type]] = []
    for name, value in list(ns.items()):
        if instance_of(value, type):
            classes.append((name, value))
        elif type(value) in kinds:
            found.extend((["ns", *path], memo) for path, memo in _memos_in(value, [name], seen))
    return found, classes


def _class_part(classes: list[tuple[str, type]]) -> list[tuple[list[str], Any]]:
    seen: set[int] = set()
    return [(["ns", *path], memo) for name, cls in classes for path, memo in _memos_in(cls, [name], seen)]


def _module_memos() -> list[tuple[list[str], Any]]:
    found: list[tuple[list[str], Any]] = []
    for mod in _modules_of_user():
        try:
            namespace = vars(mod)
        except TypeError:
            continue
        found.extend((["module", mod.__name__, *path], memo) for path, memo in _owner_memo_paths(mod, namespace))
    return found


_HOLDER_TYPES = frozenset(t for t in (types.FunctionType, _LRU_WRAPPER, staticmethod, classmethod) if t is not None)


def _holders(ns: Mapping[str, Any]) -> tuple[int, ...]:
    """The ids of the values in *ns* that can be or hold a memo (functions,
    wrappers, classes): binding or rebinding one changes this, and nothing
    else bound in the namespace can bring a memo in."""
    kinds = _HOLDER_TYPES
    # ``issubclass`` of the real type is what `instance_of` answers for
    # ``type`` (no object passes for a class by its ``__class__``), without
    # its per-value lookups: this runs over the whole namespace twice a
    # statement.
    return tuple(id(v) for v in list(ns.values()) if (kind := type(v)) in kinds or issubclass(kind, type))


#: Bumped before each statement (`new_statement`).
_statement = 0
#: ``id of a namespace -> (its holders, its memos outside classes, its
#: classes)`` and ``(statement, the user's modules then, their memos)``: a
#: namespace is looked through again only when a function or class was
#: bound in it, and the modules once per statement or when one was
#: imported (walking hundreds of modules per call made a 50,000-call loop
#: take over 30 s). The memos and classes are held weakly: an entry
#: outlives its namespace (a notebook's ``__main__`` made again), and
#: would keep the classes once bound there alive, with what their class
#: attributes hold. One gone is skipped: a namespace still alive holds what
#: was found in it.
_ns_reached: dict[int, tuple[tuple[int, ...], list[tuple[list[str], Any]], list[tuple[str, Any]]]] = {}
_modules_reached: tuple[int, Any, list[tuple[list[str], Any]]] = (-1, None, [])


def new_statement() -> None:
    """A statement is about to run: look for the memos it reaches afresh."""
    global _statement
    _statement += 1


def _where(ns: Mapping[str, Any], *, fresh: bool = False) -> list[tuple[list[str], Any]]:
    """``(where, memo)`` for every memo of the user's a statement can reach:
    ``["ns", name, *steps]`` or ``["module", module, name, *steps]``.
    *fresh*: look through the modules again, for a memo bound in one by the
    statement that just ran (``mod.f = functools.cache(g)``)."""
    global _modules_reached
    holders = _holders(ns)
    known = _ns_reached.get(id(ns))
    if known is None or known[0] != holders:
        if len(_ns_reached) > 64:
            _ns_reached.clear()
        memos, classes = _ns_memos(ns)
        _ns_reached[id(ns)] = (holders, [(w, _weak(m)) for w, m in memos], [(n, _weak(c)) for n, c in classes])
        known = (holders, memos, classes)
    else:
        known = (holders, _alive(known[1]), _alive(known[2]))
    modules = _modules_of_user()
    if (
        fresh
        or _modules_reached[0] != _statement
        or _modules_reached[1] is not modules
        # one rebound since (``mod.f = cache(g)`` in this statement)
        or any(_resolve(where, ns) is not memo for where, memo in _modules_reached[2])
    ):
        _modules_reached = (_statement, modules, _module_memos())
    return known[1] + _class_part(known[2]) + _modules_reached[2]


def _weak(obj: Any) -> Any:
    try:
        return weakref.ref(obj)
    except TypeError:
        return obj  # takes none (a ``staticmethod``): kept as it is


def _alive(kept: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """*kept* (as `_weak` left each second item) with the gone ones left out."""
    out = []
    for where, item in kept:
        if type(item) is weakref.ReferenceType:
            item = item()
            if item is None:
                continue
        out.append((where, item))
    return out


def watched(ns: Mapping[str, Any]) -> list[tuple[list[str], Any, int, int, int, Any]] | None:
    """Before a statement or a cached call runs: each memo it can reach, with
    its counters and which keys and answers it holds (`_held`)."""
    if _LRU_WRAPPER is None:
        return None
    out = []
    for where, memo in _where(ns):
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        out.append((where, memo, info.hits, info.misses, info.currsize, _held(memo)))
    return out or None


def misses_before(before: list[tuple[list[str], Any, int, int, int, Any]] | None) -> dict[int, int]:
    """``id(memo) -> its miss count`` in what `watched` saw before a run:
    a memo's count unmoved since, the run did not fill it."""
    return {id(memo): misses for _, memo, _, misses, *_ in before or ()}


def bodies_run(
    before: list[tuple[list[str], Any, int, int, int, Any]] | None, cleared: Clears | None = None
) -> dict[types.CodeType, tuple[dict, str]]:
    """After a statement ran: the code of the function under each memo of
    *before* (`watched`) whose body ran in it, as a trace records what ran
    (code -> its globals and module), for `unseen_writers` where there is no
    trace (Python 3.10 and 3.11). A body ran when the memo's miss count
    moved, or may have when the run cleared the memo (*cleared*), which
    resets the count. A memo whose counters raise is taken to have run."""
    fired = cleared.fired if cleared is not None else ()
    out: dict[types.CodeType, tuple[dict, str]] = {}
    for _, memo, _, misses, *_ in before or ():
        try:
            ran = memo.cache_info().misses != misses or id(memo) in fired
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            ran = True
        if not ran:
            continue
        fn = attribute_or(memo, "__wrapped__", None)
        code = attribute_or(fn, "__code__", None)
        globals_ = attribute_or(fn, "__globals__", None)
        if isinstance(code, types.CodeType) and isinstance(globals_, dict):
            out[code] = (globals_, str(globals_.get("__name__", "")))
    return out


#: Pairs of a memo kept to tell a clear cash saw from one it did not
#: (`watching_clears`): one it did not see dropped them all.
_CLEAR_SAMPLES = 8


class Clears:
    """What `watching_clears` saw of the ``cache_clear`` calls a statement
    made: ``fired`` holds the ids of the memos it cleared, ``first`` those
    whose first clear came before any call of the memo in the run."""

    def __init__(self) -> None:
        self.fired: set[int] = set()
        self.first: set[int] = set()


@contextlib.contextmanager
def watching_clears(before: list[tuple[list[str], Any, int, int, int, Any]] | None) -> Iterator[Clears]:
    """While a statement runs: see each ``cache_clear`` call of an unbounded
    memo it can reach (*before*, `watched`), through a ``cache_clear`` set on
    the memo for the run that records the call and clears the memo as its
    own does. Put back as it was after.

    A run that empties a memo before any call of it (``held_karp_tsp``
    clearing ``shortest_segment`` for a new problem) did nothing what the
    memo held could change: a hit empties the live memo and puts in what
    the run left, whatever it held (`used`). The memo's counters cannot say
    so, as a clear zeroes them. Taken as first only when, at the clear, its
    counters and size are those it had when the statement began and the
    pairs kept of it (`_CLEAR_SAMPLES`) are still there: a clear not seen
    (``type(f).cache_clear(f)``, a ``cache_clear`` bound earlier) drops
    them. A memo empty then is never taken as first: nothing is left to
    tell such a clear by."""
    seen = Clears()
    installed: list[tuple[Any, Any]] = []
    try:
        for _, memo, *_ in before or ():
            if not _unbounded(memo) or "cache_clear" in attribute_or(memo, "__dict__", {}):
                continue
            try:
                info = memo.cache_info()
                items = _items(memo) or []
            except Exception:  # noqa: BLE001 - a wrapper whose counters raise
                continue
            start = (info.hits, info.misses, info.currsize)
            samples = items[:_CLEAR_SAMPLES] if info.currsize else []
            real = type(memo).cache_clear

            def cache_clear(memo: Any = memo, start: tuple = start, samples: list = samples, real: Any = real) -> None:
                if id(memo) not in seen.fired:
                    seen.fired.add(id(memo))
                    if samples and _untouched(memo, start, samples):
                        seen.first.add(id(memo))
                real(memo)

            cache_clear.__doc__ = real.__doc__
            try:
                memo.cache_clear = cache_clear
            except (AttributeError, TypeError):
                continue
            installed.append((memo, cache_clear))
        yield seen
    finally:
        for memo, shadow in installed:
            if attribute_or(memo, "__dict__", {}).get("cache_clear") is shadow:
                del memo.cache_clear


def _untouched(memo: Any, start: tuple[int, int, int], samples: list[tuple[Any, Any]]) -> bool:
    """*memo*'s counters and size are *start*, and it holds each of the
    *samples* pairs, the very objects: no call of it, and no clear, since."""
    try:
        info = memo.cache_info()
        if (info.hits, info.misses, info.currsize) != start:
            return False
        held = {(id(key), id(answer)) for key, answer in _items(memo) or ()}
    except Exception:  # noqa: BLE001 - not readable: not taken as untouched
        return False
    return all((id(key), id(answer)) in held for key, answer in samples)


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
        refs = get_referents(memo)
    except Exception:  # noqa: BLE001 - a wrapper that cannot be read
        return None
    if refs and refs[0] is type(memo):
        refs = refs[1:]
    if maxsize is None:
        cache = refs[0] if refs else None
        if type(cache) is not dict or len(cache) != size or cache is attribute_or(memo, "__dict__", None):
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


def _pair_digest(key: Any, answer: Any) -> str:
    """One answer of a memo, its key with it, as 32 hex digits: an entry
    keeps one per answer it read, and the two full digests were 129
    characters each -- half a megabyte for a memo of 4,096 answers."""
    pair = f"{_key_digest(key)}:{compute_hash(answer)}".encode()
    return hashlib.blake2b(pair, digest_size=16).hexdigest()


def _item_digests(memo: Any) -> list[str] | None:
    """``key-digest:answer-digest`` for every answer *memo* holds, or None
    when they cannot be read here."""
    items = _items(memo)
    if items is None or len(items) > MAX_KEYED:
        logger.debug("[MEMO] the answers of %r are not read: never served over it", memo)
        return None
    flat = [part for item in items for part in item]
    known = _digests.get(memo)
    if (
        known is not None
        and len(known[0]) == len(flat)
        and all(a is b for a, b in zip(known[0], flat))
        and all(_fixed(k, key=True) and _fixed(v) for k, v in items)
    ):
        return known[1]
    try:
        digests: list[str] | None = sorted({_pair_digest(k, v) for k, v in items})
    except Exception:  # noqa: BLE001 - a key or answer with no digest: unreadable
        digests = None
    _digests[memo] = (flat, digests)
    return digests


def _fixed(value: Any, depth: int = 3, *, key: bool = False) -> bool:
    """Nothing can change *value* in place: a number, a string, bytes, None,
    or a tuple or frozenset of those. In a *key*, the markers ``functools``
    puts there count too: the bare ``object()`` between positional and
    keyword arguments, and the argument types of a ``typed=True`` memo."""
    if type(value) in _FIXED_TYPES or (key and (type(value) is object or instance_of(value, type))):
        return True
    if depth > 0 and type(value) in (tuple, frozenset):
        return all(_fixed(part, depth - 1, key=key) for part in value)
    return False


#: The builtins a pure helper may call: each returns a value made from its
#: arguments alone, and reads or writes nothing else.
_PURE_BUILTINS = frozenset(
    "abs all any bin bool bytes chr complex dict divmod enumerate filter float format frozenset hex int "
    "isinstance issubclass len list map max min oct ord pow range repr reversed round set slice sorted str "
    "sum tuple zip".split()
)
#: Instructions a pure helper never runs: reading an attribute (a method of
#: anything, a module's state), writing or deleting a global or an
#: attribute, importing, or reading a name a class body or exec resolves.
_IMPURE_OPS = frozenset(
    "LOAD_ATTR LOAD_METHOD LOAD_SUPER_ATTR STORE_ATTR DELETE_ATTR STORE_GLOBAL DELETE_GLOBAL IMPORT_NAME "
    "IMPORT_FROM IMPORT_STAR LOAD_NAME STORE_NAME DELETE_NAME LOAD_CLASSDEREF LOAD_FROM_DICT_OR_GLOBALS "
    "LOAD_FROM_DICT_OR_DEREF".split()
)


def _pure(memo: Any) -> bool:
    """Whether the function under *memo* provably computes its answer from
    its arguments alone and does nothing else, so running its body where a
    run would have found the answer kept differs in nothing.

    Narrow by design: the function reads no attribute, no closure variable
    and no global but a builtin of `_PURE_BUILTINS` (still the builtin, not
    shadowed by a global of that name) or the memo itself (``fib`` calling
    ``fib``), writes no global, and has no default argument but a fixed
    value (`_fixed`). ``return n * n`` and ``return sum(range(n))``
    qualify; a print, a random draw, a read of a global, the environment or
    a file, a method call, a call of another function of the user's (which
    a later cell can redefine), a default ``out=print`` or ``acc=[]`` do
    not. Asked afresh each time, as a builtin can be shadowed later."""
    import dis

    from cash.decorator.closure_fold import iter_code_scopes

    fn = attribute_or(memo, "__wrapped__", None)
    code = attribute_or(fn, "__code__", None)
    globals_ = attribute_or(fn, "__globals__", None)
    if code is None or not isinstance(globals_, dict) or code.co_freevars:
        return False
    if not is_user_code_file(code.co_filename):
        return False
    defaults = (
        *(attribute_or(fn, "__defaults__", None) or ()),
        *(attribute_or(fn, "__kwdefaults__", None) or {}).values(),
    )
    if not all(_fixed(value) for value in defaults):
        return False
    try:
        for scope in iter_code_scopes(code):
            for instr in dis.get_instructions(scope):
                if instr.opname in _IMPURE_OPS:
                    return False
                if instr.opname != "LOAD_GLOBAL":
                    continue
                name = instr.argval
                if name in globals_:
                    if globals_[name] is not memo:
                        return False
                elif name not in _BUILTINS or _BUILTINS[name] is not _builtin(name, globals_):
                    return False
    except Exception:  # noqa: BLE001 - code this cannot read: not known to be pure
        return False
    return True


_UNSET = object()


def _builtin_table() -> dict[str, Any]:
    import builtins

    return {name: vars(builtins)[name] for name in _PURE_BUILTINS if name in vars(builtins)}


#: Each pure builtin as the interpreter started with it.
_BUILTINS = _builtin_table()


def _builtin(name: str, globals_: dict) -> Any:
    """What *name* resolves to as a builtin from *globals_* now."""
    table = globals_.get("__builtins__")
    table = vars(table) if isinstance(table, types.ModuleType) else table
    return table.get(name, _UNSET) if isinstance(table, dict) else _UNSET


#: The types an answer's own items may have for `_answer_digest`'s quick form.
_SCALAR_TYPES = frozenset((int, float, complex, str, bytes, bool, type(None)))


def _answer_digest(answer: Any) -> str | bytes:
    """What `_changed_in_place` compares an answer by, from one statement to
    the next and only there. A list, set or dict holding only numbers,
    strings, bytes and None (a Held-Karp memo's segments) is taken by its
    pickle, which keeps each item's type and exact value: ten times quicker
    than `compute_hash`, which cost seconds for a memo of thousands of
    answers cleared and filled by every call. A bytes digest never equals a
    `compute_hash` one, so an answer that changes form counts as changed."""
    kind = type(answer)
    if kind is list or kind is set or kind is dict:
        parts = itertools.chain(answer.keys(), answer.values()) if kind is dict else answer
        if all(type(part) in _SCALAR_TYPES for part in parts):
            return hashlib.blake2b(pickle.dumps(answer, protocol=5), digest_size=16).digest()
    return compute_hash(answer)


def _changed_in_place(
    memo: Any,
    items: list[tuple[Any, Any]],
    *,
    seen: dict[int, tuple[Any, str | bytes]] | None = None,
    judged: list[tuple[Any, Any]] | None = None,
) -> bool | None:
    """Whether an answer *memo* holds was changed in place since the last
    statement that called it left it, None when one cannot be told (an
    answer that can change in place and was not seen then); and remember
    them as they are now. *seen*: compare with these instead of what was
    last remembered (`_held`'s third part). *judged*: the pairs whose
    answers decide the verdict, when not all of *items* (the answers the
    memo held before a run that added others)."""
    if seen is None:
        seen = _answers_seen.get(memo, {})
    now: dict[int, tuple[Any, str | bytes]] = {}
    for _, answer in items:
        if _fixed(answer):
            continue
        try:
            now[id(answer)] = (answer, _answer_digest(answer))
        except Exception:  # noqa: BLE001 - an answer with no digest: never judged unchanged
            continue
    verdict: bool | None = False
    for _, answer in items if judged is None else judged:
        if _fixed(answer):
            continue
        current = now.get(id(answer))
        if current is None and judged is not None:
            try:
                current = (answer, _answer_digest(answer))
            except Exception:  # noqa: BLE001 - an answer with no digest
                current = None
        before = seen.get(id(answer))
        if current is None or before is None or before[0] is not answer:
            verdict = None if verdict is False else verdict
        elif before[1] != current[1]:
            verdict = True
    _answers_seen[memo] = now
    return verdict


def _held(memo: Any) -> tuple[frozenset[tuple[int, int]], list[tuple[Any, Any]], Any] | None:
    """Which key and answer objects *memo* holds, or None past `MAX_KEYED`:
    a run that empties it (``cache_clear`` by any name) and fills it again
    can leave its counters as they were, not these. The objects are kept
    with their ids, so a refill cannot put a new key at a freed address.
    Third, the answers as the last statement that called it left them
    (`_answers_seen`): a call cash caches inside the statement remembers
    them again as it leaves them, and the statement's own record of what it
    changed is asked of the state before it ran."""
    items = _items(memo)
    if items is None or len(items) > MAX_KEYED:
        return None
    return frozenset((id(key), id(answer)) for key, answer in items), items, _answers_seen.get(memo)


def _label(where: list[str]) -> str:
    """``f``, ``C.f``, ``mod.table`` for the warning."""
    head = ".".join(where[1:3]) if where and where[0] == "module" else (where[1] if len(where) > 1 else "?")
    steps = where[3:] if where and where[0] == "module" else where[2:]
    return head + "".join(step for step in steps if step != "~")


def used(
    before: list[tuple[list[str], Any, int, int, int, Any]] | None,
    *,
    clears: bool = False,
    ns: Mapping[str, Any] | None = None,
    site: tuple[str | None, Any] | None = None,
    changes: list[dict] | None = None,
    cleared: Clears | None = None,
) -> tuple[str, str, str | None] | list[dict] | None:
    """After a statement ran: ``(FILLED, why, unread)`` when what it did to a memo
    leaves it otherwise than a served entry would, else each memo it called
    with the answers it held then. *clears*: the code that ran names
    ``cache_clear``, which resets the counters this reads. *ns*: look for
    memos the run made or bound too (a ``def`` in it, a module it imported),
    each taken as empty before it. *site*: ``(code, fn)``, the statement's
    text or the function called, read for the memos whose ``cache_clear``
    it names, which the warning names first (`_changed_label`). *unread*
    names the memo when the reason is that it holds more than `MAX_KEYED`
    answers, for the statement's warning, else None.

    *changes*: given (a list, by a statement whose code reads no memo's
    counters), a memo the run filled or emptied is recorded rather than
    refused, where the entry can carry the change (`_change_of`): the
    record names the pairs it added, or says it emptied the memo first, and
    *changes* gets ``{"emptied": bool, "pairs": [(key, answer), ...]}`` for
    the entry to store with the outputs, at the index the record's
    ``"delta"`` gives; for keys that are plain values (`_cleared_change`)
    ``{"emptied": True, "keys": <their pickle>, "answers": [...]}``: no
    tuple of a key and its answer for the share check to walk, and keys
    that the tiers store and copy as bytes. A hit puts them into the memo
    (`put_back`).

    *cleared*: what `watching_clears` saw. A memo the run emptied before
    any call of it is recorded as emptied with what it holds now, whatever
    it held before, which decided nothing (``"cleared"``); and one it
    cleared is never taken as untouched, though its counters end as they
    began."""
    before = list(before or ())
    if ns is not None and _LRU_WRAPPER is not None:
        known = {id(memo) for _, memo, *_ in before}
        fresh = [(where, memo) for where, memo in _where(ns, fresh=True) if id(memo) not in known]
        # A memo rebound in the run where cash looked before it (a module's,
        # by a reload in the same statement): the one bound now, from empty.
        for where, *_ in before:
            now = _resolve(where, ns)
            if _is_user_memo(now) and id(now) not in known:
                fresh.append((where, now))
        seen_now: set[int] = set()
        for where, memo in fresh:
            if id(memo) not in seen_now:
                seen_now.add(id(memo))
                before.append((where, memo, 0, 0, 0, (frozenset(), [], None)))
    if not before:
        return None
    out: list[dict] = []
    why: str | None = None
    unread: str | None = None
    for where, memo, hits, misses, size, held in before:
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        fired = cleared is not None and id(memo) in cleared.fired
        first = cleared is not None and id(memo) in cleared.first
        if (info.hits, info.misses, info.currsize) == (hits, misses, size) and not fired:
            now = _held(memo)
            if not clears:
                if held is None or (now is not None and now[0] == held[0]):
                    continue
                if changes is None:
                    clears = True  # emptied and filled again to the same counts
            elif changes is not None and held is not None and now is not None and now[0] == held[0]:
                # Code that names ``cache_clear`` (``held_karp_tsp`` clearing
                # ``shortest_segment``) leaves alone a memo whose counters
                # and very key and answer objects are as they were, and
                # which no clear it was seen making reached: a clear and a
                # refill would hold new answers. The statement did not call
                # it, so what it holds decides nothing: ``benchmark``
                # answers holding timings, computed again above, must not
                # stop the statement from being served.
                continue
        label = _label(where)
        items = _items(memo)
        change = None
        if changes is not None and first and why is None and items is not None:
            # Emptied before any call: stored whatever its size when its
            # keys are plain values (`_cleared_change`).
            change = _cleared_change(memo, items)
            moved = change if isinstance(change, bool) or change is None else False
        elif items is None or len(items) > MAX_KEYED:
            # Never served over (below), so what its answers hold is not
            # read: digesting each one cost seconds a statement for a memo of
            # 100,000 answers. Forgotten too, so a later statement reading it
            # once it is small again takes its answers as unseen.
            _answers_seen.pop(memo, None)
            moved = None
        elif changes is not None and held is not None and why is None:
            change = _change_of(memo, items, held, (info.hits, info.misses, info.currsize), (hits, misses, size))
            moved = change if isinstance(change, bool) or change is None else False
        else:
            moved = _changed_in_place(memo, items)
        if why is not None:
            continue  # already told; the rest only remember their answers
        if isinstance(change, dict):
            if not change["emptied"] and not clears:
                # A helper that provably computes from its arguments alone
                # is called again where the memo lacks a key (`_refill`),
                # and served where the memo holds it already.
                calls = _calls_of(memo, items)
                if calls is not None and all(_fixed(answer) for _, answer in items) and _pure(memo):
                    out.append({"where": where, "items": _item_digests(memo), "pure": True, "calls": calls})
                    continue
            if not _unbounded(memo):
                why = f"It filled the helper memo {label}: served, the memo would stay as it was"
            elif change["unportable"]:
                why = f"It filled the helper memo {label} with a key cash cannot store"
            else:
                out.append(
                    {
                        "where": where,
                        "items": change["items"],
                        "pure": False,
                        "calls": None,
                        "keys": change["keys"],
                        "emptied": change["emptied"] or None,
                        "cleared": change.get("cleared") or None,
                        "plain": change.get("plain") or None,
                        "marked": change.get("marked") or None,
                        "kd": KEY_DIGESTS,
                        "delta": len(changes),
                    }
                )
                if change.get("plain"):
                    changes.append({"emptied": True, "keys": change["keys_pickle"], "answers": change["answers"]})
                else:
                    changes.append({"emptied": change["emptied"], "pairs": change["pairs"]})
            continue
        if clears and changes is None:
            named = _memos_cleared_by(site[0], ns, site[1]) if site is not None else set()
            changed = _changed_label(before, named)
            memo_named = f"the helper memo {changed}" if changed else "a helper memo"
            why = f"It may empty {memo_named} or read its counters (cache_clear, cache_info)"
        elif info.misses < misses or info.hits < hits:
            why = f"It emptied the helper memo {label}"
        elif items is None or len(items) > MAX_KEYED:
            why = f"It read the helper memo {label}, which holds more answers than cash checks ({MAX_KEYED})"
            if items is not None:
                unread = f"{label}, which holds {len(items):,} answers"
        elif moved:
            why = f"It changed an answer the helper memo {label} holds in place"
        elif info.misses != misses:
            calls = _calls_of(memo, items)
            if calls is not None and all(_fixed(answer) for _, answer in items) and _pure(memo):
                out.append({"where": where, "items": _item_digests(memo), "pure": True, "calls": calls})
                continue
            why = f"It filled the helper memo {label}: served, the memo would stay as it was"
        elif moved is None:
            why = f"It read an answer of the helper memo {label} that cash had not seen"
        else:
            digests = _item_digests(memo)
            if digests is None:
                why = f"It read the helper memo {label}, whose answers cash cannot digest"
            else:
                calls = _calls_of(memo, items)
                pure = calls is not None and all(_fixed(answer) for _, answer in items) and _pure(memo)
                out.append({"where": where, "items": digests, "pure": pure, "calls": calls if pure else None})
    if why is not None:
        if changes is not None:
            changes.clear()
        return FILLED, f"{why} (statement re-executes)", unread
    return out or None


def _change_of(
    memo: Any,
    items: list[tuple[Any, Any]],
    held: tuple,
    counts: tuple[int, int, int],
    counts_before: tuple[int, int, int],
) -> dict | bool | None:
    """What a run did to *memo*, which held *held* before it and holds
    *items* now, when an entry can carry it: a dict naming the pairs it
    added (or, once it emptied the memo, every pair it holds now), with the
    digests a lookup checks (`unfilled`) and a hit puts back (`put_back`).
    Else what `_changed_in_place` says of the answers it held before:
    True when one was changed in place, None when one cannot be told (it
    was not seen before the run), False when the run only read answers.

    Emptied: a key it held is gone, or a counter went back. The lookup then
    asks for exactly the answers it held before (what the run found
    decides what it did before it emptied the memo), and a hit empties it
    and puts in what the run left. Else the lookup asks for none of the
    keys it added (a cell above that filled one makes a run find the answer
    and skip the body it ran) and, when the run found an answer the memo
    held (its hits moved), for every answer it held before: the counter
    does not say which one it found. A run that found none read none of
    them, so they decide nothing: ``benchmark`` answers holding timings,
    computed again above, do not stop a statement that only added its own
    keys from being served."""
    _, before_items, seen = held
    before_keys = {id(key) for key, _ in before_items}
    hits, misses, size = counts
    hits_before, misses_before, size_before = counts_before
    now_ids = {(id(key), id(answer)) for key, answer in items}
    emptied = (
        hits < hits_before
        or misses < misses_before
        or size < size_before
        or not {(id(key), id(answer)) for key, answer in before_items} <= now_ids
    )
    added = items if emptied else [(key, answer) for key, answer in items if id(key) not in before_keys]
    if not emptied and not added:
        # Only read: the read-only record (`used`), judged on every answer.
        return _changed_in_place(memo, items)
    if not emptied and misses - misses_before != len(added):
        # A call missed and kept no answer (it raised): the key it lacked is
        # one no check asks the memo to lack, so this cannot be put back.
        return None
    # The answers it held before decide; the ones it added are new.
    verdict = _changed_in_place(memo, items, seen=seen if seen is not None else {}, judged=before_items)
    if verdict is not False:
        return verdict
    read = emptied or hits != hits_before
    try:
        before_digests = _digests_of(memo, before_items) if read else []
        keys = [_change_key_digest(key) for key, _ in added]
    except Exception:  # noqa: BLE001 - a key or answer with no digest
        return None
    return {
        "emptied": emptied,
        "items": before_digests,
        "keys": keys,
        "pairs": list(added),
        "unportable": not all(_portable_key(key) for key, _ in added),
    }


def _cleared_change(memo: Any, items: list[tuple[Any, Any]]) -> dict | None:
    """The change a run made to *memo*, which it emptied before any call of
    it (`watching_clears`): every pair it holds now, the run's own. What it
    held before is neither recorded nor asked for.

    Keys that are plain values (`_plain_key`) and come back from a pickle
    equal are taken as they are, at any count (a Held-Karp memo refilled
    for 16 cities holds 245,760): a hit puts back the very keys the entry
    holds, so no digest of each is recorded (``"plain"``). Any other key
    takes a digest, up to `MAX_KEYED` of them; over it the change is
    unportable. None when a key has no digest. The answers are remembered
    as seen, up to `MAX_KEYED` of them (`used` forgets them over it)."""
    if len(items) <= MAX_KEYED:
        _changed_in_place(memo, items)
    else:
        _answers_seen.pop(memo, None)
    keys = [key for key, _ in items]
    plain = _plain_keys(keys)
    if plain is not None:
        return {
            "emptied": True,
            "cleared": True,
            "plain": True,
            "marked": plain[1],
            "items": [],
            "keys": None,
            "keys_pickle": plain[0],
            "answers": [answer for _, answer in items],
            "unportable": False,
        }
    if len(items) > MAX_KEYED:
        return {"emptied": True, "cleared": True, "items": [], "keys": None, "pairs": [], "unportable": True}
    try:
        digests = [_change_key_digest(key) for key in keys]
    except Exception:  # noqa: BLE001 - a key with no digest
        return None
    return {
        "emptied": True,
        "cleared": True,
        "items": [],
        "keys": digests,
        "pairs": list(items),
        "unportable": not all(_portable_key(key) for key in keys),
    }


#: The types a plain key (`_plain_key`) is made of besides tuples and
#: frozensets: what a pickle gives back as an equal value of the same type.
_PLAIN_TYPES = frozenset((int, float, complex, str, bytes, bool, type(None)))


def _plain_key(key: Any) -> bool:
    """*key* is a plain value (`_plain_value`), or a tuple of plain values
    and ``functools``' keyword marker, put back as the live one
    (`_live_key`)."""
    if type(key) is tuple:
        for part in key:
            kind = type(part)
            if kind not in _PLAIN_TYPES and kind is not object and not _plain_value(part, 3):
                return False
        return True
    return _plain_value(key, 4)


def _plain_value(value: Any, depth: int) -> bool:
    """*value* is a number, string, bytes or None, or a tuple or frozenset
    of those nested up to *depth*."""
    kind = type(value)
    if kind in _PLAIN_TYPES:
        return True
    if (kind is not tuple and kind is not frozenset) or depth <= 0:
        return False
    if _PLAIN_TYPES.issuperset(map(type, value)):
        return True
    return all(_plain_value(part, depth - 1) for part in value)


def _plain_keys(keys: list[Any]) -> tuple[bytes, bool] | None:
    """The pickle of *keys*, and whether one holds ``functools``' keyword
    marker, when every one is plain (`_plain_key`) and comes back from that
    pickle equal (not NaN, which equals nothing): what a hit puts back is
    then the run's keys. None otherwise. About a second for the 245,760
    keys of a Held-Karp memo of 16 cities, where a digest of each took
    14 s."""
    if not all(map(_plain_key, keys)):
        return None
    try:
        data = pickle.dumps(keys, protocol=5)
        back = pickle.loads(data)
    except Exception:  # noqa: BLE001 - not pickled or not read back: not plain
        return None
    mark = _live_mark()
    marked = mark is not None and any(type(key) is tuple and mark in key for key in keys)
    if marked:
        back = [_live_key(key) for key in back]
    return (data, marked) if back == keys else None


def _digests_of(memo: Any, items: list[tuple[Any, Any]]) -> list[str]:
    """`_item_digests` of *items*, pairs *memo* held: the digests remembered
    for it when they are of these very objects, which nothing changed."""
    flat = [part for item in items for part in item]
    known = _digests.get(memo)
    if (
        known is not None
        and known[1] is not None
        and len(known[0]) == len(flat)
        and all(a is b for a, b in zip(known[0], flat))
        and all(_fixed(k, key=True) and _fixed(v) for k, v in items)
    ):
        return known[1]
    return sorted({_pair_digest(key, answer) for key, answer in items})


#: How a change record's key digests (`_change_key_digest`) are made, kept
#: in the record (``"kd"``): a record whose digests were made otherwise is
#: never compared with the live memo's, as a key it holds would then read
#: as one the memo lacks.
KEY_DIGESTS = 2


def _change_key_digest(key: Any) -> str:
    """A digest of one memo key a change records (`used`), the same for
    keys a memo takes as one (``1``, ``1.0`` and ``True``, two equal
    frozensets whatever order they iterate in) and in every process. Keys
    of numbers, strings, bytes and None in tuples and frozensets are read
    directly, as a filled Held-Karp memo holds thousands of them; any other
    part of a tuple goes through `_key_digest` on its own, and any other
    key whole.

    A frozenset's order is its hash table's, which a pickle rebuilds
    otherwise: digested in that order, a key put back from an entry came
    back as another, and an equal key built in another order as one the
    memo lacks."""
    canon = _canonical_key(key, 4)
    if canon is _UNSET:
        return _key_digest(key)
    return hashlib.blake2b(repr(canon).encode(), digest_size=16).hexdigest()


def _canonical_key(key: Any, depth: int) -> Any:
    kind = type(key)
    if kind is int:
        return key
    if kind is bool:
        return int(key)
    if kind is float:
        return int(key) if key.is_integer() else ("f", repr(key))
    if kind is complex:
        return _canonical_key(key.real, depth) if key.imag == 0 else ("c", repr(key))
    if kind is str or kind is bytes:
        return (kind.__name__, key)
    if key is None:
        return ("n",)
    if kind is object:
        return ("<kwd_mark>",)
    if kind is tuple and depth > 0:
        parts = tuple(_canonical_key(part, depth - 1) for part in key)
        if any(part is _UNSET for part in parts):
            parts = tuple(
                ("d", _key_digest(part)) if canon is _UNSET else canon for part, canon in zip(key, parts, strict=True)
            )
        return ("t", parts)
    if kind is frozenset and depth > 0:
        members = [_canonical_key(part, depth - 1) for part in key]
        if any(part is _UNSET for part in members):
            return _UNSET
        return ("fs", tuple(sorted(members, key=repr)))
    return _UNSET


def _unbounded(memo: Any) -> bool:
    """*memo* keeps its answers in a plain dict (``maxsize=None``), which a
    hit can put answers into; a bounded one keeps a list only C updates."""
    try:
        return memo.cache_parameters().get("maxsize") is None
    except Exception:  # noqa: BLE001 - a wrapper without its parameters
        return False


def _portable_key(key: Any, depth: int = 4) -> bool:
    """*key* comes back from a pickle as a key the live memo finds when it
    is called with the same arguments: numbers (not NaN), strings, bytes,
    None, tuples and frozensets of those, ``functools``' keyword marker (put
    back as the live one), and functions and classes found by their module
    and name (pickled by reference). Not an object equal by identity, which
    a copy never equals."""
    kind = type(key)
    if kind in _FIXED_TYPES:
        return not ((kind is float or kind is complex) and key != key)
    if kind is object:
        return True
    if kind in (tuple, frozenset):
        return depth > 0 and all(_portable_key(part, depth - 1) for part in key)
    if kind is types.FunctionType or instance_of(key, type):
        module = sys.modules.get(attribute_or(key, "__module__", None) or "")
        value: Any = module
        qualname = attribute_or(key, "__qualname__", None)
        if module is None or not isinstance(qualname, str) or "<locals>" in qualname:
            return False
        for part in qualname.split("."):
            value = attribute_or(value, part, None)
        return value is key
    return False


def _changed_label(
    before: list[tuple[list[str], Any, int, int, int, Any]], named: set[int] = frozenset()
) -> str | None:
    """The label of the memo in *before* the statement most likely cleared,
    for a warning about code that can clear one: first one whose counters
    went back or that dropped a key it held; else one whose ``cache_clear``
    the code names (*named*, ids) and that moved, or is the only one named
    (a Run All defines the memo again, so it is empty before the clear and
    only grows). None otherwise: a memo that only grew may be one the
    statement filled beside a clear it made through a name cash could not
    read, and the memo cash found first one it never touched."""
    moved: list[tuple[int, str]] = []
    for where, memo, hits, misses, size, held in before:
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        now = _held(memo)
        went_back = info.hits < hits or info.misses < misses or info.currsize < size
        dropped = held is not None and now is not None and bool(held[0] - now[0])
        if went_back or dropped:
            return _label(where)
        if (info.hits, info.misses, info.currsize) != (hits, misses, size) or (
            held is not None and now is not None and now[0] != held[0]
        ):
            moved.append((id(memo), _label(where)))
    named_moved = [label for memo_id, label in moved if memo_id in named]
    if len(named_moved) == 1:
        return named_moved[0]
    named_watched = {_label(where) for where, memo, *_ in before if id(memo) in named}
    if len(named_watched) == 1:
        return next(iter(named_watched))
    return None


def _memos_cleared_by(code: str | None, ns: Mapping[str, Any] | None, fn: Any = None) -> set[int]:
    """The ids of the memos whose ``cache_clear`` *code* (a statement's text,
    read with *ns*) or *fn*, or a function of the user's either reaches,
    names as ``memo.cache_clear``, or through attributes of a global
    (``holder.memo.cache_clear``), read without running a property."""
    import ast
    import dis
    import inspect

    from cash.decorator.closure_fold import iter_code_scopes
    from cash.notebook.callee_reach import reached_user_code

    found: set[int] = set()

    def memo_id(value: Any, attrs: list[str] = ()) -> None:
        for attr in attrs:
            try:
                value = inspect.getattr_static(value, attr)
            except Exception:  # noqa: BLE001 - not there: no memo named
                return
        if type(value) is _LRU_WRAPPER:
            found.add(id(value))

    functions: list[Any] = [fn] if fn is not None else []
    try:
        if code is not None and ns is not None:
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Attribute) and node.attr == "cache_clear":
                    attrs: list[str] = []
                    value = node.value
                    while isinstance(value, ast.Attribute):
                        attrs.insert(0, value.attr)
                        value = value.value
                    if isinstance(value, ast.Name):
                        memo_id(ns.get(value.id), attrs)
            functions.extend(reached_user_code(code, ns).functions)
        elif fn is not None:
            name, globals_ = attribute_or(fn, "__name__", None), attribute_or(fn, "__globals__", None)
            if isinstance(name, str) and isinstance(globals_, dict) and globals_.get(name) is fn:
                functions.extend(reached_user_code(name, globals_).functions)
    except Exception:  # noqa: BLE001 - not read: the counters decide
        pass
    for function in functions:
        code_obj = attribute_or(function, "__code__", None)
        globals_ = attribute_or(function, "__globals__", None)
        if code_obj is None or not isinstance(globals_, dict):
            continue
        for scope in iter_code_scopes(code_obj):
            if "cache_clear" not in scope.co_names:
                continue
            chain: list[str] = []  # a global, then the attributes read off it
            for ins in dis.get_instructions(scope):
                if ins.opname in ("LOAD_ATTR", "LOAD_METHOD") and chain:
                    if ins.argval == "cache_clear":
                        memo_id(globals_.get(chain[0]), chain[1:])
                        chain = []
                    else:
                        chain.append(ins.argval)
                    continue
                chain = [ins.argval] if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") else []
    return found


def _calls_of(memo: Any, items: list[tuple[Any, Any]]) -> list[list[str]] | None:
    """The calls that fill *memo* with the keys of *items* again, each as
    ``[digest of the item, text of (args, kwargs)]``, or None when one
    cannot be written so (a ``typed=True`` memo, a key of a type text does
    not give back, or whose text Python refuses: an int of 5000 digits)."""
    import ast

    try:
        if memo.cache_parameters().get("typed"):
            return None
    except Exception:  # noqa: BLE001 - a wrapper without its parameters
        return None
    calls = []
    for key, answer in items:
        if not isinstance(key, tuple):
            call: tuple = ((key,), {})
        else:
            mark = next((i for i, part in enumerate(key) if type(part) is object), None)
            if mark is None:
                call = (key, {})
            else:
                pairs = key[mark + 1 :]
                if len(pairs) % 2 or not all(isinstance(k, str) for k in pairs[0::2]):
                    return None
                call = (key[:mark], dict(zip(pairs[0::2], pairs[1::2])))
        try:
            text = repr(call)
            if not _same(ast.literal_eval(text), call):
                return None
            calls.append([_pair_digest(key, answer), text])
        except Exception:  # noqa: BLE001 - no text, or none that gives it back
            return None
    return calls


def _same(a: Any, b: Any) -> bool:
    """Equal, and of the same types all the way down (``1`` is not ``1.0``)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def _refill(memo: Any, calls: list[list[str]]) -> bool:
    """Call *memo* for each of *calls* whose answer it does not hold, as the
    run that filled it did; False when one fails."""
    import ast

    held = set(_item_digests(memo) or ())
    for call in calls:
        # A record an earlier layout wrote (``[text]``): not read, refused.
        if not (isinstance(call, (list, tuple)) and len(call) == 2 and all(isinstance(p, str) for p in call)):
            return False
        digest, text = call
        if digest in held:
            continue
        try:
            args, kwargs = ast.literal_eval(text)
            memo(*args, **kwargs)
        except Exception:  # noqa: BLE001 - it did not answer as the run's did
            return False
    return True


def _resolve(where: list[str], ns: Mapping[str, Any]) -> Any:
    if where[0] == "ns":
        value, steps = ns.get(where[1]), where[2:]
    else:
        mod = sys.modules.get(where[1])
        value = attribute_or(mod, where[2], None) if mod is not None else None
        steps = where[3:]
    for step in steps:
        if value is None:
            return None
        if step == "~":
            value = attribute_or(value, "__wrapped__", None)
            continue
        raw = vars(value).get(step[1:]) if instance_of(value, type) else attribute_or(value, step[1:], None)
        value = attribute_or(raw, "__func__", raw) if instance_of(raw, (staticmethod, classmethod)) else raw
    return value


def class_memo_record(cls: type, name: str, memo: Any) -> dict | None:
    """A record (as `used` writes one) of the answers *memo*, the memo the
    class *cls* holds under *name*, holds now: the class reached by its
    module and qualified name, the way a later process finds it once the
    module is imported. None when it cannot be found that way or its
    answers cannot be read: an entry cannot then be held against it."""
    module = attribute_or(cls, "__module__", None)
    qualname = attribute_or(cls, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    head, *rest = qualname.split(".")
    where = ["module", module, head, *("." + part for part in rest), "." + name]
    if _resolve(where, {}) is not memo:
        return None
    digests = _item_digests(memo)
    if digests is None:
        return None
    return {"where": where, "items": digests, "pure": False, "calls": None}


def unfilled(records: list[dict] | None, ns: Mapping[str, Any], *, refill: bool = True) -> str | None:
    """Why an entry cannot be served over the memos it read, or None. A memo
    of a pure function must not hold another answer for a key it read (one
    it lacks, a run would compute again, the same); any other must hold
    every key and answer it held when the statement ran. *refill*: False
    only checks (the forward probe, before the statement's place, where
    what the helper reads may still change): a pure memo lacking a key is
    refused there, as the lookup at the statement's place may refuse it
    once a cell above has run, and a variable the probe had left to the
    entry would then not be rebuilt."""
    if not records:
        return None
    for record in records:
        where = record.get("where") or []
        label = _label(where) if where else "?"
        memo = _resolve(where, ns) if where else None
        if not _is_user_memo(memo):
            return f"the helper memo {label} it read is not bound"
        if record.get("unread"):
            return f"the helper memo {label} it read holds more than cash checks"
        if record.get("cleared"):
            # Emptied before any call: what it holds now decides nothing.
            why = _change_refused(record, memo, label, set(), set())
            if why is not None:
                return why
            continue
        now = _item_digests(memo)
        recorded = set(record.get("items") or ())
        if now is None:
            return f"the helper memo {label} does not hold the answers the statement read"
        if record.get("delta") is not None:
            why = _change_refused(record, memo, label, recorded, set(now))
            if why is not None:
                return why
            continue
        if recorded <= set(now):
            continue
        if record.get("pure") and record.get("calls") is not None and _pure(memo):
            if not refill:
                return f"the helper memo {label} lacks answers the statement read"
            # Fill it now as the run did, while what the helper reads is
            # what it read then: a later cell can still shadow a builtin
            # it calls, and an empty memo would then compute otherwise.
            if not _refill(memo, record["calls"]) or not recorded <= set(_item_digests(memo) or ()):
                return f"the helper memo {label} holds another answer than the statement read"
        else:
            return f"the helper memo {label} does not hold the answers the statement read"
    return None


def _change_refused(record: dict, memo: Any, label: str, recorded: set[str], now: set[str]) -> str | None:
    """Why the change a run made to *memo* (a record `used` wrote with
    *changes*) cannot be put back here, or None: the memo must be one a hit
    can fill, hold what it held when the run began (exactly, once the run
    emptied it; the record names none of it when the run found no answer
    there, `_change_of`), and lack every key the run added (a cell above
    that filled one makes a run find the answer, and skip what the helper's
    body did)."""
    if not _unbounded(memo):
        return f"the helper memo {label} is bounded, and cash cannot put answers into it"
    if record.get("keys") and record.get("kd") != KEY_DIGESTS:
        return f"the keys it put into the helper memo {label} were recorded by another version of cash"
    if record.get("cleared"):
        return None
    if record.get("emptied"):
        if recorded != now:
            return f"the helper memo {label} does not hold what it held when the statement emptied it"
        return None
    if not recorded <= now:
        return f"the helper memo {label} does not hold the answers the statement read"
    keys = _key_digests(memo)
    if keys is None:
        return f"the helper memo {label} holds keys cash cannot digest"
    if not keys.isdisjoint(record.get("keys") or ()):
        return f"the helper memo {label} already holds an answer the statement computed"
    return None


#: ``memo -> (the keys it held, their digests)``: digested again when one
#: is not the very object it was (keys are hashable, and so do not change).
_key_digest_memo: weakref.WeakKeyDictionary[Any, tuple[list[Any], set[str]]] = weakref.WeakKeyDictionary()


def _key_digests(memo: Any) -> set[str] | None:
    """The digest of every key *memo* holds (`_key_digest`), or None when
    one cannot be read."""
    items = _items(memo)
    if items is None or len(items) > MAX_KEYED:
        return None
    keys = [key for key, _ in items]
    known = _key_digest_memo.get(memo)
    if known is not None and len(known[0]) == len(keys) and all(a is b for a, b in zip(known[0], keys)):
        return known[1]
    try:
        digests = {_change_key_digest(key) for key in keys}
    except Exception:  # noqa: BLE001 - a key with no digest
        return None
    _key_digest_memo[memo] = (keys, digests)
    return digests


def _answers_dict(memo: Any) -> dict | None:
    """The dict an unbounded C memo keeps its answers in (`_items`), or None."""
    try:
        size = memo.cache_info().currsize
        refs = get_referents(memo)
    except Exception:  # noqa: BLE001 - a wrapper that cannot be read
        return None
    if refs and refs[0] is type(memo):
        refs = refs[1:]
    cache = refs[0] if refs else None
    if type(cache) is not dict or len(cache) != size or cache is attribute_or(memo, "__dict__", None):
        return None
    return cache


#: The marker ``functools`` puts between positional and keyword arguments,
#: found once from a memo of its own (`_live_mark`).
_MARK: list[Any] = []


def _live_mark() -> Any:
    """The keyword marker this process's memos use, or None."""
    if not _MARK:
        probe = functools.lru_cache(maxsize=None)(lambda *args, **kwargs: None)
        probe(k=1)
        cache = _answers_dict(probe)
        key = next(iter(cache), None) if cache else None
        mark = key[0] if isinstance(key, tuple) and key and type(key[0]) is object else None
        _MARK.append(mark)
    return _MARK[0]


def _live_key(key: Any) -> Any:
    """*key* as unpickled, with the keyword marker the live one."""
    if type(key) is tuple and any(type(part) is object for part in key):
        mark = _live_mark()
        if mark is None:
            raise ValueError("the keyword marker of functools was not found")
        return tuple(mark if type(part) is object else part for part in key)
    return key


def put_back(records: list[dict] | None, changes: Any, ns: Mapping[str, Any]) -> Any:
    """Prepare to put the changes a served entry carries (`used` with
    *changes*) into the live memos: a function that does it, called once the
    outputs are restored, or a string saying why it cannot be done (the
    entry is then not served). Each key comes back as the run's: its digest
    is the one recorded."""
    plans: list[tuple[Any, bool, Any]] = []
    for record in records or ():
        index = record.get("delta")
        if index is None:
            continue
        where = record.get("where") or []
        label = _label(where) if where else "?"
        memo = _resolve(where, ns) if where else None
        try:
            change = changes[index]
            if record.get("plain"):
                # Checked plain, and equal after a pickle, when stored
                # (`_plain_keys`): the keys are the run's. Only as many as
                # stored, none taken for another.
                keys, answers = pickle.loads(change["keys"]), change["answers"]
                if record.get("marked"):
                    keys = [_live_key(key) for key in keys]
                pairs: Any = dict(zip(keys, answers, strict=True))
                same = len(pairs) == len(keys)
            else:
                pairs = [(_live_key(key), answer) for key, answer in change["pairs"]]
                digests = [_change_key_digest(key) for key, _ in pairs]
                same = record.get("kd") == KEY_DIGESTS and digests == list(record.get("keys") or ())
        except Exception:  # noqa: BLE001 - an entry this cannot read
            return f"the answers it put into the helper memo {label} cannot be read back"
        if not _is_user_memo(memo) or not _unbounded(memo) or _answers_dict(memo) is None:
            return f"the helper memo {label} is not one cash can put answers into"
        if not same:
            return f"the keys it put into the helper memo {label} come back as other keys"
        plans.append((memo, bool(record.get("emptied")), pairs))
    if not plans:
        return lambda: None

    def apply() -> None:
        for memo, emptied, pairs in plans:
            if emptied:
                memo.cache_clear()
            cache = _answers_dict(memo)
            if cache is None:
                continue
            cache.update(pairs)
            # Seen as the run left them: the next statement reading one
            # compares with these.
            items = _items(memo)
            if items is not None and len(items) <= MAX_KEYED:
                _changed_in_place(memo, items)
            else:
                _answers_seen.pop(memo, None)

    return apply


@contextlib.contextmanager
def lifted(records: list[dict] | None, changes: list[dict] | None, ns: Mapping[str, Any]) -> Iterator[None]:
    """Take the pairs a run put into its memos (`used` with *changes*) out
    of the memos while the share check walks the outputs with the answers:
    the entry holds them then, and the memo the restored ones, so the
    memo's reference to an answer the outputs hold is no holder. An answer
    anything else holds -- one an earlier statement put in the memo, a
    variable -- still is. Put back as they were after."""
    lifted_out: list[tuple[dict, list[tuple[Any, Any]], Any, list[int]]] = []
    try:
        for record in records or ():
            index = record.get("delta")
            if index is None or changes is None or index >= len(changes):
                continue
            memo = _resolve(record.get("where") or [], ns)
            cache = _answers_dict(memo) if _is_user_memo(memo) else None
            if cache is None:
                continue
            change = changes[index]
            if "pairs" in change:
                # The pairs themselves: no other tuple holds an answer.
                taken: list[Any] = []
                for pair in change["pairs"]:
                    if cache.get(pair[0], _UNSET) is pair[1]:
                        del cache[pair[0]]
                        taken.append(pair)
                lifted_out.append((cache, taken, None, []))
                continue
            # Emptied first, the memo holds the answers in the order the
            # change lists them: taken by position, and no tuple of a key
            # and its answer is made, one more holder of the answer.
            keys, answers = list(cache), change["answers"]
            at = [i for i, key in enumerate(keys[: len(answers)]) if cache[key] is answers[i]]
            for i in at:
                del cache[keys[i]]
            lifted_out.append((cache, [], (keys, answers), at))
        yield
    finally:
        for cache, taken, change, at in lifted_out:
            for key, answer in taken:
                cache.setdefault(key, answer)
            for i in at:
                cache.setdefault(change[0][i], change[1][i])


def change_answers(changes: list[dict] | None) -> list[Any]:
    """The answers of *changes* (`used`), in one list for the share check."""
    out: list[Any] = []
    for change in changes or ():
        out.extend(change["answers"] if "answers" in change else (pair[1] for pair in change["pairs"]))
    return out


def change_holders(changes: list[dict] | None) -> list[Any]:
    """The containers of *changes* (`used`) that hold its answers: cash's
    own, which the share check does not count as another holder."""
    out: list[Any] = []
    for change in changes or ():
        if "answers" in change:
            out.append(change["answers"])
        else:
            out.extend(change["pairs"])
    return out


def held_by_cash() -> list[Any]:
    """The containers through which this module holds memo keys and answers
    (`_digests`, `_answers_seen`): cash's own references, which the share
    check does not count as another holder."""
    held: list[Any] = [flat for flat, _ in list(_digests.values())]
    for seen in list(_answers_seen.values()):
        held.extend(seen.values())
    return held


def carries_changes(records: list[dict] | None) -> bool:
    """Whether an entry's memo records include a change a hit puts back."""
    return any(record.get("delta") is not None for record in records or ())


#: What a statement naming them reads or changes of a memo beyond its
#: answers: its counters, which an entry served does not give back.
_COUNTER_NAMES = ("cache_clear", "cache_info")

#: What a statement naming it reads of a memo: its counters, which neither
#: a served entry nor the answers it puts back give back.
_INFO_NAMES = ("cache_info",)

#: ``names -> code -> whether it or a function it calls names one of them``.
_clearing_code: dict[tuple[str, ...], weakref.WeakKeyDictionary[Any, bool]] = {
    _COUNTER_NAMES: weakref.WeakKeyDictionary(),
    _INFO_NAMES: weakref.WeakKeyDictionary(),
}


def _is_clear(value: Any, names: tuple[str, ...] = _COUNTER_NAMES) -> bool:
    """*value* is a memo's own ``cache_clear`` or ``cache_info`` (of
    *names*), under any name."""
    return (
        attribute_or(value, "__name__", None) in names and type(attribute_or(value, "__self__", None)) is _LRU_WRAPPER
    )


def _names_clear(fn: Any, names: tuple[str, ...] = _COUNTER_NAMES) -> bool:
    """*fn* names ``cache_clear`` or ``cache_info`` (of *names*; an
    attribute, or a string it builds a name from), or a global it names is
    a memo's."""
    code = attribute_or(fn, "__code__", None)
    if code is None:
        return False
    memo = _clearing_code[names]
    try:
        found = memo[code]
    except (KeyError, TypeError):
        from cash.decorator.closure_fold import iter_code_scopes

        scopes = list(iter_code_scopes(code))
        found = any(
            any(name in scope.co_names for name in names)
            or any(
                isinstance(c, str) and (any(name in c for name in names) or c.startswith("cache_"))
                for c in scope.co_consts
            )
            for scope in scopes
        )
        try:
            memo[code] = found
        except TypeError:
            pass
    if found:
        return True
    globals_ = attribute_or(fn, "__globals__", None)
    if not isinstance(globals_, dict):
        return False
    from cash.decorator.closure_fold import iter_code_scopes

    return any(_is_clear(globals_.get(name), names) for scope in iter_code_scopes(code) for name in scope.co_names)


def _text_names_counters(code: str, names: tuple[str, ...] = _COUNTER_NAMES) -> bool:
    """The statement's *code* names ``cache_clear`` or ``cache_info``, as an
    attribute or a name, or holds a string with one or starting with
    ``cache_`` (``getattr(f, "cache_" + "info")``), as `_names_clear` asks
    of a function. Not a name that only contains one (``cache_info_rows``).
    Text that does not parse is taken by its letters."""
    import ast

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return any(name in code for name in names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in names:
                return True
        elif isinstance(node, ast.Name):
            if node.id in names:
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("cache_") or any(name in node.value for name in names):
                return True
    return False


def clears(
    code: str | None, ns: Mapping[str, Any] | None, fn: Any = None, names: tuple[str, ...] = _COUNTER_NAMES
) -> bool:
    """Whether a statement's *code* (or the call of *fn*) can empty a memo
    or read its counters: it, or a function of the user's it reaches, names
    ``cache_clear`` or ``cache_info``, a string with one
    (``getattr(f, "cache_" + "clear")``), or a name bound to a memo's.
    ``cache_info()`` reads counters a served entry leaves as they were (an
    upstream cell filling another key); ``cache_clear()`` resets the
    counters `used` compares; one reached some other way is caught by the keys it leaves
    (`_held`, which compares the very key and answer objects held before,
    kept alive for that) unless it fills the memo again with those same
    objects."""
    from cash.notebook.callee_reach import names_read, reached_user_code

    if fn is not None:
        if _names_clear(fn, names):
            return True
        name = attribute_or(fn, "__name__", None)
        globals_ = attribute_or(fn, "__globals__", None)
        if not (isinstance(name, str) and isinstance(globals_, dict) and globals_.get(name) is fn):
            return False
        code, ns = name, globals_
    elif code is not None:
        if _text_names_counters(code, names):
            return True
        try:
            if ns is not None and any(not attrs and _is_clear(ns.get(root), names) for root, attrs in names_read(code)):
                return True
        except Exception:  # noqa: BLE001 - not parsed: the reach below decides
            pass
    try:
        reached = reached_user_code(code or "", ns)
    except Exception:  # noqa: BLE001 - not followed: not known to clear
        return False
    return any(_names_clear(found, names) for found in reached.functions)


def reads_counters(code: str | None, ns: Mapping[str, Any] | None) -> bool:
    """Whether a statement's *code*, or a function of the user's it reaches,
    can read a memo's counters (``cache_info``, as `clears` finds a name):
    a served entry gives back neither the counters nor what a run that
    reads them does with them, so such a statement runs every time."""
    return clears(code, ns, names=_INFO_NAMES)
