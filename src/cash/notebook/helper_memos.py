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
import logging
import sys
import types
import weakref
from collections.abc import Iterator, Mapping
from typing import Any

from cash._user_attrs import attribute_or, instance_of
from cash.install_paths import is_user_code_file, is_user_module
from cash.value_hash import compute_hash

__all__ = ["FILLED", "MAX_KEYED", "new_statement", "unfilled", "used", "watched"]

logger = logging.getLogger(__name__)

#: The type ``functools.cache`` and ``functools.lru_cache`` return.
_LRU_WRAPPER = attribute_or(functools, "_lru_cache_wrapper", None)

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
_answers_seen: weakref.WeakKeyDictionary[Any, dict[int, tuple[Any, str]]] = weakref.WeakKeyDictionary()

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
    return tuple(id(v) for v in list(ns.values()) if type(v) in kinds or instance_of(v, type))


#: Bumped before each statement (`new_statement`).
_statement = 0
#: ``id of a namespace -> (its holders, its memos outside classes, its
#: classes)`` and ``(statement, the user's modules then, their memos)``: a
#: namespace is looked through again only when a function or class was
#: bound in it, and the modules once per statement or when one was
#: imported (walking hundreds of modules per call made a 50,000-call loop
#: take over 30 s).
_ns_reached: dict[int, tuple[tuple[int, ...], list[tuple[list[str], Any]], list[tuple[str, type]]]] = {}
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
        known = (holders, *_ns_memos(ns))
        _ns_reached[id(ns)] = known
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
        digests: list[str] | None = sorted({f"{_key_digest(k)}:{compute_hash(v)}" for k, v in items})
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


def _changed_in_place(memo: Any, items: list[tuple[Any, Any]]) -> bool | None:
    """Whether an answer *memo* holds was changed in place since the last
    statement that called it left it, None when one cannot be told (an
    answer that can change in place and was not seen then); and remember
    them as they are now."""
    seen = _answers_seen.get(memo, {})
    now: dict[int, tuple[Any, str]] = {}
    verdict: bool | None = False
    for _, answer in items:
        if _fixed(answer):
            continue
        try:
            digest = compute_hash(answer)
        except Exception:  # noqa: BLE001 - an answer with no digest
            verdict = None if verdict is False else verdict
            continue
        now[id(answer)] = (answer, digest)
        before = seen.get(id(answer))
        if before is None or before[0] is not answer:
            verdict = None if verdict is False else verdict
        elif before[1] != digest:
            verdict = True
    _answers_seen[memo] = now
    return verdict


def _held(memo: Any) -> tuple[frozenset[tuple[int, int]], list[tuple[Any, Any]]] | None:
    """Which key and answer objects *memo* holds, or None past `MAX_KEYED`:
    a run that empties it (``cache_clear`` by any name) and fills it again
    can leave its counters as they were, not these. The objects are kept
    with their ids, so a refill cannot put a new key at a freed address."""
    items = _items(memo)
    if items is None or len(items) > MAX_KEYED:
        return None
    return frozenset((id(key), id(answer)) for key, answer in items), items


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
) -> tuple[str, str] | list[dict] | None:
    """After a statement ran: ``(FILLED, why)`` when what it did to a memo
    leaves it otherwise than a served entry would, else each memo it called
    with the answers it held then. *clears*: the code that ran names
    ``cache_clear``, which resets the counters this reads. *ns*: look for
    memos the run made or bound too (a ``def`` in it, a module it imported),
    each taken as empty before it."""
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
                before.append((where, memo, 0, 0, 0, (frozenset(), [])))
    if not before:
        return None
    out: list[dict] = []
    why: str | None = None
    for where, memo, hits, misses, size, held in before:
        try:
            info = memo.cache_info()
        except Exception:  # noqa: BLE001 - a wrapper whose counters raise
            continue
        if (info.hits, info.misses, info.currsize) == (hits, misses, size) and not clears:
            now = _held(memo)
            if held is None or (now is not None and now[0] == held[0]):
                continue
            clears = True  # emptied and filled again to the same counts
        label = _label(where)
        items = _items(memo)
        moved = _changed_in_place(memo, items) if items is not None else None
        if why is not None:
            continue  # already told; the rest only remember their answers
        if clears:
            why = f"It may empty the helper memo {label} or read its counters (cache_clear, cache_info)"
        elif info.misses < misses or info.hits < hits:
            why = f"It emptied the helper memo {label}"
        elif items is None or len(items) > MAX_KEYED:
            why = f"It read the helper memo {label}, which holds more answers than cash checks ({MAX_KEYED})"
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
        return FILLED, f"{why} (statement re-executes)"
    return out or None


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
            calls.append([f"{_key_digest(key)}:{compute_hash(answer)}", text])
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
        now = _item_digests(memo)
        recorded = set(record.get("items") or ())
        if now is None:
            return f"the helper memo {label} does not hold the answers the statement read"
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


#: What a statement naming them reads or changes of a memo beyond its
#: answers: its counters, which an entry served does not give back.
_COUNTER_NAMES = ("cache_clear", "cache_info")

#: ``code -> whether it or a function it calls names cache_clear or cache_info``.
_clearing_code: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()


def _is_clear(value: Any) -> bool:
    """*value* is a memo's own ``cache_clear`` or ``cache_info``, under any
    name."""
    return (
        attribute_or(value, "__name__", None) in _COUNTER_NAMES
        and type(attribute_or(value, "__self__", None)) is _LRU_WRAPPER
    )


def _names_clear(fn: Any) -> bool:
    """*fn* names ``cache_clear`` or ``cache_info`` (an attribute, or a
    string it builds a name from), or a global it names is a memo's."""
    code = attribute_or(fn, "__code__", None)
    if code is None:
        return False
    try:
        found = _clearing_code[code]
    except (KeyError, TypeError):
        from cash.decorator.closure_fold import iter_code_scopes

        scopes = list(iter_code_scopes(code))
        found = any(
            any(name in scope.co_names for name in _COUNTER_NAMES)
            or any(
                isinstance(c, str) and (any(name in c for name in _COUNTER_NAMES) or c.startswith("cache_"))
                for c in scope.co_consts
            )
            for scope in scopes
        )
        try:
            _clearing_code[code] = found
        except TypeError:
            pass
    if found:
        return True
    globals_ = attribute_or(fn, "__globals__", None)
    if not isinstance(globals_, dict):
        return False
    from cash.decorator.closure_fold import iter_code_scopes

    return any(_is_clear(globals_.get(name)) for scope in iter_code_scopes(code) for name in scope.co_names)


def _text_names_counters(code: str) -> bool:
    """The statement's *code* names ``cache_clear`` or ``cache_info``, as an
    attribute or a name, or holds a string with one or starting with
    ``cache_`` (``getattr(f, "cache_" + "info")``), as `_names_clear` asks
    of a function. Not a name that only contains one (``cache_info_rows``).
    Text that does not parse is taken by its letters."""
    import ast

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return any(name in code for name in _COUNTER_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _COUNTER_NAMES:
                return True
        elif isinstance(node, ast.Name):
            if node.id in _COUNTER_NAMES:
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("cache_") or any(name in node.value for name in _COUNTER_NAMES):
                return True
    return False


def clears(code: str | None, ns: Mapping[str, Any] | None, fn: Any = None) -> bool:
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
        if _names_clear(fn):
            return True
        name = attribute_or(fn, "__name__", None)
        globals_ = attribute_or(fn, "__globals__", None)
        if not (isinstance(name, str) and isinstance(globals_, dict) and globals_.get(name) is fn):
            return False
        code, ns = name, globals_
    elif code is not None:
        if _text_names_counters(code):
            return True
        try:
            if ns is not None and any(not attrs and _is_clear(ns.get(root)) for root, attrs in names_read(code)):
                return True
        except Exception:  # noqa: BLE001 - not parsed: the reach below decides
            pass
    try:
        reached = reached_user_code(code or "", ns)
    except Exception:  # noqa: BLE001 - not followed: not known to clear
        return False
    return any(_names_clear(found) for found in reached.functions)
