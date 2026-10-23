"""What a reducer of the user's that cash ran may have written.

Cash pickles a statement's values to digest and store them, and a pickle
runs the value's own ``__reduce__``, ``__reduce_ex__``, ``__getstate__``,
``__getnewargs__`` or ``__getnewargs_ex__`` (or a reducer of the user's
registered with ``copyreg``): the user's code, run by cash, between
statements. One that writes (``df.at[3, 'o'] = ...``) changes a variable
no statement named, and whatever read it before would be served what it
read then. `writers` reads each such reducer's body, and the functions of
the user's it calls, for a write; `reach` names the variables a writing
one's code can reach, which then move on (`StatementProcessor`).

A body that only reads -- ``return self.__dict__.copy()`` -- writes
nothing, and nothing moves; nor does building a new object of the user's
class (``State(self.rows)``), whose ``__init__`` sets only what is new. A
global it hands to a library call (``np.copyto(ARR, 5)``) counts as
written: what the call does is not read, and watching it change would be
too late, as the reducer has then changed what plain Python never does.

A writer whose code reaches a variable of the notebook is not run at all
(`refuses`): cash's pickle refuses the object, as it refuses any it cannot
pickle, so the statement is not stored and nothing is written, as in plain
Python, where no pickle runs it. So is one whose code cannot be read, and
one that writes the object it pickles: what either changes cannot be told,
and not running it changes nothing. What a writer reaches is found by
object, through its functions' own globals: a module's ``SEEN`` the notebook
holds as ``seen``, or the module itself when the notebook reads it as
``rmod.SEEN``, never a notebook variable that only shares a name with it,
or only an equal number or string.

The notebook arms this while cash is on (`watch`); ``%cash_off`` disarms it
(`unwatch`), and cash's pickles run every reducer again.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import copyreg
import functools
import inspect
import json
import pickle
import sys
import textwrap
import types
import weakref
from collections.abc import Iterator, Mapping
from typing import Any

from cash._user_attrs import attribute_or, instance_of
from cash.analysis.ast_util import bytecode_global_refs
from cash.analysis.callee_effects import (
    free_vars_mutated_in_function,
    params_mutated_in_function,
    parse_function_source,
    positional_param_names,
    source_global_rebinds,
)
from cash.analysis.mutations import MutationVisitor
from cash.kept_state import next_mark, set_reducer_veto, user_state_runs, user_state_types_since, users_function
from cash.source_reading import getsourcelines

__all__ = ["Writer", "reach", "refused_since", "refuses", "runs_now", "unwatch", "watch", "writers", "writes_into"]

#: The methods a pickle runs on an object, found on its class.
_REDUCERS = ("__reduce__", "__reduce_ex__", "__getstate__", "__getnewargs__", "__getnewargs_ex__")


class Writer:
    """A class whose reducer of the user's writes: its *name* for the
    warning, and the functions of the user's its pickle runs (*functions*,
    None when one of them could not be read)."""

    __slots__ = ("functions", "handed", "name", "written")

    def __init__(
        self,
        name: str,
        functions: list[Any] | None,
        written: frozenset[str] = frozenset(),
        handed: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.functions = functions
        #: The global names its code changes, as read from it.
        self.written = written
        #: The global names it hands to a library call, which may change
        #: them: taken as written (`reach`).
        self.handed = handed


def runs_now() -> int:
    """The mark `writers` and `refused_since` count from:
    `kept_state.user_state_runs`. Taken as each statement begins, it also
    forgets what `refuses` decided for the statement before."""
    _REFUSES.clear()
    return user_state_runs()


#: The notebook namespace `refuses` judges a writer's reach against.
_NAMESPACE: list[Mapping[str, Any] | None] = [None]
#: `refuses` by class, for the statement being processed.
_REFUSES: dict[Any, bool] = {}
#: The `runs_now` mark of the last pickle `refuses` stopped, by class.
_REFUSED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def watch(user_ns: Mapping[str, Any]) -> None:
    """Judge from now on what writers reach against *user_ns*, and refuse
    to run one that reaches a variable of it (`refuses`)."""
    _NAMESPACE[0] = user_ns
    set_reducer_veto(refuses, refusal_reason)


def unwatch() -> None:
    """Stop judging: cash's pickles run every reducer of the user's, as
    outside a notebook (``%cash_off``)."""
    _NAMESPACE[0] = None
    set_reducer_veto(None)


def refuses(t: type) -> bool:
    """Should cash's pickle refuse an instance of *t* rather than run its
    reducer of the user's? Yes when its code, read, writes and reaches a
    variable of the watched namespace (`reach`), or what it reaches cannot
    be told: its code cannot be read, or it writes the object it pickles.
    A writer of a counter of its own module's, which the notebook holds
    neither as a variable nor through the module, runs, as it did."""
    try:
        known = _REFUSES.get(t)
    except TypeError:  # a class whose metaclass makes it unhashable
        known = None
    user_ns = _NAMESPACE[0]
    if known is None:
        writer = _writer(t)
        if writer is None or user_ns is None:
            known = False
        elif writer.functions is None:
            # What code that cannot be read, or that writes the object being
            # pickled, changes is not told: run, it moved every variable.
            # Not run, it changes nothing, and only this value is not stored.
            known = True
        else:
            names = reach([writer], user_ns)
            known = names is None or bool(names)
        with contextlib.suppress(TypeError):
            _REFUSES[t] = known
    if known:
        mark = next_mark()
        with contextlib.suppress(TypeError):
            _REFUSED[t] = mark
    return known


def refusal_reason(t: type) -> str:
    """Why `refuses` stops *t*'s reducer, for the error that says so."""
    writer = _writer(t)
    if writer is not None and writer.functions is None:
        return "cash cannot tell what it changes (its source cannot be read, or it changes the object it pickles)"
    return "it changes other values"


def refused_since(since: int) -> list[str]:
    """The names of the classes `refuses` stopped since the mark *since*."""
    return sorted(
        attribute_or(t, "__qualname__", None) or "an object" for t, mark in list(_REFUSED.items()) if mark > since
    )


def writers(since: int) -> list[Writer]:
    """The classes whose reducer of the user's ran in a pickle of cash's
    since the mark *since* (`runs_now`) and writes. One that ran for a
    class cash could not name is a writer whose functions are unknown."""
    ran, unnamed = user_state_types_since(since)
    found = [writer for t in ran if (writer := _writer(t)) is not None]
    if unnamed:
        found.append(Writer("an object", None))
    return found


def writes_into(since: int, user_ns: Mapping[str, Any]) -> bool:
    """Has a reducer that writes, run since the mark *since*, reached a
    variable of *user_ns* (`reach`)? One whose code reaches none (a counter
    of its own module's) leaves nothing a statement reads changed, and a
    hit that skips it gives what plain Python gives, where it never runs.
    One refused since (`refuses`) left an object cash could not pickle."""
    if refused_since(since):
        return True
    found = writers(since)
    if not found:
        return False
    names = reach(found, user_ns)
    return names is None or bool(names)


def reach(found: list[Writer], user_ns: Mapping[str, Any]) -> set[str] | None:
    """The variables of *user_ns* the code of the writers *found* can reach:
    those bound to an object a global name of their functions looks up, in
    that function's own globals, or to the module whose globals a writer
    changes (``rmod``, read as ``rmod.SEEN``), or to a function of it. None
    when a writer's functions are unknown, or one writes the object it
    pickles (which variables hold it is not known): every variable may have
    changed.

    A global handed to a library call (`Writer.handed`) counts as written:
    what the call does to it is not read, and plain Python never runs it."""
    reached: dict[int, Any] = {}
    written: set[int] = set()
    #: Names bound to a value no write can change (a small int, an interned
    #: string): matched by name in the writer's own namespace, never by id.
    rebound: set[str] = set()
    modules: set[int] = set()
    for writer in found:
        if writer.functions is None:
            return None
        for function in writer.functions:
            g = attribute_or(function, "__globals__", None)
            if not isinstance(g, dict):
                return None
            changes_module = False
            refs = list(bytecode_global_refs(function))
            named = {chain[0] for chain in refs}
            # Written but not named by its code: a parameter's default
            # (``_c=CFG``) stands for the global.
            refs += [(n,) for n in sorted((writer.written | writer.handed) - named) if n in g]
            for cell_name, value in _cells(function).items():
                if cell_name in writer.written and type(value) not in _ATOMS:
                    # A closure's cell (``acc.append(1)``): the notebook may
                    # hold its object (``ACC``).
                    written.add(id(value))
                    reached[id(value)] = value
            for chain in refs:
                value = g.get(chain[0], _ABSENT)
                if value is _ABSENT:
                    continue
                is_written = chain[0] in writer.written or chain[0] in writer.handed
                if type(value) in _ATOMS:
                    if is_written:
                        changes_module = True
                        if g is user_ns:
                            rebound.add(chain[0])
                    continue
                # A module, class or function it only names (``time.sleep``)
                # is code it runs, not data it writes; one it writes counts.
                if is_written:
                    written.add(id(value))
                elif type(value) in _CODE_TYPES:
                    continue
                reached[id(value)] = value
                # What it reaches of its module, the notebook reads through
                # the module too.
                changes_module = True
            if changes_module and g is not user_ns:
                module = sys.modules.get(attribute_or(function, "__module__", None) or "")
                if module is not None and attribute_or(module, "__dict__", None) is g:
                    reached[id(module)] = module
                    written.add(id(module))
                    modules.add(id(g))
    if not reached and not rebound:
        return set()
    found_names = set(rebound & user_ns.keys())
    for name, value in list(user_ns.items()):
        if name.startswith("_"):
            continue
        if modules and type(value) is types.FunctionType and id(attribute_or(value, "__globals__", None)) in modules:
            found_names.add(name)  # ``from rmod import seen_count``
        elif id(value) in reached and (id(value) in written or type(value) not in _CODE_TYPES):
            found_names.add(name)
    return found_names


def _cells(function: Any) -> dict[str, Any]:
    """*function*'s closure, name to what each cell holds (an empty one left out)."""
    code = attribute_or(function, "__code__", None)
    out: dict[str, Any] = {}
    for cell_name, cell in zip(
        attribute_or(code, "co_freevars", ()), attribute_or(function, "__closure__", None) or ()
    ):
        try:
            out[cell_name] = cell.cell_contents
        except ValueError:
            continue
    return out


_CODE_TYPES = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)
#: Values no write changes, and that many unrelated names share by id.
_ATOMS = frozenset({int, float, complex, bool, str, bytes, type(None)})
_ABSENT = object()


#: `_writer` by class: None for a class whose reducers write nothing.
_WRITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PURE = object()


def _writer(t: type) -> Writer | None:
    if copyreg.dispatch_table.get(t) is not None:
        # Registered with ``copyreg``, perhaps after it was first decided.
        return _decide(t)
    try:
        found = _WRITERS.get(t)
    except TypeError:
        return _decide(t)
    if found is None:
        found = _WRITERS[t] = _decide(t) or _PURE
    return None if found is _PURE else found


def _decide(t: type) -> Writer | None:
    functions = _reducers_of(t)
    name = attribute_or(t, "__qualname__", None) or attribute_or(t, "__name__", "an object")
    seen: dict[int, Any] = {}
    roots = {id(function) for function in functions}
    pending = list(functions)
    #: By callee, each call that hands it something, and the caller's body.
    handed: dict[int, list[tuple[ast.Call | None, ast.AST]]] = {}
    writes = False
    written: set[str] = set()
    to_libraries: set[str] = set()
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen[id(function)] = function
        body = _body(function)
        if body is None or (id(function) in roots and _writes_itself(body)):
            return Writer(name, None)
        if _reads_a_namespace(body, attribute_or(function, "__globals__", {})):
            # ``globals()['CFG']['rows'].append(1)``, ``vars(OBJ)['seen']``:
            # what it changes is reached by a string.
            return Writer(name, None)
        changed = free_vars_mutated_in_function(body) | _changed_through_copies(function, body)
        to_libraries |= _handed_to_libraries(function, body) - changed
        if changed or _rebinds(function):
            written |= changed
            writes = True
        elif id(function) not in roots and not _changes_only_fresh(body, handed.get(id(function), [])):
            # A function of the user's the reducer hands something to, and
            # which changes what it is handed: what was handed may be a
            # variable's object.
            writes = True
        for callee, call in _users_callees(function, body, t):
            handed.setdefault(id(callee), []).append((call, body))
            pending.append(callee)
    if writes or to_libraries:
        return Writer(name, list(seen.values()), frozenset(written), frozenset(to_libraries))
    return None


def _changes_only_fresh(
    body: ast.FunctionDef | ast.AsyncFunctionDef, calls: list[tuple[ast.Call | None, ast.AST]]
) -> bool:
    """Does the callee *body* change only what each of its *calls* made
    fresh for it? ``_drop(self.__dict__.copy(), SKIP)`` hands a new dict,
    and ``d.pop(k)`` changes nothing else: no variable's object changes.
    Only a call by name is read; a parameter it changes past its top level
    (``d['rows'].append``) is fresh only from a deep copy."""
    changed = params_mutated_in_function(body)
    if not changed:
        return True
    if not calls:
        return False
    params = positional_param_names(body)
    for call, caller in calls:
        if call is None or not isinstance(call.func, ast.Name):
            return False
        shift = 0
        if params[:1] in (["self"], ["cls"]):
            # A method's arguments shift. Only a class's ``__init__``, run
            # by building one (``State(rows)``), is read: its ``self`` is
            # new, and setting its attributes changes nothing else.
            if not _builds(call, caller, body):
                return False
            if "self" in changed and not _sets_only_attributes(body, "self"):
                return False
            shift = 1
        given = {params[i + shift]: arg for i, arg in enumerate(call.args) if i + shift < len(params)}
        given.update((k.arg, k.value) for k in call.keywords if k.arg is not None)
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
            return False
        for param in changed - ({"self"} if shift else set()):
            arg = given.get(param)
            depth = _fresh(arg, caller) if arg is not None else None
            if depth is None or (depth == "top" and not _changes_only_itself(body, param)):
                return False
    return True


#: Calls that build a new top-level object from what they are given.
_SHALLOW_COPIES = frozenset({"dict", "list", "set", "bytearray"})


def _fresh(arg: ast.expr, caller: ast.AST, depth: int = 0) -> str | None:
    """How new the object *arg* evaluates to is in *caller*: ``"deep"``
    (``copy.deepcopy(x)``), ``"top"`` (``x.copy()``, ``dict(x)``, a
    literal: new, its items shared), or None (it may be anyone's)."""
    if isinstance(arg, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return "top"
    if isinstance(arg, ast.Call):
        func = arg.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "copy":
            return {"deepcopy": "deep", "copy": "top"}.get(func.attr)
        if isinstance(func, ast.Name):
            return "deep" if func.id == "deepcopy" else "top" if func.id in _SHALLOW_COPIES else None
        if isinstance(func, ast.Attribute) and func.attr == "copy" and not arg.args and not arg.keywords:
            return "top"
        return None
    if isinstance(arg, ast.Name) and depth < 4:
        # A local bound once, to something fresh.
        bound = [
            node.value
            for node in ast.walk(caller)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
            and any(isinstance(t, ast.Name) and t.id == arg.id for t in _targets(node))
        ]
        params = {a.arg for a in ast.walk(caller) if isinstance(a, ast.arg)}
        if len(bound) == 1 and bound[0] is not None and arg.id not in params and not _loops_bind(caller, arg.id):
            return _fresh(bound[0], caller, depth + 1)
    return None


def _targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return [t for target in node.targets for t in ast.walk(target)]
    return [node.target]


def _loops_bind(caller: ast.AST, name: str) -> bool:
    """Is *name* bound in *caller* other than by assignment: a loop, ``with``,
    ``except``, comprehension or import, a ``global`` or ``nonlocal``?"""
    for node in ast.walk(caller):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(node.target)):
                return True
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            if any(isinstance(t, ast.Name) and t.id == name for t in ast.walk(node.optional_vars)):
                return True
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (a.asname or a.name.split(".")[0]) == name for a in node.names
        ):
            return True
    return False


def _builds(call: ast.Call, caller: ast.AST, body: ast.AST) -> bool:
    """Is *call* (by name) the building of a class whose ``__init__`` is
    *body*? Told by the name: `_users_callees` resolved it to that
    ``__init__`` only through a class."""
    return isinstance(body, ast.FunctionDef) and body.name == "__init__" and isinstance(call.func, ast.Name)


def _sets_only_attributes(body: ast.AST, param: str) -> bool:
    """Does *body* use *param* only to set or delete its own attributes
    (``self.d = d``, ``self.n += 1``) or to read them? Taking one out and
    changing it (``self.d.append``) may change what was handed in."""
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(body):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    for node in ast.walk(body):
        if not (isinstance(node, ast.Name) and node.id == param):
            continue
        parent = parents.get(id(node))
        if not (isinstance(parent, ast.Attribute) and parent.value is node):
            return False
        if isinstance(parent.ctx, (ast.Store, ast.Del)):
            continue
        # A read of an attribute: fine unless something is done to it.
        outer = parents.get(id(parent))
        if isinstance(outer, (ast.Attribute, ast.Subscript)) and outer.value is parent:
            return False
        if isinstance(outer, ast.Call) and outer.func is parent:
            return False  # a method of its own, which may change anything
    return True


#: Builtins that read what they are handed and change none of it.
_READS_ONLY = frozenset(
    {
        "len",
        "isinstance",
        "issubclass",
        "type",
        "id",
        "hash",
        "repr",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "tuple",
        "list",
        "dict",
        "set",
        "frozenset",
        "sorted",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "getattr",
        "hasattr",
        "vars",
        "abs",
        "round",
        "enumerate",
        "zip",
        "range",
        "print",
        "format",
        "map",
        "filter",
        "iter",
        "reversed",
        "callable",
        "ord",
        "chr",
        "ascii",
        "bin",
        "hex",
        "oct",
        "divmod",
        "pow",
        "complex",
        "slice",
        "bytearray",
    }
)

#: Modules whose functions read what they are handed and change none of it
#: (``itertools.chain(LST)``, ``math.fsum(XS)``).
_MODULES_THAT_READ = frozenset({"itertools", "math", "cmath", "statistics", "string", "textwrap"})

#: Functions, by module and name, that read what they are handed, of
#: modules with others that do not: ``json.dump(D, LOG)`` writes to ``LOG``,
#: ``copy.copy(CFG)`` shares what ``CFG`` holds, ``copy.deepcopy(D, MEMO)``
#: fills ``MEMO`` (`_reads_only_here`).
_FUNCTIONS_THAT_READ = frozenset({("json", "dumps"), ("json", "loads"), ("copy", "deepcopy")})

#: Values nothing can change: handing one to a library is no write.
_NEVER_CHANGED = frozenset({str, bytes, int, float, complex, bool, type(None), range})


def _handed_to_libraries(function: Any, body: ast.AST) -> set[str]:
    """The global names *body* hands, as an argument, to a call of a
    function that is not the user's (``np.copyto(ARR, 5)``) and not one
    that only reads. What such a call does to them is not read. A value
    nothing can change (``KEY = 'a'``, ``DT = np.float64``) is never one."""
    g = attribute_or(function, "__globals__", {})
    local = {a.arg for a in ast.walk(body) if isinstance(a, ast.arg)} | {
        n.id for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    found: set[str] = set()
    # What an import inside the body binds (``import copy``) is read as the
    # module, or the module's function, it names.
    imported = _imported_in(body)
    g = {**g, **imported} if imported else g
    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id in _READS_ONLY and node.func.id not in g:
                continue
            callee = g.get(node.func.id)
            if callee is not None and (_users(callee) or _only_reads(callee, node, g)):
                continue  # the user's is read through `_users_callees`
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Constant):
                continue  # ``'-'.join(PARTS)``: a method of a constant
            if isinstance(node.func.value, ast.Name):
                holder = g.get(node.func.value.id)
                if node.func.value.id in ("self", "cls") or (instance_of(holder, type) and _users(holder)):
                    continue
                if node.func.value.id not in local and node.func.value.id in g and type(holder) in _NEVER_CHANGED:
                    continue  # ``SEP.join(PARTS)``: a method of a str (``list.append(LST, 1)`` is not)
                if instance_of(holder, types.ModuleType) and _only_reads(
                    attribute_or(holder, node.func.attr, None), node, g
                ):
                    continue
        for arg in [*node.args, *(k.value for k in node.keywords)]:
            if isinstance(arg, ast.Starred):
                arg = arg.value
            if isinstance(arg, ast.Name) and arg.id not in local and arg.id in g and not _never_changed(g[arg.id]):
                found.add(arg.id)
    return found


def _imported_in(body: ast.AST) -> dict[str, Any]:
    """The names an ``import`` in *body* binds, to the loaded module or the
    module's attribute they name; one not loaded yet is left out."""
    bound: dict[str, Any] = {}
    for node in ast.walk(body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    module = sys.modules.get(alias.name)
                    if module is not None:
                        bound[alias.asname] = module
                else:
                    top = alias.name.partition(".")[0]
                    if (module := sys.modules.get(top)) is not None:
                        bound[top] = module
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            module = sys.modules.get(node.module)
            for alias in node.names:
                if module is not None and (value := attribute_or(module, alias.name, None)) is not None:
                    bound[alias.asname or alias.name] = value
    return bound


def _only_reads(callee: Any, call: ast.Call, g: Mapping[str, Any]) -> bool:
    """Does *call* of *callee* only read what it is handed: a function of a
    module whose functions only read (``from itertools import chain``), one
    of `_FUNCTIONS_THAT_READ` (``copy.deepcopy`` with no memo), or
    ``functools.reduce`` with no initial value of the notebook's and a
    function that does not change its first argument (``operator.iadd``
    grows the first item, or ``ACC``)."""
    if callee is None:
        return False
    module = attribute_or(callee, "__module__", None)
    if module in _MODULES_THAT_READ:
        return True
    name = attribute_or(callee, "__name__", None)
    if (module, name) in _FUNCTIONS_THAT_READ:
        return name != "deepcopy" or (len(call.args) == 1 and not call.keywords)
    if callee is not functools.reduce or call.keywords or not call.args or len(call.args) > 3:
        return False
    if len(call.args) == 3 and _of_the_notebook(call.args[2], g):
        return False
    return not _changes_its_first(call.args[0], g)


def _of_the_notebook(arg: ast.expr, g: Mapping[str, Any]) -> bool:
    """Is *arg* built from a global that something can change?"""
    return any(isinstance(node, ast.Name) and node.id in g and not _never_changed(g[node.id]) for node in ast.walk(arg))


def _changes_its_first(fn: ast.expr, g: Mapping[str, Any]) -> bool:
    """Is the function *fn* names one that may change its first argument:
    an in-place operator (``operator.iadd``), or anything but a lambda or
    another of ``operator``'s functions?"""
    if isinstance(fn, ast.Lambda):
        return bool(free_vars_mutated_in_function(fn)) or _mutates_its_first(fn)
    value: Any = None
    if isinstance(fn, ast.Name):
        value = g.get(fn.id)
    elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        value = attribute_or(g.get(fn.value.id), fn.attr, None)
    name = attribute_or(value, "__name__", "")
    return attribute_or(value, "__module__", None) not in ("operator", "_operator") or name.startswith("i")


def _mutates_its_first(fn: ast.Lambda) -> bool:
    """Does the lambda *fn* change one of its parameters in place?"""
    visitor = _RunningMutations()
    visitor.visit(fn.body)
    return any(m.variable in {a.arg for a in fn.args.args} for m in visitor.mutations)


class _RunningMutations(MutationVisitor):
    """The mutations code makes as it runs: not those of a ``def``,
    ``lambda`` or ``class`` it builds."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None


def _changed_through_copies(function: Any, body: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The globals *body* changes through a local bound from an expression
    that reads one of them (``c = CFG``, ``c = CFG['rows']``, ``c =
    copy.copy(CFG)``, ``c = OBJ.seen``, ``for c in [CFG]``, ``c, k = CFG,
    1``, ``len(c := CFG['rows'])``, ``with nullcontext(CFG) as c``) or from
    a parameter whose default is one (``_c=CFG``): a change made through the
    local may change what the global holds. Fails closed: any expression
    reading the global counts but a deep copy and a few calls whose result
    holds nothing of it (`_HOLDS_NOTHING_OF_IT`). Any change counts, the
    top-level one of a copy too."""
    g = attribute_or(function, "__globals__", {})
    params = {a.arg for a in ast.walk(body.args) if isinstance(a, ast.arg)}
    aliases: dict[str, set[str]] = {}
    arguments = body.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaulted = [
        *zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults),
        *((a, d) for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults) if d is not None),
    ]
    for arg, default in defaulted:
        source = _globals_read(default, g, set(), aliases)
        if source:
            aliases[arg.arg] = source
    bindings = list(_bindings(body))
    grown = True
    while grown:
        grown = False
        for targets, value in bindings:
            source = _globals_read(value, g, params, aliases)
            if not source:
                continue
            for target in targets:
                known = aliases.setdefault(target, set())
                if not source <= known:
                    known |= source
                    grown = True
    if not aliases:
        return set()
    visitor = _RunningMutations()
    for stmt in body.body:
        visitor.visit(stmt)
    out: set[str] = set()
    for mutation in visitor.mutations:
        out |= aliases.get(mutation.variable, set())
    return out


def _reads_a_namespace(body: ast.AST, g: Mapping[str, Any]) -> bool:
    """Does *body* call ``globals()``, ``vars()`` or ``locals()``, the builtins?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("globals", "vars", "locals")
        and node.func.id not in g
        for node in ast.walk(body)
    )


def _bindings(body: ast.AST) -> Iterator[tuple[list[str], ast.expr]]:
    """``(names, value)`` for each binding *body* makes of local names from
    a value: an assignment (each name of an unpacked target), a walrus, a
    loop's or a comprehension's target from what it iterates, a ``with``'s
    ``as`` from its context manager."""
    for node in ast.walk(body):
        if isinstance(node, ast.Assign):
            yield [n for t in node.targets for n in _names(t)], node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            yield _names(node.target), node.value
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            yield _names(node.target), node.iter
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            yield _names(node.optional_vars), node.context_expr


def _names(target: ast.AST) -> list[str]:
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


#: Calls whose result holds nothing of what they are handed.
_HOLDS_NOTHING_OF_IT = frozenset(
    {"len", "str", "int", "float", "bool", "repr", "hash", "id", "abs", "round", "format", "isinstance", "callable"}
)


def _globals_read(value: ast.expr, g: Mapping[str, Any], params: set[str], aliases: Mapping[str, set[str]]) -> set[str]:
    """The changeable globals, directly or through *aliases*, an object
    *value* gives may hold or be: every one it reads, but through a deep
    copy or a call in `_HOLDS_NOTHING_OF_IT`."""
    if isinstance(value, ast.Call):
        func = value.func
        callee = None
        if isinstance(func, ast.Name):
            if func.id in _HOLDS_NOTHING_OF_IT and func.id not in g:
                return set()
            callee = g.get(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            callee = attribute_or(g.get(func.value.id), func.attr, None)
        if callee is copy.deepcopy or any(callee is fresh for fresh in _FRESH_FROM_TEXT):
            return set()
    found: set[str] = set()
    for node in ast.walk(value):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id in aliases:
            found |= aliases[node.id]
        elif (
            node.id in g
            and node.id not in params
            and not _never_changed(g[node.id])
            and not instance_of(g[node.id], types.ModuleType)
        ):
            found.add(node.id)
    return found


#: Calls that build a new object from text: ``pickle.loads(pickle.dumps(CFG))``.
_FRESH_FROM_TEXT = (pickle.loads, json.loads)


def _never_changed(value: Any) -> bool:
    """Can nothing change *value*: an int, a str, a type, a function, a
    NumPy scalar, or a tuple of such?"""
    if type(value) in _NEVER_CHANGED or isinstance(value, (type, types.FunctionType, types.BuiltinFunctionType)):
        return True
    if type(value) in (tuple, frozenset):
        return len(value) <= 64 and all(_never_changed(item) for item in value)
    np = sys.modules.get("numpy")
    return np is not None and isinstance(value, np.generic) and type(value) is not np.void


def _changes_only_itself(body: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> bool:
    """Does *body* use its parameter *param* only at its top level: a
    method call on it as a statement of its own (``d.pop(k, None)``), an
    item set or deleted on it, a test of what it holds (``k in d``),
    returned? Nothing it holds is taken out, so nothing inside it changes."""
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(body):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    for node in ast.walk(body):
        if not (isinstance(node, ast.Name) and node.id == param and isinstance(node.ctx, ast.Load)):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Return):
            continue
        if isinstance(parent, ast.Compare) and node in parent.comparators:
            continue
        if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, (ast.Store, ast.Del)):
            continue
        if isinstance(parent, ast.Attribute) and parent.value is node:
            call = parents.get(id(parent))
            if isinstance(call, ast.Call) and call.func is parent and isinstance(parents.get(id(call)), ast.Expr):
                continue
        return False
    return True


def _reducers_of(t: type) -> list[Any]:
    """The functions of the user's a pickle of an instance of *t* runs: a
    reducer registered for it with ``copyreg``, else the reducing methods
    its classes define."""
    registered = copyreg.dispatch_table.get(t)
    if registered is not None:
        return [registered] if _users(registered) else []
    found = []
    for klass in t.__mro__:
        if klass is object:
            continue
        for name in _REDUCERS:
            method = _function(klass.__dict__.get(name))
            if method is not None and _users(method):
                found.append(method)
    return found


def _function(member: Any) -> Any:
    """The plain function behind a class member, or None."""
    if type(member) in (classmethod, staticmethod):
        member = member.__func__
    return member if type(member) is types.FunctionType else None


def _users(function: Any) -> bool:
    return users_function(function)


def _body(function: Any) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """*function*'s definition, parsed; None when its source is not there.
    A lambda is read as the one ``lambda`` on its lines with its parameters,
    made a function that returns its expression."""
    try:
        lines, _first = getsourcelines(function)
    except (OSError, TypeError, IndentationError, SyntaxError):
        return None
    source = textwrap.dedent("".join(lines))
    if attribute_or(function, "__name__", None) != "<lambda>":
        return parse_function_source(source)
    code = attribute_or(function, "__code__", None)
    if code is None:
        return None
    wanted = list(code.co_varnames[: code.co_argcount + code.co_kwonlyargcount])
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Lambda)
        and [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)] == wanted
    ]
    if len(found) != 1:
        return None
    body = ast.FunctionDef(name="_lambda", args=found[0].args, body=[ast.Return(found[0].body)], decorator_list=[])
    return ast.fix_missing_locations(ast.copy_location(body, found[0]))


def _writes_itself(body: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Does *body* change its first argument (the object being pickled, or
    the class) in place? Which variables hold that object is not known."""
    params = [*body.args.posonlyargs, *body.args.args]
    return bool(params) and params[0].arg in params_mutated_in_function(body)


def _rebinds(function: Any) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return True
    return bool(source_global_rebinds(source))


def _users_callees(function: Any, body: ast.AST, t: type) -> list[tuple[Any, ast.Call]]:
    """The functions of the user's *body* calls by name, or as a method
    of its first argument (``self.parts()``), found in *function*'s globals
    and on *t*, each with the call."""
    found = []
    g = attribute_or(function, "__globals__", {})
    for node in ast.walk(body):
        if not isinstance(node, ast.Call):
            continue
        callee = None
        if isinstance(node.func, ast.Name):
            callee = g.get(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in ("self", "cls"):
                callee = inspect.getattr_static(t, node.func.attr, None)
            else:
                holder = g.get(node.func.value.id)
                if instance_of(holder, type):
                    callee = inspect.getattr_static(holder, node.func.attr, None)
        if instance_of(callee, type):
            callee = _function(callee.__dict__.get("__init__"))
        callee = _function(callee)
        if callee is not None and _users(callee):
            found.append((callee, node))
    return found
