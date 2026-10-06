"""Which module data a function's code reads, from its bytecode and source:
the globals it loads, its ``module.ATTR`` chains, the names imports in its
body bind, the names its decorators read, whether it reads a docstring.
Analysis only, memoized per code object; the folds hash what it names."""

from __future__ import annotations

import ast
import dis
import sys
import textwrap
import types
from collections.abc import Callable
from typing import Any

from .._memo import CODE_OBJECTS, LruMemo
from ..analysis.helper_bindings import local_import_map
from ..analysis.helper_code import own_code_is_user
from ..analysis.purity_policy import REPORTED_METHODS
from ..exceptions import SOURCE_RETRIEVAL_ERRORS
from ..source_reading import getsource, own_source
from .closure_fold import iter_code_scopes, unsafe_uses_of, waived_use_filter

#: Dunder globals that are machine or import machinery, never user data.
#:
#: These are skipped: ``__file__`` and ``__name__`` differ per checkout and
#: per invocation, so folding them would make a cache key un-shareable
#: between two machines and between ``python job.py`` and ``python -m
#: job``. Every other dunder is folded like any global, because a library
#: declares its data that way too: a bumped ``__version__`` must invalidate
#: what a report stamped with it.
MACHINERY_DUNDERS = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__path__",
        "__cached__",
        "__debug__",
        "__annotations__",
        "__dict__",
        "__module__",
        "__qualname__",
    }
)


#: Names whose load means the code reads a docstring at run time.
DOCSTRING_READS = frozenset({"__doc__", "getdoc", "cleandoc"})


#: A plain operand load: the key between a global's load and a subscript store.
#: 3.12 adds LOAD_FAST_CHECK (a local that may be unbound); 3.14 loads most
#: locals with LOAD_FAST_BORROW and small int constants with LOAD_SMALL_INT, so
#: without them `G[k] = v` and `del G[0]` read as plain reads there.
_OPERAND_LOADS = frozenset(
    {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_CONST", "LOAD_SMALL_INT", "LOAD_DEREF", "LOAD_NAME"}
)


def bytecode_written_attrs(code: types.CodeType) -> set[str]:
    """Attribute names *code* stores, deletes or mutates in place:
    ``C.count += 1``, ``self.n = 0``, ``cls.registry[k] = v``,
    ``type(self).seen.append(x)``. Exact shapes, as for
    `bytecode_mutated_globals`."""
    found: set[str] = set()
    instrs = list(dis.get_instructions(code))
    for i, ins in enumerate(instrs):
        if ins.opname in ("STORE_ATTR", "DELETE_ATTR"):
            found.add(ins.argval)
            continue
        if ins.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
            continue
        after = instrs[i + 1 : i + 3]
        if not after:
            continue
        first = after[0]
        if first.opname in ("LOAD_ATTR", "LOAD_METHOD") and first.argval in REPORTED_METHODS:
            found.add(ins.argval)
        elif (
            len(after) == 2 and first.opname in _OPERAND_LOADS and after[1].opname in ("STORE_SUBSCR", "DELETE_SUBSCR")
        ):
            found.add(ins.argval)
    return {n for n in found if isinstance(n, str)}


def bytecode_mutated_globals(scopes: tuple, names: set[str]) -> set[str]:
    """The *names* compiled code plainly writes into, for code without source.

    A load of the global followed by a writing method (`REPORTED_METHODS`:
    ``append``, ``update``, ...) or an attribute store (``obj.x = v``), or
    by one operand and a subscript store (``table[k] = v``). Exact shapes
    only: a global read as the VALUE stored (``d[k] = G``) must stay
    folded. What this misses is caught at run time by the provisional
    watch, one miss later.
    """
    found: set[str] = set()
    for scope in scopes:
        instrs = list(dis.get_instructions(scope))
        for i, ins in enumerate(instrs):
            if ins.opname != "LOAD_GLOBAL" or ins.argval not in names:
                continue
            after = instrs[i + 1 : i + 3]
            if not after:
                continue
            first = after[0]
            if (first.opname in ("LOAD_ATTR", "LOAD_METHOD") and first.argval in REPORTED_METHODS) or first.opname in (
                "STORE_ATTR",
                "DELETE_ATTR",
            ):
                found.add(ins.argval)
            elif (
                len(after) == 2
                and first.opname in _OPERAND_LOADS
                and after[1].opname in ("STORE_SUBSCR", "DELETE_SUBSCR")
            ):
                found.add(ins.argval)
    return found


#: A miss in `GlobalReads._local_binding_cache`, whose entries may be None.
_NO_PLAN = object()


def _is_cash_decorator(deco: ast.expr, module_globals: dict[str, Any]) -> bool:
    """Whether the decorator expression *deco* is cash's own ``@app.cache``.

    Its names configure the caching, not the result, and the ``Cash``
    instance they reach cannot be hashed: ``@app.cache`` over a
    ``functools.wraps`` decorator warned that the function read the
    unhashable global ``app``.
    """
    from ..core import Cash  # deferred: core builds this module

    root = deco
    while isinstance(root, (ast.Call, ast.Attribute)):
        root = root.func if isinstance(root, ast.Call) else root.value
    if not isinstance(root, ast.Name):
        return False
    value = module_globals.get(root.id)
    return isinstance(value, Cash) or value is sys.modules.get("cash")


#: The opcodes that read a module global by name (``LOAD_NAME`` in a class
#: body or at module level; ``LOAD_FROM_DICT_OR_GLOBALS`` in 3.12+ class bodies).
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})

#: Names through which code reads a module namespace by a string:
#: ``globals()[name]``, ``vars(mod)[name]``, ``eval(name)``,
#: ``sys.modules[__name__]``, ``mod.__dict__``, a frame's ``f_globals``.
NAMESPACE_BY_NAME = frozenset(
    {"globals", "vars", "eval", "exec", "__dict__", "modules", "f_globals", "import_module", "__import__"}
)


def reaches_namespace_by_name(scopes: tuple, module_globals: dict[str, Any]) -> bool:
    """Can a string in *scopes* name a global of *module_globals*?

    True when the code itself names a `NAMESPACE_BY_NAME` accessor, or a
    function of the same module it loads does (followed transitively, its
    own methods included for a class): ``get("K")`` with ``def get(name):
    return globals()[name]``. Without one, a string is just text.
    """
    seen: set[int] = set()
    stack = list(scopes)
    while stack:
        scope = stack.pop()
        if id(scope) in seen:
            continue
        seen.add(id(scope))
        if NAMESPACE_BY_NAME.intersection(scope.co_names or ()):
            return True
        for instr in dis.get_instructions(scope):
            if instr.opname not in _GLOBAL_LOADS:
                continue
            value = module_globals.get(instr.argval)
            members = vars(value).values() if isinstance(value, type) else (value,)
            for member in members:
                member = getattr(member, "__func__", member)
                member = getattr(member, "__wrapped__", member)  # a cached helper
                if isinstance(member, types.FunctionType) and member.__globals__ is module_globals:
                    stack.extend(iter_code_scopes(member.__code__))
    return False
#: The opcodes that read an attribute (``LOAD_METHOD`` before 3.12).
_ATTR_OPS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})


class GlobalReads:
    """The module data each function's code reads, worked out once per code
    object: the names alone, never their values."""

    def __init__(self) -> None:
        # code object -> global names its decorator expressions read
        self._decorator_names_cache: LruMemo[Any, tuple[str, ...]] = LruMemo(CODE_OBJECTS)
        # code object -> (tuple of global names it reads, the names among them
        # folded only provisionally). One entry, so the two never disagree.
        # See `read_global_data_names`; a missing entry means "unknown", which
        # `GlobalsFold.fold_read_globals` treats as "watch everything".
        self._global_read_cache: LruMemo[Any, tuple[tuple[str, ...], frozenset]] = LruMemo(CODE_OBJECTS)
        # (module_global, attribute) read pairs per code object; see
        # `module_attr_pairs`.
        self._module_attr_cache: LruMemo[Any, tuple[tuple[str, str], ...]] = LruMemo(CODE_OBJECTS)
        self._local_binding_cache: LruMemo[Any, tuple | None] = LruMemo(CODE_OBJECTS)
        # code object -> whether it reads a docstring; see `reads_docstrings`.
        self._docstring_reads: LruMemo[Any, bool] = LruMemo(CODE_OBJECTS)
        # code object -> whether folding its globals can find anything; see
        # `may_read_data`.
        self._reads_anything: LruMemo[Any, bool] = LruMemo(CODE_OBJECTS)

    def provisional_names(self, code: Any) -> frozenset | None:
        """The names `read_global_data_names` folds only provisionally for
        *code*, or None when it has not been worked out ("unknown", which
        the fold treats as "watch every folded name")."""
        cached = self._global_read_cache.get(code)
        return cached[1] if cached is not None else None

    def known_module_attr_pairs(self, code: Any) -> tuple[tuple[str, str], ...]:
        """`module_attr_pairs` as already worked out for *code*; ``()`` when
        it has not been."""
        return self._module_attr_cache.get(code) or ()

    def read_global_data_names(self, func: Callable) -> tuple[str, ...]:
        """Global names *func* references that are candidates for data-folding.

        ``co_names`` intersected with the function's globals, minus the import
        machinery dunders (``MACHINERY_DUNDERS``) and minus any global the
        function WRITES (``STORE_GLOBAL`` /
        ``DELETE_GLOBAL``). A written global is a side-effect accumulator (a
        ``global counter; counter += 1``) whose value drifts every call - folding
        it would make every call miss (the lesson, applied to globals).
        Modules / callables / classes are filtered per-call at fold time (a
        name's bound value can change). Cached per code object.

        Both bytecode-derived channels walk the NESTED scopes too:
        a global read only inside a genexp/lambda otherwise never invalidated
        (silent stale results), and — the reason the two must move together —
        the ``STORE_GLOBAL`` of a walrus accumulator inside a genexp lives in
        the genexp's own code object, so collecting nested reads without
        collecting nested writes would fold a drifting counter and miss
        forever. The in-place-mutation exclusion below needs no such change:
        it is AST-based, and ``ast.walk`` over the function's source already
        descends into comprehension and lambda bodies.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._global_read_cache.get(code)
        if cached is not None:
            return cached[0]
        g = getattr(func, "__globals__", {}) or {}

        scopes = tuple(iter_code_scopes(code))
        written = {
            instr.argval
            for scope in scopes
            for instr in dis.get_instructions(scope)
            if instr.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
        }
        # Names LOADED as globals, not every name in ``co_names``: that also
        # holds attribute names, so `b.lock` read the module's unrelated `lock`
        # and warned KEY-UNHASHABLE-GLOBAL about a global never read.
        candidates = {
            instr.argval
            for scope in scopes
            for instr in dis.get_instructions(scope)
            if instr.opname in _GLOBAL_LOADS
            and instr.argval in g
            and instr.argval not in MACHINERY_DUNDERS
            and instr.argval not in written
        }
        # A name spelled as a string reads the same global: `globals()["K"]`
        # is a LOAD_CONST, so `co_names` never had it and editing K served the
        # old answer -- 20 where an uncached run gives 500. The code channel already resolves string
        # constants this way (`CodeRefs.targets`); this is its data twin.
        # Only where a string CAN reach the namespace: code that reads it by
        # name, its own or a same-module helper's. Otherwise a column label
        # `frame["x"]` or an f-string piece `f"t{i}"` folded the unrelated
        # global `x` or `t`, hashed in full on every hit and recomputing
        # whenever a loop moved it.
        if reaches_namespace_by_name(scopes, g):
            candidates |= {
                c
                for scope in scopes
                for c in (scope.co_consts or ())
                if isinstance(c, str)
                and c.isidentifier()
                and c in g
                and c not in MACHINERY_DUNDERS
                and c not in written
            }
        # Also exclude globals the body mutates IN PLACE (``g['k'] += 1``,
        # ``g.append(...)``) - a STORE_GLOBAL-free accumulator that would
        # otherwise drift every call and cause a permanent miss.
        #
        # `hard` is that set: mutations visible in this function's own source.
        # `provisional` is the weaker case - a name merely PASSED to a call. Those are folded (so a change
        # invalidates) and confirmed at runtime by
        # `PurityChecks.learn_mutating_captures`, which demotes any that the call is actually
        # observed to mutate.
        provisional: frozenset = frozenset()
        if candidates:
            try:
                tree = ast.parse(textwrap.dedent(getsource(func)))
                hard = unsafe_uses_of(
                    tree,
                    candidates,
                    bare_args=False,
                    mutating_methods_only=True,
                )
                suspected = unsafe_uses_of(tree, candidates) - hard
                provisional = unsafe_uses_of(tree, suspected, waived=waived_use_filter(func, tree))
                # Suspected only on waived lines (`LEDGER.record(r)  #
                # @cash:assume-safe`, or inside `with cash.assume_safe():`):
                # the audited effect moves it on every call, so keying on it
                # made every hit impossible and demoting it warned about the
                # very line that was audited.
                hard |= suspected - provisional
                candidates -= hard
            except SOURCE_RETRIEVAL_ERRORS:
                # No source (`python - <<EOF`, `python -c`, `exec`): the
                # bytecode stands in for the AST. What it plainly writes
                # (`calls.append(x)`, `table[k] = v`) stays out; the rest is
                # folded PROVISIONALLY -- watched after a miss, and dropped
                # once a call is seen to move it. Folding none of them
                # served a stale result whenever a global it reads changed.
                candidates -= bytecode_mutated_globals(scopes, candidates)
                provisional = frozenset(candidates)
        names = tuple(sorted(candidates))
        # A MISSING entry is not "nothing is provisional" --
        # `GlobalsFold.fold_read_globals` reads that as "watch every folded
        # name", which costs an extra hash per miss and is the safe direction.
        self._global_read_cache[code] = (names, provisional)
        return names

    def reads_docstrings(self, code: Any) -> bool:
        """Does *code* read a docstring at run time (``f.__doc__``,
        ``inspect.getdoc(tool)``, ``getattr(C, "__doc__")``)? Cached per code."""
        cached = self._docstring_reads.get(code)
        if cached is None:
            cached = any(
                DOCSTRING_READS & set(scope.co_names or ()) or "__doc__" in (scope.co_consts or ())
                for scope in iter_code_scopes(code)
            )
            self._docstring_reads[code] = cached
        return cached

    @staticmethod
    def function_defaults(func: Callable) -> list[types.FunctionType]:
        """The user functions among *func*'s parameter defaults."""
        found: list[types.FunctionType] = []
        for container in (
            getattr(func, "__defaults__", None) or (),
            (getattr(func, "__kwdefaults__", None) or {}).values(),
        ):
            for value in container:
                if (
                    isinstance(value, types.FunctionType)
                    and value is not func
                    and own_code_is_user(value, getattr(func, "__module__", None))
                ):
                    found.append(value)
        return found

    def may_read_data(self, fn: Any) -> bool:
        """Could `GlobalsFold.fold_read_globals` find anything in *fn*? False for
        a method that reads no global, no ``module.attr``, no import in its
        body, no docstring and has no function default -- most of a class's
        methods, and every one a dataclass generates -- so a class costs a
        lookup per such method instead of a fold."""
        code = fn.__code__
        cached = self._reads_anything.get(code)
        if cached is None:
            cached = bool(
                self.read_global_data_names(fn)
                or self.module_attr_pairs(fn)
                or self.reads_docstrings(code)
                or self.local_binding_plan(fn)
            )
            self._reads_anything[code] = cached
        return cached or bool(self.function_defaults(fn))

    def decorator_global_names(self, fn: Callable) -> tuple[str, ...]:
        """Names the decorator expressions on *fn*'s ``def`` read from its module.

        ``@np.vectorize(otypes=OT)`` is evaluated once, at import, from the
        module's ``OT`` -- a name the function's body never mentions, so
        changing it changed nothing the key could see. Only the function a
        decorator wraps has these lines (its source starts at the first
        decorator); the values are folded like any other read global, so
        modules and callables among them are skipped there. Cached per code.
        """
        code = getattr(fn, "__code__", None)
        if code is None:
            return ()
        cached = self._decorator_names_cache.get(code)
        if cached is not None:
            return cached
        names: tuple[str, ...] = ()
        try:
            tree = ast.parse(textwrap.dedent(getsource(code)))
            node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if node is not None and node.decorator_list:
                g = getattr(fn, "__globals__", None) or {}
                names = tuple(
                    dict.fromkeys(
                        n.id
                        for deco in node.decorator_list
                        if not _is_cash_decorator(deco, g)
                        for n in ast.walk(deco)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    )
                )
        except SOURCE_RETRIEVAL_ERRORS + (SyntaxError, ValueError):
            names = ()
        self._decorator_names_cache[code] = names
        return names

    def module_attr_pairs(self, func: Callable) -> tuple[tuple[str, str], ...]:
        """``(module_path, attribute)`` pairs the body reads, from bytecode.

        ``conf.RATE``, ``pkg.conf.RATE`` and ``m = pkg.conf; m.RATE`` are
        three spellings of one dependency, and the global the body names is
        a module in each: none of them reaches
        `GlobalReads.read_global_data_names`, so the attribute is keyed here.

        * A global followed by a chain of attribute loads
          (``LOAD_GLOBAL pkg; LOAD_ATTR conf; LOAD_ATTR RATE``) gives one
          pair per link: ``("pkg", "conf")`` and ``("pkg.conf", "RATE")``.
          `ModuleAttrFold.module_attr_parts` resolves the dotted path and folds
          the pairs whose path is a user module.
        * A module the code takes whole (bound to a local, passed on, or read
          with ``vars(conf)["K"]`` / ``getattr(conf, "K")``) is paired with
          every attribute name and identifier-shaped string constant in the
          code. A pair that names nothing is dropped at fold time; one that
          names an attribute the code does not read only adds a part.

        Nested scopes count: a read inside a genexp is a read of the body.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return ()
        cached = self._module_attr_cache.get(code)
        if cached is not None:
            return cached

        g = getattr(func, "__globals__", None) or {}
        pairs: set[tuple[str, str]] = set()
        whole: set[str] = set()
        names: set[str] = set()
        scopes = list(iter_code_scopes(code))
        for scope in scopes:
            names.update(c for c in scope.co_consts or () if isinstance(c, str) and c.isidentifier())
            instrs = [i for i in dis.get_instructions(scope) if i.opname != "EXTENDED_ARG"]
            for i, ins in enumerate(instrs):
                if ins.opname in _ATTR_OPS and isinstance(ins.argval, str):
                    names.add(ins.argval)
                if ins.opname != "LOAD_GLOBAL" or not isinstance(ins.argval, str):
                    continue
                path = ins.argval
                value = g.get(path)
                j = i + 1
                while j < len(instrs) and instrs[j].opname in _ATTR_OPS and isinstance(instrs[j].argval, str):
                    attr = instrs[j].argval
                    if attr.startswith("__"):
                        break
                    pairs.add((path, attr))
                    path = f"{path}.{attr}"
                    value = getattr(value, attr, None) if isinstance(value, types.ModuleType) else None
                    j += 1
                else:
                    if isinstance(value, types.ModuleType):
                        whole.add(path)
        for path in whole:
            pairs.update((path, n) for n in names if not n.startswith("__"))
        result = tuple(sorted(pairs))
        self._module_attr_cache[code] = result
        return result

    def local_binding_plan(self, func: Callable) -> tuple | None:
        """What `ModuleAttrFold.local_binding_parts` needs from *func*'s source, per code object.

        ``(imports, attr_reads, bare_reads)``: the names an import written in
        the body binds (``name -> (module, prefix)``), the ``name.ATTR`` reads
        of any local or closure name, and the bare reads of local names.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return None
        cached = self._local_binding_cache.get(code, _NO_PLAN)
        if cached is not _NO_PLAN:
            return cached
        plan = None
        try:
            tree = ast.parse(textwrap.dedent(own_source(func)))
            func_def = next(
                (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))), None
            )
            if func_def is not None:
                imports = local_import_map(func_def, func)
                watched = set(imports) | set(code.co_freevars or ())
                attr_reads: dict[str, set[str]] = {}
                bare_reads: set[str] = set()
                for node in ast.walk(func_def):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in watched
                        and not node.attr.startswith("__")
                    ):
                        attr_reads.setdefault(node.value.id, set()).add(node.attr)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imports:
                        bare_reads.add(node.id)
                if attr_reads or bare_reads:
                    plan = (imports, attr_reads, bare_reads)
        except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError, ValueError):
            plan = None
        self._local_binding_cache[code] = plan
        return plan
