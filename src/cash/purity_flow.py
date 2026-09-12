"""Two data-flow questions the purity analyzer asks of one function body.

Both exist to stop advisories that were wrong on ordinary code (round 17,
four of five testers), without going quiet on the code they are for:

* **Is this receiver fresh where it is mutated?** ``df["x"] = ...`` is harmless
  when ``df`` is an object the function made itself, and a real side effect
  when it is the caller's. The flow-insensitive rule ("every binding of the
  name is a fresh allocation") missed three common shapes: tuple unpacking
  (``a, b = [], []``), a view of a local array (``inner = u[1:-1, 1:-1]``),
  and a pipeline that rebinds a helper's result to a copy
  (``df = load(p); df = df[mask]; df = df.merge(...)``). This pass walks the
  body in order, so the answer is per point: after ``df = df.merge(...)`` the
  name is fresh, right after ``df = load(p)`` it is not.

* **Does this ambient read reach anything but a log line?**
  ``t = time.perf_counter()`` feeding only ``print(f"{... - t:.2f}s")`` cannot
  freeze into a result, so KEY-AMBIENT-READ on it was noise -- and noise that
  suppressed the network warning a tester needed. Anything that reaches a
  ``return``, a stored container, a condition or a non-logging call still warns.

Both are conservative: whatever they do not understand counts as not fresh and
not log-only, which keeps the warning.
"""
from __future__ import annotations

import ast

__all__ = ["fresh_name_nodes", "receiver_is_fresh", "LogOnlyFlow", "is_log_helper",
           "is_log_line", "is_read_only_sql"]

_FRESH_CONSTRUCTOR_NAMES = frozenset({
    "list", "dict", "set", "bytearray", "defaultdict", "OrderedDict", "Counter", "deque",
})
_FRESH_CONSTRUCTOR_ATTRS = frozenset({
    "zeros", "empty", "ones", "full", "array", "asarray",
    "zeros_like", "empty_like", "ones_like", "full_like", "arange", "linspace",
    "DataFrame", "Series", "copy", "deepcopy", "fromkeys",
})
_FRESH_LITERAL_NODES = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)

#: Readers that hand back an object nobody else holds.
_FRESH_READER_ATTRS = frozenset({
    "read_csv", "read_parquet", "read_excel", "read_json", "read_feather",
    "read_table", "read_pickle", "read_sql", "read_sql_query", "read_fwf",
    "read_hdf", "read_orc", "loadtxt", "genfromtxt", "fromfile", "concat",
})

#: pandas / numpy methods that return a NEW object whatever they are called on
#: (without ``inplace=True``). Views -- ``reshape``, ``ravel``, ``view``,
#: ``.T`` -- are deliberately absent: mutating a view of the caller's data
#: mutates the caller's data. So is ``fit``: sklearn returns ``self``.
_NEW_OBJECT_METHODS = frozenset({
    "merge", "join", "dropna", "fillna", "assign", "rename", "drop",
    "reset_index", "set_index", "sort_values", "sort_index", "agg",
    "aggregate", "pivot", "pivot_table", "melt", "astype", "query",
    "head", "tail", "sample", "reindex", "round", "cumsum", "diff",
    "shift", "to_frame", "explode", "where", "mask", "clip", "replace",
    "drop_duplicates", "nlargest", "nsmallest", "value_counts", "describe",
    "copy", "deepcopy", "tolist", "to_numpy",
})

#: Methods on a fresh container that hand back one of its ELEMENTS, which may
#: be anyone's object -- unless the container is DEEP-fresh (below).
_ELEMENT_METHODS = frozenset({"get", "pop", "popitem", "setdefault", "item", "__getitem__"})

#: Prefix, in the flow's set of fresh names, for a container whose ELEMENTS are
#: fresh too: built empty or from fresh values, never aliased or handed to a
#: call that could fill it, and only ever given fresh or immutable values. A
#: per-key accumulator -- ``by_user[k].append(x)``, ``acc = out.get(k); acc[0]
#: += 1``, ``for u, stamps in by_user.items(): stamps.sort()`` -- mutates
#: elements of such a container, and was reported as a side effect in every
#: parser three round-20 testers wrote. One level only: an element of an
#: element may still be anyone's.
_DEEP = "\0deep:"

#: Methods that put their arguments INTO the container they are called on,
#: and which argument: -1 = every positional argument.
_INSERTING_METHODS = {"append": -1, "add": -1, "appendleft": -1, "insert": 1, "setdefault": 1}
#: ...or every element of their argument.
_MERGING_METHODS = frozenset({"extend", "update", "extendleft"})
#: Methods that neither insert nor let anything else do so.
_NON_INSERTING_METHODS = frozenset({
    "get", "items", "keys", "values", "pop", "popitem", "copy", "sort", "clear",
    "remove", "index", "count", "most_common", "reverse", "discard", "elements",
    "total", "popleft",
})

#: Calls a container can be handed without anything being put into it.
_READ_ONLY_CALLS = frozenset({
    "len", "sorted", "sum", "min", "max", "any", "all", "print", "repr", "str",
    "enumerate", "zip", "reversed", "iter", "isinstance", "bool", "list", "tuple",
    "set", "frozenset", "dict", "next", "map", "filter", "hash", "id", "type",
    "format", "round", "abs",
})

#: Constructors of an EMPTY container, and the defaultdict factories whose
#: values are fresh or immutable.
_EMPTY_CONTAINERS = frozenset({"dict", "list", "set", "OrderedDict", "Counter", "deque"})
_FRESH_FACTORIES = frozenset({
    "list", "dict", "set", "int", "float", "str", "bool", "tuple", "frozenset",
    "Counter", "OrderedDict", "deque",
})


def root_name(node: ast.AST) -> ast.Name | None:
    """The Name at the bottom of an attribute / subscript / call chain."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node if isinstance(node, ast.Name) else None


def _is_mask(slc: ast.AST) -> bool:
    if isinstance(slc, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(slc, ast.UnaryOp) and isinstance(slc.op, (ast.Not, ast.Invert)):
        return True
    return isinstance(slc, ast.BinOp) and isinstance(slc.op, (ast.BitAnd, ast.BitOr, ast.BitXor))


def _has_slice(slc: ast.AST) -> bool:
    if isinstance(slc, ast.Slice):
        return True
    return isinstance(slc, ast.Tuple) and any(isinstance(e, ast.Slice) for e in slc.elts)


def _fresh(node: ast.AST | None, name_is_fresh, name_is_deep=None) -> bool:
    """Does *node* evaluate to an object no caller-visible name can reach?

    *name_is_fresh* answers the question for a bare name; everything else is
    decided by the shape of the expression. *name_is_deep*, when given, says
    whether a name holds a deep-fresh container, whose elements are fresh.
    """
    def element_of_deep(container: ast.AST) -> bool:
        return (name_is_deep is not None and isinstance(container, ast.Name)
                and name_is_deep(container))

    if node is None:
        return False
    if isinstance(node, (*_FRESH_LITERAL_NODES, ast.Tuple)):
        return True
    if isinstance(node, ast.Name):
        return name_is_fresh(node)
    if isinstance(node, ast.BinOp):
        return True                     # a + b builds a new object
    if isinstance(node, ast.IfExp):
        return _fresh(node.body, name_is_fresh) and _fresh(node.orelse, name_is_fresh)
    if isinstance(node, ast.BoolOp):
        return all(_fresh(v, name_is_fresh) for v in node.values)

    def rooted(n: ast.AST) -> bool:
        name = root_name(n)
        return name is not None and name_is_fresh(name)

    if isinstance(node, ast.Subscript):
        if _is_mask(node.slice):
            return True                 # a boolean filter copies
        if _has_slice(node.slice):
            return rooted(node.value)   # a view: only of our own data
        return element_of_deep(node.value)  # an element: anyone's, unless deep
    if isinstance(node, ast.Attribute):
        return rooted(node)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id in _FRESH_CONSTRUCTOR_NAMES
        if isinstance(f, ast.Attribute):
            if any(kw.arg == "inplace" for kw in node.keywords):
                return False
            if (f.attr in _FRESH_CONSTRUCTOR_ATTRS or f.attr in _FRESH_READER_ATTRS
                    or f.attr in _NEW_OBJECT_METHODS):
                return True
            if f.attr in _ELEMENT_METHODS:
                return element_of_deep(f.value)
            return rooted(f.value)
    return False


def expr_is_fresh(node: ast.AST | None, fresh: set[str]) -> bool:
    """`_fresh` against a set of names known to be fresh here."""
    return _fresh(node, lambda n: n.id in fresh, lambda n: _DEEP + n.id in fresh)


def _fresh_or_immutable(node: ast.AST | None, fresh: set[str]) -> bool:
    """May *node*'s value go into a deep-fresh container? A fresh object, or
    one nobody can mutate."""
    if isinstance(node, (ast.Constant, ast.JoinedStr)):
        return True
    if isinstance(node, ast.Tuple):
        return all(_fresh_or_immutable(e, fresh) for e in node.elts)
    return expr_is_fresh(node, fresh)


def receiver_is_fresh(node: ast.AST, fresh_nodes: frozenset[int]) -> bool:
    """`_fresh` for a mutation's receiver, using `fresh_name_nodes` marks --
    which also hold the element expressions of deep-fresh containers."""
    if id(node) in fresh_nodes:
        return True
    return _fresh(node, lambda n: id(n) in fresh_nodes)


def _callee_name(func: ast.AST) -> str | None:
    """``dict`` for ``dict(...)``, ``defaultdict`` for ``collections.defaultdict(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id == "collections"):
        return func.attr
    return None


def _deep_value(value: ast.AST | None, fresh: set[str]) -> bool:
    """Does *value* build a container whose elements are all fresh or immutable?"""
    if isinstance(value, (ast.List, ast.Set, ast.Tuple)):
        return not any(isinstance(e, ast.Starred) for e in value.elts) and all(
            _fresh_or_immutable(e, fresh) for e in value.elts)
    if isinstance(value, ast.Dict):
        return None not in value.keys and all(
            _fresh_or_immutable(v, fresh) for v in value.values)
    if isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp)):
        inner = set(fresh)
        for gen in value.generators:
            names: set[str] = set()
            _bound_names(gen.target, names)
            inner = _without(inner, names)
        return _fresh_or_immutable(
            value.value if isinstance(value, ast.DictComp) else value.elt, inner)
    if isinstance(value, ast.Call) and not value.keywords:
        name = _callee_name(value.func)
        if name in _EMPTY_CONTAINERS and not value.args:
            return True
        if name == "defaultdict" and len(value.args) <= 1:
            if not value.args:
                return True
            factory = value.args[0]
            if isinstance(factory, ast.Name):
                return factory.id in _FRESH_FACTORIES
            if isinstance(factory, ast.Lambda) and not factory.args.args:
                return _fresh_or_immutable(factory.body, set())
    return False


def _without(fresh: set[str], names: set[str]) -> set[str]:
    """*fresh* with *names* unbound -- their deep marks too."""
    if not names:
        return set(fresh)
    deep = {_DEEP + n for n in names}
    return {f for f in fresh if f not in names and f not in deep}


def _is_csv_rows(node: ast.AST) -> bool:
    """``csv.reader(f)`` / ``csv.DictReader(f)``: each row is a new list or
    dict, handed to nobody else."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute):
        return (f.attr in ("reader", "DictReader") and isinstance(f.value, ast.Name)
                and f.value.id == "csv")
    return isinstance(f, ast.Name) and f.id == "DictReader"


def _read_only_use(node: ast.Name, up: ast.AST | None) -> bool:
    """Is this load of a name one that cannot put anything INTO its value?"""
    if isinstance(up, (ast.Subscript, ast.Attribute)):
        return up.value is node
    if isinstance(up, (ast.For, ast.AsyncFor, ast.comprehension)):
        return up.iter is node
    if isinstance(up, ast.Call):
        return (node in up.args and isinstance(up.func, ast.Name)
                and up.func.id in _READ_ONLY_CALLS)
    if isinstance(up, ast.IfExp):
        return up.test is node
    return isinstance(up, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
                           ast.FormattedValue, ast.Return, ast.Expr, ast.If,
                           ast.While, ast.Assert))


def _escaped_names(func_def: ast.AST) -> frozenset[str]:
    """Names whose value may be aliased or filled by other code: loaded
    anywhere but a read-only use, or at all inside a nested scope."""
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(func_def):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    escaped: set[str] = set()
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    for node in ast.walk(func_def):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        up = parent.get(id(node))
        anc = up
        while anc is not None and anc is not func_def and not isinstance(anc, scopes):
            anc = parent.get(id(anc))
        if (anc is not None and anc is not func_def) or not _read_only_use(node, up):
            escaped.add(node.id)
    return frozenset(escaped)


def _bound_names(target: ast.AST, out: set[str]) -> None:
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            _bound_names(e, out)
    elif isinstance(target, ast.Starred):
        _bound_names(target.value, out)


def _walrus_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.NamedExpr):
            _bound_names(sub.target, out)
    return out


class _FreshFlow:
    """Forward walk: which names hold a fresh object, at each point.

    The state is a set of fresh names, plus ``_DEEP + name`` for each one that
    holds a deep-fresh container.
    """

    def __init__(self, excluded: frozenset[str], escaped: frozenset[str] = frozenset()):
        self.excluded = excluded
        #: names that may be aliased or filled elsewhere: never deep
        self.escaped = escaped
        #: id(node) -> was it fresh where evaluated: every Name, and each
        #: element expression of a deep-fresh container
        self.marks: dict[int, bool] = {}

    # -- expressions ------------------------------------------------------

    @staticmethod
    def _deep(node: ast.AST, fresh: set[str]) -> bool:
        return isinstance(node, ast.Name) and _DEEP + node.id in fresh

    def _check_insertion(self, call: ast.Call, fresh: set[str]) -> None:
        """A call on a deep container that puts something into it: keep it
        deep only if what goes in is fresh or immutable. Updates *fresh*."""
        f = call.func
        if not (isinstance(f, ast.Attribute) and self._deep(f.value, fresh)):
            return
        key = _DEEP + f.value.id
        if f.attr in _INSERTING_METHODS:
            which = _INSERTING_METHODS[f.attr]
            args = call.args if which < 0 else call.args[which:which + 1]
            ok = all(_fresh_or_immutable(a, fresh) for a in args)
        elif f.attr in _MERGING_METHODS:
            ok = all(_deep_value(a, fresh) or self._deep(a, fresh) for a in call.args) and all(
                _fresh_or_immutable(kw.value, fresh) for kw in call.keywords)
        else:
            ok = f.attr in _NON_INSERTING_METHODS
        if not ok:
            fresh.discard(key)

    def _mark(self, node: ast.AST | None, fresh: set[str]) -> None:
        if node is None:
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = set(fresh)
            for gen in node.generators:
                self._mark(gen.iter, inner)
                names: set[str] = set()
                _bound_names(gen.target, names)
                inner = _without(inner, names)
                for cond in gen.ifs:
                    self._mark(cond, inner)
            for part in ("elt", "key", "value"):
                self._mark(getattr(node, part, None), inner)
            return
        if isinstance(node, ast.Lambda):
            args = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
            self._mark(node.body, _without(fresh, args))
            return
        if isinstance(node, ast.Name):
            self.marks[id(node)] = node.id in fresh
        elif isinstance(node, ast.Subscript):
            if self._deep(node.value, fresh):
                self.marks[id(node)] = True     # an element of a deep container
        elif isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr in _ELEMENT_METHODS
                    and self._deep(f.value, fresh)):
                self.marks[id(node)] = True
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.stmt):
                self._mark(child, fresh)
        if isinstance(node, ast.Call):
            self._check_insertion(node, fresh)

    # -- statements -------------------------------------------------------

    def block(self, stmts: list[ast.stmt], fresh: set[str]) -> set[str]:
        fresh = set(fresh)
        for s in stmts:
            fresh = self.stmt(s, fresh)
        return fresh

    def _assign(self, targets: list[ast.AST], value: ast.AST, fresh: set[str]) -> set[str]:
        out = set(fresh)
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                if expr_is_fresh(value, fresh) and tgt.id not in self.excluded:
                    out.add(tgt.id)
                else:
                    out.discard(tgt.id)
                if tgt.id not in self.excluded and tgt.id not in self.escaped \
                        and _deep_value(value, fresh):
                    out.add(_DEEP + tgt.id)
                else:
                    out.discard(_DEEP + tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                # `a, b = [], []` pairs up; anything else unpacks elements,
                # which may be anyone's objects.
                pairs = (isinstance(value, (ast.Tuple, ast.List))
                         and len(value.elts) == len(tgt.elts)
                         and not any(isinstance(e, ast.Starred) for e in tgt.elts))
                for i, elt in enumerate(tgt.elts):
                    names: set[str] = set()
                    _bound_names(elt, names)
                    out = _without(out, names)
                    if (pairs and isinstance(elt, ast.Name) and elt.id not in self.excluded
                            and expr_is_fresh(value.elts[i], fresh)):
                        out.add(elt.id)
            elif isinstance(tgt, ast.Subscript) and self._deep(tgt.value, fresh):
                # `d[k] = v` puts v into d.
                if not _fresh_or_immutable(value, fresh):
                    out.discard(_DEEP + tgt.value.id)
            # An Attribute / Subscript target mutates; it does not rebind.
        return out

    def _loop_targets(self, target: ast.AST, it: ast.AST, fresh: set[str]) -> set[str]:
        """Loop variables that hold a fresh object each time round: the
        elements of a deep container, and the rows a csv reader makes."""
        # `sorted(d.items())`, `list(d)`: the same elements, reordered or copied.
        while (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
               and it.func.id in ("sorted", "list", "tuple", "reversed")
               and len(it.args) == 1):
            it = it.args[0]
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "enumerate" and it.args
                and isinstance(target, ast.Tuple) and len(target.elts) == 2):
            it, target = it.args[0], target.elts[1]
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and not it.args:
            if it.func.attr == "values" and self._deep(it.func.value, fresh):
                return {target.id} if isinstance(target, ast.Name) else set()
            if (it.func.attr == "items" and self._deep(it.func.value, fresh)
                    and isinstance(target, ast.Tuple) and len(target.elts) == 2
                    and isinstance(target.elts[1], ast.Name)):
                return {target.elts[1].id}
        if (self._deep(it, fresh) or _is_csv_rows(it)) and isinstance(target, ast.Name):
            return {target.id}
        return set()

    def _loop(self, header: list[ast.AST], names: set[str], body: list[ast.stmt],
              orelse: list[ast.stmt], fresh: set[str], bound=None) -> set[str]:
        head = _without(fresh, names)
        while True:                     # to the fixpoint: an iteration may undo freshness
            state = head | (bound(head) if bound is not None else set())
            for h in header:
                self._mark(h, state)
            end = self.block(body, state)
            nxt = head & _without(end, names)
            if nxt == head:
                break
            head = nxt
        return head & self.block(orelse, head)

    def stmt(self, s: ast.stmt, fresh: set[str]) -> set[str]:
        if isinstance(s, ast.Assign):
            self._mark(s.value, fresh)
            for t in s.targets:
                self._mark(t, fresh)
            return _without(self._assign(s.targets, s.value, fresh), _walrus_names(s.value))
        if isinstance(s, ast.AnnAssign):
            self._mark(s.value, fresh)
            self._mark(s.target, fresh)
            return set(fresh) if s.value is None else self._assign([s.target], s.value, fresh)
        if isinstance(s, ast.AugAssign):
            self._mark(s.value, fresh)
            self._mark(s.target, fresh)
            return set(fresh)           # in place: the identity is unchanged
        if isinstance(s, (ast.For, ast.AsyncFor)):
            self._mark(s.iter, fresh)
            names: set[str] = set()
            _bound_names(s.target, names)
            return self._loop([s.target], names, s.body, s.orelse, fresh,
                              bound=lambda state: self._loop_targets(s.target, s.iter, state))
        if isinstance(s, ast.While):
            return self._loop([s.test], set(), s.body, s.orelse, fresh)
        if isinstance(s, ast.If):
            self._mark(s.test, fresh)
            return self.block(s.body, fresh) & self.block(s.orelse, fresh)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            out = set(fresh)
            for item in s.items:
                self._mark(item.context_expr, fresh)
                if item.optional_vars is not None:
                    bound: set[str] = set()
                    _bound_names(item.optional_vars, bound)
                    out = _without(out, bound)
            return self.block(s.body, out)
        if isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
            body_end = self.block(s.body, fresh)
            start = set(fresh) & body_end   # a handler may run from any point
            ends = [self.block(s.orelse, body_end)]
            for h in s.handlers:
                self._mark(h.type, start)
                ends.append(self.block(h.body, _without(start, {h.name} if h.name else set())))
            return self.block(s.finalbody, set.intersection(*ends))
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in s.decorator_list:
                self._mark(d, fresh)
            return _without(fresh, {s.name})
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            return _without(fresh, {(a.asname or a.name).split(".")[0] for a in s.names})
        if isinstance(s, ast.Delete):
            for t in s.targets:
                self._mark(t, fresh)
            return _without(fresh, {t.id for t in s.targets if isinstance(t, ast.Name)})
        # Anything else (Expr, Return, Raise, Assert, Match, ...): mark every
        # expression with the current state; a construct binding names we do
        # not model loses them.
        for child in ast.iter_child_nodes(s):
            if not isinstance(child, ast.stmt):
                self._mark(child, fresh)
        bodies = [v for _, v in ast.iter_fields(s)
                  if isinstance(v, list) and v and isinstance(v[0], ast.stmt)]
        bodies += [case.body for case in getattr(s, "cases", [])]
        out = set(fresh)
        if bodies:
            out = set.intersection(*(self.block(b, fresh) for b in bodies))
            out = _without(out, {n.id for n in ast.walk(s)
                                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)})
        return _without(out, _walrus_names(s))


def fresh_name_nodes(func_def: ast.AST) -> frozenset[int]:
    """ids of the nodes in *func_def* that hold a fresh object where evaluated:
    Name nodes, and the element expressions of deep-fresh containers.

    Nested function bodies are not walked; their names are never marked.
    """
    excluded: set[str] = set()
    for node in ast.walk(func_def):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            excluded.update(node.names)
    flow = _FreshFlow(frozenset(excluded), _escaped_names(func_def))
    flow.block(getattr(func_def, "body", []), set())
    return frozenset(k for k, v in flow.marks.items() if v)


# -- ambient reads that only reach a log line --------------------------------

_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception",
                          "critical", "log"})

#: Expression nodes a value passes through on its way to wherever it ends up.
#: A dict / list / set literal too: ``logger.info(json.dumps({"ts": now}))``
#: is a structured log line, and the literal only carries the value to it.
_TRANSPARENT = (ast.BinOp, ast.UnaryOp, ast.JoinedStr, ast.FormattedValue,
                ast.Call, ast.Attribute, ast.Subscript, ast.Compare, ast.IfExp,
                ast.BoolOp, ast.Tuple, ast.keyword, ast.Starred,
                ast.Dict, ast.List, ast.Set)


def _is_log_sink(call: ast.Call, log_helpers: frozenset[str] = frozenset()) -> bool:
    """``print(...)``, ``logger.info(...)`` and kin, ``warnings.warn(...)``,
    ``sys.stderr.write(...)``, and a call to one of *log_helpers* -- the
    caller's own functions that do nothing but that (see `is_log_helper`)."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id == "print" or f.id in log_helpers
    if not isinstance(f, ast.Attribute):
        return False
    recv = f.value
    recv_name = (recv.attr if isinstance(recv, ast.Attribute)
                 else recv.id if isinstance(recv, ast.Name) else "")
    if f.attr in _LOG_METHODS and "log" in recv_name.lower():
        return True
    if f.attr == "write" and recv_name in ("stderr", "stdout"):
        return True
    return f.attr == "warn" and recv_name == "warnings"


def _is_stderr(node: ast.AST) -> bool:
    """``sys.stderr`` / ``sys.__stderr__``, or ``stderr`` imported from sys."""
    if isinstance(node, ast.Attribute):
        return (node.attr in ("stderr", "__stderr__") and isinstance(node.value, ast.Name)
                and node.value.id == "sys")
    return isinstance(node, ast.Name) and node.id == "stderr"


def is_log_line(call: ast.Call) -> bool:
    """A call whose only effect is a diagnostic line: ``print(..., file=
    sys.stderr)``, ``logger.info(...)`` and kin, ``logging.warning(...)``,
    ``sys.stderr.write(...)``.

    A cache hit skipping one is what caching means -- the work it reported on
    did not happen -- so it is not a side effect to warn about. Three of five
    round-20 testers got an IMPURE-SIDE-EFFECTS on every function that called
    their progress helper. A ``print`` to stdout is not in this set: stdout
    may be the program's output, which a hit would drop.
    """
    f = call.func
    if isinstance(f, ast.Name):
        return f.id == "print" and any(
            kw.arg == "file" and _is_stderr(kw.value) for kw in call.keywords)
    if not isinstance(f, ast.Attribute):
        return False
    recv = f.value
    recv_name = (recv.attr if isinstance(recv, ast.Attribute)
                 else recv.id if isinstance(recv, ast.Name) else "")
    if f.attr in _LOG_METHODS and "log" in recv_name.lower():
        return True
    return f.attr in ("write", "flush") and _is_stderr(recv)


_SQL_WRITES = ("INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE", "UPSERT", "CREATE",
               "DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA")


def is_read_only_sql(call: ast.Call) -> bool:
    """``con.execute("SELECT ...")``: a query that reads, written as a literal.

    ``execute`` is a write-shaped method -- through it most database writes
    happen -- but a literal SELECT (or a WITH that only selects) changes
    nothing, and round 20 was told a sqlite lookup was a "write method". A
    query built at run time, or any statement naming a write verb, is still
    reported.
    """
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "execute" and call.args):
        return False
    sql = call.args[0]
    if not (isinstance(sql, ast.Constant) and isinstance(sql.value, str)):
        return False
    text = " ".join(line.split("--", 1)[0] for line in sql.value.splitlines()).upper()
    words = set(text.replace("(", " ").replace(")", " ").replace(";", " ").split())
    head = text.lstrip()
    return head.startswith(("SELECT", "WITH")) and not words.intersection(_SQL_WRITES)


def is_log_helper(func_def: ast.AST) -> bool:
    """Is *func_def* a log helper: a body of nothing but log calls?

    ``def _log(msg): print(msg, file=sys.stderr)`` is how most analytics code
    logs, and a value passed to it reaches a log line and nothing else: it
    cannot be part of any result. A docstring and a bare ``return`` are
    allowed; anything else -- a returned value, an assignment, a branch --
    makes it an ordinary function.
    """
    body = list(getattr(func_def, "body", []))
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if not body:
        return False
    for stmt in body:
        if isinstance(stmt, ast.Return) and stmt.value is None:
            continue
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and _is_log_sink(stmt.value)):
            return False
    return True


class LogOnlyFlow:
    """Answers "does this value reach nothing but a print / log call?"."""

    def __init__(self, func_def: ast.AST, log_helpers: frozenset[str] = frozenset()):
        self._log_helpers = log_helpers
        self._parent: dict[int, ast.AST] = {}
        self._loads: dict[str, list[ast.Name]] = {}
        for node in ast.walk(func_def):
            for child in ast.iter_child_nodes(node):
                self._parent[id(child)] = node
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self._loads.setdefault(node.id, []).append(node)

    def only_logged(self, node: ast.AST, _names: frozenset[str] = frozenset()) -> bool:
        child = node
        while True:
            up = self._parent.get(id(child))
            if up is None:
                return False
            if (isinstance(up, ast.Call) and child is not up.func
                    and _is_log_sink(up, self._log_helpers)):
                return True
            if isinstance(up, _TRANSPARENT):
                child = up
                continue
            if isinstance(up, ast.Assign) and child is up.value:
                return all(self._name_only_logged(t, _names) for t in up.targets)
            if isinstance(up, (ast.AugAssign, ast.AnnAssign)) and child is up.value:
                return self._name_only_logged(up.target, _names)
            return False            # returned, stored, tested, passed on, ...

    def _name_only_logged(self, target: ast.AST, names: frozenset[str]) -> bool:
        if not isinstance(target, ast.Name):
            return False            # stored into something: it escapes
        if target.id in names:
            return True             # already being followed up the chain
        uses = self._loads.get(target.id, [])
        return all(self.only_logged(u, names | {target.id}) for u in uses)
