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

__all__ = ["fresh_name_nodes", "receiver_is_fresh", "LogOnlyFlow", "is_log_helper"]

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
#: be anyone's object.
_ELEMENT_METHODS = frozenset({"get", "pop", "popitem", "setdefault", "item", "__getitem__"})


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


def _fresh(node: ast.AST | None, name_is_fresh) -> bool:
    """Does *node* evaluate to an object no caller-visible name can reach?

    *name_is_fresh* answers the question for a bare name; everything else is
    decided by the shape of the expression.
    """
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
        return False                    # an element: could be anyone's
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
            return rooted(f.value) and f.attr not in _ELEMENT_METHODS
    return False


def expr_is_fresh(node: ast.AST | None, fresh: set[str]) -> bool:
    """`_fresh` against a set of names known to be fresh here."""
    return _fresh(node, lambda n: n.id in fresh)


def receiver_is_fresh(node: ast.AST, fresh_nodes: frozenset[int]) -> bool:
    """`_fresh` for a mutation's receiver, using `fresh_name_nodes` marks."""
    return _fresh(node, lambda n: id(n) in fresh_nodes)


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
    """Forward walk: which names hold a fresh object, at each point."""

    def __init__(self, excluded: frozenset[str]):
        self.excluded = excluded
        #: id(Name node) -> was that name fresh where the node is evaluated
        self.marks: dict[int, bool] = {}

    # -- expressions ------------------------------------------------------

    def _mark(self, node: ast.AST | None, fresh: set[str]) -> None:
        if node is None:
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = set(fresh)
            for gen in node.generators:
                self._mark(gen.iter, inner)
                names: set[str] = set()
                _bound_names(gen.target, names)
                inner -= names
                for cond in gen.ifs:
                    self._mark(cond, inner)
            for part in ("elt", "key", "value"):
                self._mark(getattr(node, part, None), inner)
            return
        if isinstance(node, ast.Lambda):
            args = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
            self._mark(node.body, set(fresh) - args)
            return
        if isinstance(node, ast.Name):
            self.marks[id(node)] = node.id in fresh
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.stmt):
                self._mark(child, fresh)

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
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                # `a, b = [], []` pairs up; anything else unpacks elements,
                # which may be anyone's objects.
                pairs = (isinstance(value, (ast.Tuple, ast.List))
                         and len(value.elts) == len(tgt.elts)
                         and not any(isinstance(e, ast.Starred) for e in tgt.elts))
                for i, elt in enumerate(tgt.elts):
                    names: set[str] = set()
                    _bound_names(elt, names)
                    if (pairs and isinstance(elt, ast.Name) and elt.id not in self.excluded
                            and expr_is_fresh(value.elts[i], fresh)):
                        out.add(elt.id)
                    else:
                        out -= names
            # An Attribute / Subscript target mutates; it does not rebind.
        return out

    def _loop(self, header: list[ast.AST], names: set[str], body: list[ast.stmt],
              orelse: list[ast.stmt], fresh: set[str]) -> set[str]:
        head = set(fresh) - names
        while True:                     # to the fixpoint: an iteration may undo freshness
            for h in header:
                self._mark(h, head)
            end = self.block(body, head)
            nxt = head & (end - names)
            if nxt == head:
                break
            head = nxt
        return head & self.block(orelse, head)

    def stmt(self, s: ast.stmt, fresh: set[str]) -> set[str]:
        if isinstance(s, ast.Assign):
            self._mark(s.value, fresh)
            for t in s.targets:
                self._mark(t, fresh)
            return self._assign(s.targets, s.value, fresh) - _walrus_names(s.value)
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
            return self._loop([s.target], names, s.body, s.orelse, fresh)
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
                    out -= bound
            return self.block(s.body, out)
        if isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
            body_end = self.block(s.body, fresh)
            start = set(fresh) & body_end   # a handler may run from any point
            ends = [self.block(s.orelse, body_end)]
            for h in s.handlers:
                self._mark(h.type, start)
                ends.append(self.block(h.body, start - ({h.name} if h.name else set())))
            return self.block(s.finalbody, set.intersection(*ends))
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in s.decorator_list:
                self._mark(d, fresh)
            return set(fresh) - {s.name}
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            return set(fresh) - {(a.asname or a.name).split(".")[0] for a in s.names}
        if isinstance(s, ast.Delete):
            for t in s.targets:
                self._mark(t, fresh)
            return set(fresh) - {t.id for t in s.targets if isinstance(t, ast.Name)}
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
            out -= {n.id for n in ast.walk(s)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        return out - _walrus_names(s)


def fresh_name_nodes(func_def: ast.AST) -> frozenset[int]:
    """ids of the Name nodes in *func_def* that hold a fresh object where read.

    Nested function bodies are not walked; their names are never marked.
    """
    excluded: set[str] = set()
    for node in ast.walk(func_def):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            excluded.update(node.names)
    flow = _FreshFlow(frozenset(excluded))
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
