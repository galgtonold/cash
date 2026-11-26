"""What Python itself says a snippet binds and reads, for checking cash's
scope model (`cash.analysis.scope_model`) against.

Two independent oracles:

* :func:`differences` compares the model's scopes with ``symtable``, scope by
  scope and name by name. ``symtable`` reports the same code differently from
  one Python to the next (comprehensions lose their table in 3.12, annotations
  gain one in 3.14), so both sides are first brought to one form; each step of
  that is commented where it is done.
* :func:`compiled_names` reads the compiled bytecode: the names the code
  stores into and loads from the module's namespace. It knows nothing of the
  model, so it also judges the analysis that ran before the model existed.

Also the corpus both are run over (:data:`SNIPPETS`), one entry per kind of
statement and expression that binds or reads a name.
"""

from __future__ import annotations

import ast
import dis
import symtable
import sys
import types

from cash.analysis import scope_model as sm

PY = sys.version_info[:2]

# ---------------------------------------------------------------------------
# The corpus: (first version it parses on, source)
# ---------------------------------------------------------------------------

_ANY = (3, 12)

SNIPPETS: list[tuple[tuple[int, int], str]] = [
    # --- plain bindings --------------------------------------------------
    (_ANY, "x = a"),
    (_ANY, "x = y = a + b"),
    (_ANY, "x, (y, *z) = a"),
    (_ANY, "[x, y] = a"),
    (_ANY, "x += a"),
    (_ANY, "x.attr = a"),
    (_ANY, "x[i] = a"),
    (_ANY, "x[i].attr += a"),
    (_ANY, "x: int = a"),
    (_ANY, "x: int"),
    (_ANY, "x.attr: int = a"),
    (_ANY, "x[k]: int = a"),
    (_ANY, "del x"),
    (_ANY, "del x[i], y.attr"),
    (_ANY, "a + b * c"),
    (_ANY, "a.b.c(d, e=f, *g, **h)"),
    (_ANY, "a[b:c, d]"),
    (_ANY, "a if b else c"),
    (_ANY, "a and b or not c"),
    (_ANY, "a < b <= c"),
    (_ANY, "{a: b, **c}"),
    (_ANY, "{a, *b}"),
    (_ANY, "(a, [b, c])"),
    (_ANY, "print(y := a)"),
    (_ANY, "x = (y := a) + y"),
    (_ANY, "x = f'{a} {b!r:>{w}} {c=}'"),
    (_ANY, "x = f'{f\"{a}\"}'"),
    (_ANY, "x = lambda: a"),
    # --- imports ---------------------------------------------------------
    (_ANY, "import os"),
    (_ANY, "import os.path"),
    (_ANY, "import os.path as p, sys as s"),
    (_ANY, "from os import path, sep as s"),
    (_ANY, "from os.path import *"),
    (_ANY, "from . import sibling"),
    # --- loops, branches, with, try ---------------------------------------
    (_ANY, "for i in xs:\n    total = total + i\nelse:\n    done = True"),
    (_ANY, "for i, (j, *k) in xs:\n    pass"),
    (_ANY, "for x.attr in xs:\n    pass"),
    (_ANY, "for x[0] in xs:\n    pass"),
    (_ANY, "while n:\n    n -= step\nelse:\n    z = 1"),
    (_ANY, "if a:\n    x = 1\nelif b:\n    y = 2\nelse:\n    z = 3"),
    (_ANY, "with open(p) as fh, lock:\n    data = fh.read()"),
    (_ANY, "with a as (b, c), d as e.f:\n    pass"),
    (
        _ANY,
        "try:\n    x = f()\nexcept (E1, E2) as e:\n    y = str(e)\nexcept E3:\n    pass\nelse:\n    z = 1\nfinally:\n    w = 2",
    ),
    ((3, 11), "try:\n    x = f()\nexcept* E as e:\n    y = e.exceptions"),
    (_ANY, "for i in xs:\n    if i:\n        break\n    continue"),
    (_ANY, "assert a, b"),
    (_ANY, "raise E(a) from b"),
    (
        _ANY,
        "async def co():\n    async with a as b:\n        async for c in d:\n            e = await c\n    return [x async for x in y]",
    ),
    # --- match -----------------------------------------------------------
    (_ANY, "match data:\n    case {'k': [a, *rest]}:\n        pass"),
    (_ANY, "match data:\n    case {'k': v, **others}:\n        pass"),
    (_ANY, "match p:\n    case Point(x=a, y=0):\n        pass\n    case Point(b, c):\n        pass"),
    (_ANY, "match xs:\n    case [first, *others]:\n        pass\n    case [] | None:\n        pass"),
    (_ANY, "match v:\n    case x if x > lim:\n        hit = x"),
    (_ANY, "match v:\n    case (1 | 2) as small:\n        pass\n    case str() as s:\n        pass"),
    (_ANY, "match v:\n    case Color.RED | mod.C.BLUE:\n        pass\n    case _:\n        pass"),
    (_ANY, "match v:\n    case [(a, b) as pair, *_]:\n        pass\n    case {**kw}:\n        pass"),
    (
        _ANY,
        "def m(v):\n    match v:\n        case [a, *rest] if g:\n            return a, rest\n        case {'k': b, **kw} as whole:\n            return b, kw, whole, outside",
    ),
    # --- comprehensions ----------------------------------------------------
    (_ANY, "[x for x in xs]"),
    (_ANY, "ys = [x * k for x in xs if x > lo]"),
    (_ANY, "{k: v for k, v in items if k}"),
    (_ANY, "{x for row in rows for x in row}"),
    (_ANY, "g = (x for x in xs)"),
    (_ANY, "[[y for y in x] for x in xs]"),
    (_ANY, "[x for x in x]"),
    (_ANY, "[(last := x) for x in xs]"),
    (_ANY, "any((hit := x) > lim for x in xs)"),
    (_ANY, "[lambda: x + free for x in xs]"),
    (_ANY, "[x for x in xs if (lambda q: q + z)(x)]"),
    (_ANY, "def comp(rows):\n    return [(cell, w := cell) for row in rows for cell in row if cell != skip]"),
    (_ANY, "class K:\n    xs = [1]\n    ys = [x for x in xs]\n    zs = [x + n for x in outer]"),
    # --- functions -------------------------------------------------------
    (_ANY, "def f(a, b=d1, *args, c=d2, **kw):\n    return a + b + c + g"),
    (_ANY, "def f(a, /, b, *, c):\n    loc = a\n    return loc, glob"),
    (_ANY, "def f():\n    global g\n    g = 1\n    return g"),
    (_ANY, "def f():\n    global g\n    g += 1"),
    (
        _ANY,
        "def outer():\n    n = 0\n    def inner():\n        nonlocal n\n        n += 1\n        return n + free\n    return inner",
    ),
    (_ANY, "def outer():\n    def inner():\n        return late\n    late = 1\n    return inner"),
    (_ANY, "def outer():\n    global g\n    def inner():\n        return g\n    return inner"),
    (_ANY, "def f():\n    import os\n    from sys import path as p\n    return os, p"),
    (
        _ANY,
        "def f():\n    try:\n        pass\n    except E as e:\n        return e\n    with a as b:\n        for c in b:\n            del c",
    ),
    (_ANY, "def f():\n    x: Hint = 1\n    y: Other\n    return x"),
    (_ANY, "def rec(n):\n    return rec(n - 1) if n else 0"),
    (_ANY, "@deco\n@reg.add(key)\ndef f():\n    pass"),
    (_ANY, "def f(x=(lambda: dflt)()):\n    return lambda y: x + y + z"),
    (_ANY, "f = lambda a, b=c, *d, e=g, **h: a + b + i"),
    (_ANY, "def gen():\n    x = yield a\n    yield from b"),
    (_ANY, "def f(x: A, *args: B, k: C = d, **kw: D) -> R:\n    return x"),
    (_ANY, "def f(x: 'Quoted', y: 'list[Inner]' = None) -> 'Ret':\n    return x"),
    (_ANY, "def f(x: Optional['Fwd']):\n    return x"),
    (_ANY, "from __future__ import annotations\ndef f(x: A) -> R:\n    v: B = 1\n    return x\ny: C = 2"),
    (_ANY, "def f():\n    del gone\n    return gone"),
    (_ANY, "def f():\n    x = [y := v for v in vs]\n    return y"),
    # --- classes ---------------------------------------------------------
    (_ANY, "class C(Base, metaclass=Meta, kw=val):\n    attr = default\n    def m(self):\n        return attr, self"),
    (_ANY, "@deco(arg)\nclass C:\n    x = 1\n    y = x + z\n    def m(self):\n        return x"),
    (_ANY, "class C:\n    global g\n    g = 1"),
    (_ANY, "class C:\n    hint: Ann\n    val: Ann2 = init"),
    (_ANY, "class C:\n    def m(self):\n        return __class__, super().m()"),
    (_ANY, "class Outer:\n    class Inner:\n        v = top\n    w = Inner"),
    (
        _ANY,
        "def mk():\n    loc = 1\n    class C:\n        a = loc\n        def m(self):\n            return loc\n    return C",
    ),
    (_ANY, "class C:\n    for i in rng:\n        pass\n    with a as b:\n        pass\n    import os"),
    # --- type parameters (3.12) ------------------------------------------
    ((3, 12), "def ident[T: Bound](x: T) -> T:\n    return x"),
    ((3, 12), "def pick[T: (A, B), *Ts, **P](x: T, *a: *Ts) -> R:\n    v: T = x\n    return v"),
    ((3, 12), "class Box[T](Base):\n    item: T\n    def get(self) -> T:\n        return self.item"),
    ((3, 12), "type Alias = list[Elem]"),
    ((3, 12), "type Pair[T] = tuple[T, Other]"),
    ((3, 12), "@deco(a)\ndef f[T](x: T = dflt):\n    return x, g"),
    ((3, 13), "def f[T = Dflt](x: T) -> T:\n    return x"),
    ((3, 13), "class C[T: Bound = Dflt, *Ts = *tuple[D2]]:\n    pass"),
    # --- f-strings as 3.12 parses them -------------------------------------
    ((3, 12), "x = f\"{a['k']} {f'{b:{w}}'} {(c := d)!r}\""),
    ((3, 12), "x = f'''{\n    a  # comment\n    + b\n}'''"),
    # --- template strings (3.14) -----------------------------------------
    ((3, 14), "v = t'{a} {b!r:>{w}} {(c := d)}'"),
    ((3, 14), "def f(x):\n    return t'{x} {glob}'"),
]


#: Snippets symtable is not asked about on some versions, with why. The
#: bytecode check still covers them.
SYMTABLE_QUIRKS = {
    # 3.12+ reports the module-level `x` as read only, though the folded-in
    # comprehension binds a hidden `x` too.
    "[x for x in x]": (3, 12),
}


def symtable_snippets() -> list[str]:
    return [s for s in snippets() if s not in SYMTABLE_QUIRKS or PY < SYMTABLE_QUIRKS[s]]


def snippets() -> list[str]:
    """The corpus entries this Python can parse."""
    return [source for since, source in SNIPPETS if PY >= since]


# ---------------------------------------------------------------------------
# symtable
# ---------------------------------------------------------------------------

_COMPREHENSION_TABLES = ("listcomp", "setcomp", "dictcomp", "genexpr")
#: Names Python adds to a scope for its own use.
_IMPLICIT = ("__class__", "__classdict__", "__type_params__", "__conditional_annotations__")

Facts = dict[str, set[str]]


class _Node:
    """One real scope (module, function, lambda, class, comprehension) in the
    common form: its name facts and its real child scopes."""

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name
        self.facts: Facts = {}
        self.children: list[_Node] = []

    def add(self, name: str, *flags: str) -> None:
        self.facts.setdefault(name, set()).update(flags)


def _table_kind(table) -> str:
    kind = table.get_type()
    return str(getattr(kind, "value", kind))


def _is_real(table) -> bool:
    return _table_kind(table) in ("module", "function", "class")


def _symbol_resolution(symbol) -> str:
    if symbol.is_free():
        return "free"
    return "local" if symbol.is_local() else "global"


def _escaping(table) -> dict[str, str]:
    """The names an annotation or type-parameter table (and the ones inside
    it) reads from outside itself, each with how it resolves there."""
    found: dict[str, str] = {}
    own = set()
    for symbol in table.get_symbols():
        name = symbol.get_name()
        if name.startswith(".") or name in _IMPLICIT:
            continue
        if symbol.is_local():
            own.add(name)
        elif symbol.is_referenced():
            found[name] = _symbol_resolution(symbol)
    for child in table.get_children():
        if not _is_real(child):
            for name, how in _escaping(child).items():
                if name not in own:
                    found.setdefault(name, how)
    return found


def _real_children(table) -> list:
    out = []
    for child in table.get_children():
        out.extend([child] if _is_real(child) else _real_children(child))
    return out


def _from_symtable(table) -> _Node:
    kind, name = _table_kind(table), table.get_name()
    node = _Node(kind, name)
    is_module = kind == "module"
    is_comprehension = kind == "function" and name in _COMPREHENSION_TABLES
    for symbol in table.get_symbols():
        ident = symbol.get_name()
        if ident.startswith(".") or ident in _IMPLICIT:
            continue
        flags: set[str] = set()
        if symbol.is_parameter():
            flags.add("param")
        if symbol.is_assigned() or symbol.is_imported():
            flags.add("bound")
        if symbol.is_referenced():
            flags.add("read")
        if symbol.is_declared_global():
            flags.add("global_decl")
        if symbol.is_nonlocal():
            flags.add("nonlocal_decl")
        if is_module:
            # symtable marks the target of a `:=` inside a module-level
            # comprehension as declared global and not as assigned.
            # Nor does it tell that from a `global` declaration in a function
            # of the code, which binds nothing now. Left to the bytecode.
            if "global_decl" in flags:
                flags.discard("global_decl")
                flags.add("bound?")
        else:
            flags.add(_symbol_resolution(symbol))
        if is_comprehension and not symbol.is_local():
            # The target of a `:=`: stored in the scope around the
            # comprehension, where both sides report it.
            flags -= {"bound", "global_decl", "nonlocal_decl"}
            if "read" not in flags:
                continue
        node.facts[ident] = flags
    for child in table.get_children():
        if not _is_real(child):
            # Annotations, type parameters, their bounds and type alias
            # values: scopes of their own from 3.12 / 3.14 on, read in place
            # before. Brought to one form: what they read from outside
            # counts as read by the scope around them.
            for ident, how in _escaping(child).items():
                had = ident in node.facts
                node.add(ident, "read")
                if not had and not is_module:
                    node.add(ident, how)
    node.children = [_from_symtable(child) for child in _real_children(table)]
    return node


# ---------------------------------------------------------------------------
# the model, in the same form
# ---------------------------------------------------------------------------

_AUX = (sm.TYPE_PARAMS, sm.ANNOTATION)


def _model_kind(scope: sm.Scope) -> tuple[str, str]:
    if scope.kind == sm.COMPREHENSION:
        return "function", {"generatorexp": "genexpr"}.get(scope.name, scope.name)
    if scope.kind == sm.LAMBDA:
        return "function", "lambda"
    return scope.kind, scope.name


class _ModelForm:
    def __init__(self, model: sm.ScopeModel, future_annotations: bool) -> None:
        self.future_annotations = future_annotations
        # symtable does not count the target of `x += 1` as read.
        self.reads: dict[int, set[str]] = {}
        for event in model.events:
            if event.kind == sm.READ and isinstance(event.node.ctx, ast.Load):
                self.reads.setdefault(id(event.scope), set()).add(event.name)

    def reads_of(self, scope: sm.Scope) -> set[str]:
        return self.reads.get(id(scope), set()) - set(_IMPLICIT)

    def seen(self, scope: sm.Scope) -> bool:
        """Does Python's symbol table look into this annotation scope?"""
        if scope.evaluated == sm.QUOTED:
            return False  # a string to Python
        if scope.evaluated == sm.NEVER:
            # A variable annotated inside a function: never evaluated; the
            # symbol table stopped visiting it in 3.14.
            return PY < (3, 14) and not self.future_annotations
        if scope.evaluated == sm.EVALUATED:
            return not self.future_annotations
        return True

    def escaping(self, scope: sm.Scope) -> set[str]:
        if scope.kind == sm.ANNOTATION and not self.seen(scope):
            return set()
        own = scope.local_names
        found = {name for name in self.reads_of(scope) if name not in own}
        for child in scope.children:
            if child.kind in _AUX:
                found |= self.escaping(child) - own
        return found

    def real_children(self, scope: sm.Scope) -> list[sm.Scope]:
        out: list[sm.Scope] = []
        for child in scope.children:
            if child.kind in _AUX:
                if child.kind == sm.ANNOTATION and not self.seen(child):
                    continue
                out.extend(self.real_children(child))
            else:
                out.append(child)
        return out

    def node(self, scope: sm.Scope, like) -> _Node:
        """*scope* in the common form; *like* is the symtable node it is
        compared with, which decides which comprehensions are folded in."""
        kind, name = _model_kind(scope)
        node = _Node(kind, name)
        is_module = scope.parent is None
        self.own_facts(scope, scope, node, is_module)
        children = self.real_children(scope)
        # From 3.12 on Python compiles a list, set or dict comprehension into
        # the scope around it and the symbol table reports its names there.
        # Where the table has no scope for a comprehension, the model's is
        # folded into its parent the same way.
        tables = {child.name for child in like.children if child.kind == "function"} if like is not None else set()
        while True:
            inlined = [c for c in children if c.kind == sm.COMPREHENSION and _model_kind(c)[1] not in tables]
            if not inlined:
                break
            for comp in inlined:
                for ident in comp.local_names:
                    node.add(ident, "bound")
                    if ident in self.reads_of(comp):
                        node.add(ident, "read")
                    if not is_module:
                        node.facts[ident] -= {"global", "free"}
                        node.add(ident, "local")
                self.own_facts(comp, scope, node, is_module, skip=comp.local_names)
                at = children.index(comp)
                children[at : at + 1] = self.real_children(comp)
        pairs = _pair(children, like.children if like is not None else [], key=_model_kind)
        node.children = [self.node(child, partner) for child, partner in pairs]
        return node

    def own_facts(self, scope: sm.Scope, into: sm.Scope, node: _Node, is_module: bool, skip=frozenset()) -> None:
        reads = self.reads_of(scope)
        names = (scope.params | scope.stores.keys() | reads | scope.declared_global | scope.declared_nonlocal) - set(
            skip
        )
        folded = scope is not into
        for ident in names:
            flags: set[str] = set()
            if ident in scope.params:
                flags.add("param")
            if ident in scope.stores and not folded:
                flags.add("bound")
            if ident in reads:
                flags.add("read")
            if not folded:
                if ident in scope.declared_global and not is_module:
                    flags.add("global_decl")
                if ident in scope.declared_nonlocal:
                    flags.add("nonlocal_decl")
            if not flags:
                continue
            had = ident in node.facts
            node.add(ident, *flags)
            if not is_module and not (folded and had):
                node.add(ident, self.resolution(into, ident))
        for child in scope.children:
            if child.kind in _AUX:
                for ident in self.escaping(child) - set(skip):
                    had = ident in node.facts
                    node.add(ident, "read")
                    if not had and not is_module:
                        node.add(ident, self.resolution(into, ident))

    @staticmethod
    def resolution(scope: sm.Scope, name: str) -> str:
        owner = scope.owner(name)
        if owner is scope:
            return "local"
        return "global" if owner.parent is None else "free"


def _pair(children: list, others: list, key) -> list[tuple]:
    """Each of *children* with the first unused entry of *others* of its kind
    and name (None when there is none)."""
    unused = list(others)
    pairs = []
    for child in children:
        want = key(child)
        partner = next((o for o in unused if (o.kind, o.name) == want), None)
        if partner is not None:
            unused.remove(partner)
        pairs.append((child, partner))
    return pairs


def _compare(ours: _Node, theirs: _Node | None, path: str, out: list[str]) -> None:
    here = f"{path}/{ours.kind}:{ours.name}"
    if theirs is None:
        out.append(f"{here}: a scope symtable does not have")
        return
    for name in sorted(ours.facts.keys() | theirs.facts.keys()):
        mine, python = set(ours.facts.get(name, ())), set(theirs.facts.get(name, ()))
        if "bound?" in python:
            mine.discard("bound")
            python -= {"bound", "bound?"}
        if mine != python:
            out.append(f"{here}: {name}: model {sorted(mine)} != symtable {sorted(python)}")
    if len(ours.children) != len(theirs.children):
        out.append(
            f"{here}: scopes inside: model {[(c.kind, c.name) for c in ours.children]}"
            f" != symtable {[(c.kind, c.name) for c in theirs.children]}"
        )
    for child, partner in _pair(ours.children, theirs.children, key=lambda c: (c.kind, c.name)):
        _compare(child, partner, here, out)


def differences(source: str) -> list[str]:
    """Where the scope model of *source* and Python's symbol table disagree
    (an empty list when they agree)."""
    theirs = _from_symtable(symtable.symtable(source, "<snippet>", "exec"))
    tree = ast.parse(source)
    future = any(
        isinstance(stmt, ast.ImportFrom)
        and stmt.module == "__future__"
        and any(a.name == "annotations" for a in stmt.names)
        for stmt in tree.body
    )
    model = sm.scope_model(tree)
    ours = _ModelForm(model, future).node(model.module, theirs)
    out: list[str] = []
    _compare(ours, theirs, "", out)
    return out


# ---------------------------------------------------------------------------
# bytecode
# ---------------------------------------------------------------------------

_GLOBAL_STORES = ("STORE_GLOBAL",)
_NAME_STORES = ("STORE_NAME",)
_DELETES = ("DELETE_NAME", "DELETE_GLOBAL")
_GLOBAL_LOADS = ("LOAD_GLOBAL", "LOAD_FROM_DICT_OR_GLOBALS")
_COMPREHENSION_CODE = ("<listcomp>", "<setcomp>", "<dictcomp>", "<genexpr>")


class Compiled:
    """The module-level names the compiled code touches."""

    def __init__(self) -> None:
        #: Stored when the code runs.
        self.stores: set[str] = set()
        #: Stored when a function of the code is called.
        self.stores_later: set[str] = set()
        #: Deleted, when the code runs or later.
        self.deletes: set[str] = set()
        #: Loaded from the module's namespace or the builtins, now or later.
        self.loads: set[str] = set()
        #: Is there a ``from m import *``?
        self.star = False


def compiled_names(source: str) -> Compiled:
    """What compiling *source* shows it stores into, deletes from and loads
    from the module's namespace."""
    found = Compiled()
    _scan(compile(source, "<snippet>", "exec", dont_inherit=True), found, later=False, top=True)
    return found


def _scan(code: types.CodeType, found: Compiled, later: bool, top: bool) -> None:
    stored_here = {ins.argval for ins in dis.get_instructions(code) if ins.opname in _NAME_STORES}
    for ins in dis.get_instructions(code):
        op, name = ins.opname, ins.argval
        if op == "IMPORT_STAR" or (op == "CALL_INTRINSIC_1" and "IMPORT_STAR" in str(ins.argrepr)):
            found.star = True
        elif op in _GLOBAL_STORES or (top and op in _NAME_STORES):
            (found.stores_later if later else found.stores).add(name)
        elif op in _DELETES and (top or op == "DELETE_GLOBAL"):
            found.deletes.add(name)
        elif op in _GLOBAL_LOADS:
            found.loads.add(name)
        elif op == "LOAD_NAME" and (top or name not in stored_here):
            # In a class body: the class's own name when it has stored it,
            # else the module's.
            found.loads.add(name)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            # A class body, a comprehension and the helper around a generic
            # definition run with the statement; any other code when called.
            runs_now = (
                not const.co_flags & 0x1
                or const.co_name in _COMPREHENSION_CODE
                or const.co_name.startswith("<generic parameters")
            )
            _scan(const, found, later=later or not runs_now, top=False)


if __name__ == "__main__":
    failed = 0
    for text in symtable_snippets():
        try:
            problems = differences(text)
        except SyntaxError as error:
            problems = [f"does not parse here: {error}"]
        if problems:
            failed += 1
            print(repr(text))
            for line in problems:
                print("   ", line)
    print(f"{sys.version.split()[0]}: {len(snippets())} snippets, {failed} with differences")
    sys.exit(1 if failed else 0)
