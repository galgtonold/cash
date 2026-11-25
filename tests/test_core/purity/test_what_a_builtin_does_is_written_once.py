"""What a builtin does with its arguments is written down once.

Ten modules each kept their own list of "builtins that only read what they
are handed", and the lists disagreed: two under the same sentence differed
by fifteen names, ``getattr`` (which hands back ``G.append``) counted as
reading only, ``hash`` counted as following from its arguments alone, and a
name added to one list was missing from the next. Every builtin now has one
row saying what it does, checked here against the builtin itself, and each
module's question is a set derived from the rows.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "cash"

#: A call of each builtin on arguments it accepts.
SAMPLE_CALLS = {
    "abs": "abs(-1)", "ascii": "ascii(X)", "bin": "bin(3)", "bool": "bool(X)", "bytes": "bytes(iter([1]))",
    "callable": "callable(X)", "chr": "chr(65)", "complex": "complex(1)", "divmod": "divmod(7, 2)",
    "float": "float(1)", "format": "format(1)", "hex": "hex(3)", "int": "int(1.5)", "isinstance": "isinstance(X, list)",
    "issubclass": "issubclass(int, object)", "len": "len(X)", "oct": "oct(3)", "ord": "ord('a')", "pow": "pow(2, 3)",
    "range": "range(3)", "repr": "repr(X)", "round": "round(1.5)", "str": "str(X)", "all": "all(X)", "any": "any(X)",
    "bytearray": "bytearray(iter([1]))", "max": "max(X)", "min": "min(X)", "sum": "sum(X, [])", "next": "next(IT)",
    "dict": "dict(a=X[0])", "frozenset": "frozenset(X[0])", "list": "list(X)", "set": "set(X[0])",
    "slice": "slice(X[0])", "sorted": "sorted(X, key=id)", "tuple": "tuple(X)", "enumerate": "enumerate(X)",
    "filter": "filter(None, X)", "iter": "iter(X)", "map": "map(lambda x: x, X)", "reversed": "reversed(X)",
    "zip": "zip(X)", "getattr": "getattr(O, 'rows')", "hasattr": "hasattr(O, 'rows')", "hash": "hash('a')",
    "id": "id(X)", "type": "type(O)", "vars": "vars(O)", "print": "print(X)",
}  # fmt: skip


class _Holder:
    def __init__(self, rows):
        self.rows = rows


def _run(name: str):
    """The sample call of *name*, and the arguments it was handed."""
    items = [[float("1.5")], [float("2.5")]]
    scope = {"X": items, "IT": iter(items), "O": _Holder(items)}
    return eval(SAMPLE_CALLS[name], scope), scope


def _reaches(value, wanted, depth=3) -> bool:
    """Whether *value* is, or holds within *depth* levels, one of *wanted*."""
    if any(value is w for w in wanted):
        return True
    if depth == 0 or isinstance(value, (str, bytes, bytearray, range, int, float, complex)):
        return False
    if isinstance(value, slice):
        return _reaches([value.start, value.stop, value.step], wanted, depth - 1)
    if isinstance(value, dict):
        value = list(value.values())
    try:
        parts = list(value)
    except TypeError:
        return False
    return any(_reaches(part, wanted, depth - 1) for part in parts)


def test_every_builtin_of_the_table_has_a_sample_call():
    from cash.builtin_calls import ARGUMENT_BUILTINS

    assert set(SAMPLE_CALLS) == ARGUMENT_BUILTINS
    assert all(hasattr(builtins, name) for name in ARGUMENT_BUILTINS)


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
def test_a_row_says_what_the_builtin_really_does(name, capsys):
    from cash import builtin_calls as table

    row = table.traits(name)
    result, scope = _run(name)
    items, holder = scope["X"], scope["O"]

    one_shot = iter(result) is result if hasattr(result, "__next__") else False
    assert one_shot == (table.LAZY in row)

    part = result is holder.__dict__ or result is items and name in ("getattr", "vars")
    assert bool(part) == (table.PART in row)
    held = not part and result is not type(holder) and _reaches(result, [items, *items, items[0][0], items[1][0]])
    assert held == bool(row & {table.HOLDS, table.ITEM}), "the result holds what was handed, or does not"
    assert (result is type(holder)) == (table.CLASS in row)

    assert (next(scope["IT"]) is items[1]) == (table.ADVANCES in row)
    assert bool(capsys.readouterr().out) == (table.PRINTS in row)
    if table.NUMBER in row and name != "sum":
        assert isinstance(result, (int, float, complex, tuple))


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
def test_a_builtin_that_empties_an_iterator_is_marked(name):
    from cash import builtin_calls as table

    gen = (number for number in [1, 2])
    builtin = getattr(builtins, name)
    try:
        out = builtin(lambda number: number, gen) if name in ("map", "filter") else builtin(gen)
        if table.LAZY in table.traits(name) and out is not gen:
            list(out)
    except TypeError:
        pass
    took_one = next(gen, None) != 1
    assert took_one == (table.DRAWS in table.traits(name))


@pytest.mark.parametrize("name", ["map", "filter", "sorted", "min", "max", "iter"])
def test_a_builtin_that_calls_a_function_it_is_handed_is_marked(name):
    from cash.builtin_calls import CALL_WHAT_THEY_ARE_HANDED

    calls = []

    def seen(*args):
        calls.append(args)
        return 1

    builtin = getattr(builtins, name)
    if name in ("map", "filter"):
        list(builtin(seen, [3]))
    elif name == "iter":
        next(builtin(seen, 0))
    else:
        builtin([3, 4], key=seen)
    assert calls and name in CALL_WHAT_THEY_ARE_HANDED
    assert CALL_WHAT_THEY_ARE_HANDED == {"map", "filter", "sorted", "min", "max", "iter"}


def test_each_question_is_answered_from_the_rows():
    from cash import builtin_calls as table
    from cash.analysis import callee_effects, namespace_effects, purity_flow, unloaded_modules
    from cash.notebook import helper_memos
    from cash.notebook.control_structures import share_candidates, single_unit_policy
    from cash.notebook.statement import reducer_writes, unseen_writes
    from cash.notebook.upstream import simulator
    from cash.purity import KNOWN_PURE_BUILTINS

    every = table.ARGUMENT_BUILTINS
    # Changing nothing handed: all but the one that advances and the two that
    # hand back a part to change the object through.
    assert reducer_writes._READS_ONLY == every - {"next", "getattr", "vars"}
    # The two lists that were kept under one sentence differ by what calls a
    # function on the items, and by nothing else.
    assert reducer_writes._READS_ONLY - namespace_effects.ARGUMENT_READING_BUILTINS == {"map", "filter"}
    assert namespace_effects.ARGUMENT_READING_BUILTINS <= reducer_writes._READS_ONLY
    assert purity_flow._READ_ONLY_CALLS == every - {"getattr", "vars"}
    no_effect = every - {"next", "getattr", "vars", "hasattr", "print"}
    assert unseen_writes._PURE_BUILTINS == single_unit_policy.PURE_ITER_PRODUCERS == no_effect
    # An address or a salted hash does not follow from the arguments alone.
    assert helper_memos._PURE_BUILTINS == simulator._PURE_BUILTINS == no_effect - {"hash", "id"}
    assert {f"builtins.{name}" for name in no_effect - {"hash", "id"}} <= unloaded_modules._SOURCE_ONLY_CALLS
    assert not {"builtins.hash", "builtins.id", "builtins.print"} & unloaded_modules._SOURCE_ONLY_CALLS
    assert helper_memos._CALLING_BUILTINS == table.CALL_WHAT_THEY_ARE_HANDED
    assert callee_effects._SCALAR_BUILTINS == reducer_writes._HOLDS_NOTHING_OF_IT
    assert {"len", "str", "all", "bytes", "range", "hash"} <= callee_effects._SCALAR_BUILTINS
    assert not {"list", "max", "sum", "next", "getattr", "type", "sorted"} & callee_effects._SCALAR_BUILTINS
    assert {"max", "sum", "print", "type", "bytes"} <= share_candidates._STORE_NOTHING
    assert not {"list", "zip", "next", "getattr", "vars"} & share_candidates._STORE_NOTHING
    assert KNOWN_PURE_BUILTINS == every - {"print"}


def test_a_global_looked_into_by_name_counts_as_handed_over():
    from cash.notebook.statement.reducer_writes import _handed_to_libraries

    scope: dict = {"ROWS": [1, 2]}
    source = (
        "def add(x):\n    getattr(ROWS, 'append')(x)\n"
        "def into(x):\n    vars(ROWS)\n"
        "def size(x):\n    return len(ROWS) + sum(map(abs, ROWS))\n"
    )
    exec(source, scope)
    bodies = {node.name: node for node in ast.parse(source).body}
    assert _handed_to_libraries(scope["add"], bodies["add"]) == {"ROWS"}
    assert _handed_to_libraries(scope["into"], bodies["into"]) == {"ROWS"}
    assert _handed_to_libraries(scope["size"], bodies["size"]) == set()


def test_a_binding_from_a_hash_is_not_taken_as_following_from_the_text():
    from cash.notebook.upstream.simulator import _binds_without_reading

    assert not _binds_without_reading("KEY = hash('run-7')", {})
    assert not _binds_without_reading("KEY = getattr(Path('.'), 'name')", {"Path": Path})
    assert _binds_without_reading("N = len(range(3)) + sum([1, 2]) + all([1])", {})
    assert _binds_without_reading("LABEL = format(3, '02d') + chr(65)", {})


def _literal_names(node: ast.AST) -> set[str] | None:
    """The words a literal collection of strings, or one string of words, holds."""
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        if node.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return {e.value for e in node.elts}
    elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" not in node.value:
        words = node.value.split()
        if len(words) > 1 and all(word.isidentifier() for word in words):
            return set(words)
    return None


def test_no_module_writes_out_its_own_list_of_these_builtins():
    """A list of twenty or more names, most of them rows of the table, is a
    copy of it: the question it answers belongs at the foot of the table."""
    from cash.builtin_calls import ARGUMENT_BUILTINS

    copies = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "builtin_calls.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            # A set written as an assignment or as an argument, in one piece
            # or as pieces joined together.
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            names: set[str] = set()
            for sub in ast.walk(node.value):
                names |= _literal_names(sub) or set()
            rows = names & ARGUMENT_BUILTINS
            if len(rows) >= 20 and len(rows) > 0.75 * len(names):
                copies.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert copies == []
