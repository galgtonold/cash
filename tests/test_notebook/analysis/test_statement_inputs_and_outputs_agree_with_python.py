"""What cash says a statement reads and writes agrees with what Python does
when it compiles the statement.

`CodeAnalyzer.analyze_code_block` feeds the cache keys: a name it misses as
an output leaves the cells below reading a value from before an edit, and a
name it wrongly lists as an input ties the statement to a variable it never
reads. The compiled bytecode says which names a statement stores into and
loads from the module's namespace; over the corpus of `tests/_scope_oracle.py`
(every kind of statement) the analysis has to stay between the two.
"""

from __future__ import annotations

import ast
import sys

import pytest

from cash.analysis import scope_model as sm
from cash.analysis.code_analyzer import CodeAnalyzer
from tests import _scope_oracle as oracle


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _only_in_annotations(source: str) -> set[str]:
    model = sm.scope_model(ast.parse(source))
    return model.global_reads() - model.global_reads(annotations=False)


@pytest.mark.parametrize("source", oracle.snippets())
def test_outputs_are_what_the_compiled_statement_stores(source):
    compiled = oracle.compiled_names(source)
    _inputs, outputs = CodeAnalyzer.analyze_code_block(source)

    # `except E as e` stores `e` and deletes it when the handler ends.
    stored = {name for name in compiled.stores - compiled.deletes if not _dunder(name)}
    assert stored <= outputs
    # Anything else is an object changed in place (`x[i] = a`), which the
    # statement loads, or the mark a star import leaves.
    extra = outputs - stored
    assert extra <= compiled.loads | ({"*"} if compiled.star else set())
    assert ("*" in outputs) == compiled.star


@pytest.mark.parametrize("source", oracle.snippets())
def test_inputs_are_what_the_compiled_statement_loads_and_does_not_bind_first(source):
    compiled = oracle.compiled_names(source)
    inputs, _outputs = CodeAnalyzer.analyze_code_block(source)
    loaded = {name for name in compiled.loads if not _dunder(name)}
    annotated = _only_in_annotations(source)

    # Never a name the statement does not load (a `match` capture, a
    # comprehension variable, a parameter, a local of a function).
    # (An annotation Python keeps as text or never evaluates loads nothing;
    # the names in it are inputs all the same.)
    assert inputs <= loaded | annotated
    # Every name it loads and never stores is an input.
    assert loaded - compiled.stores - compiled.stores_later <= inputs


MATCH_SHAPES = {
    "a mapping holding a sequence": ("match data:\n    case {'k': [a, *rest]}:\n        pass", {"a", "rest"}),
    "the rest of a mapping": ("match data:\n    case {'k': v, **others}:\n        pass", {"v", "others"}),
    "a class pattern": ("match data:\n    case Point(x=px, y=0):\n        pass", {"px"}),
    "positional class patterns": ("match data:\n    case Point(px, py):\n        pass", {"px", "py"}),
    "a sequence with a star": ("match data:\n    case [first, *others]:\n        pass", {"first", "others"}),
    "a capture with a guard": ("match data:\n    case x if x > 0:\n        pass", {"x"}),
    "an as capture": ("match data:\n    case [1, 2] as whole:\n        pass", {"whole"}),
    "an as capture of a class": ("match data:\n    case str() as s:\n        pass", {"s"}),
    "alternatives": ("match data:\n    case [b] | (b, _):\n        pass", {"b"}),
}


@pytest.mark.parametrize("shape", sorted(MATCH_SHAPES))
def test_a_name_a_match_pattern_captures_is_written_not_read(shape):
    source, captured = MATCH_SHAPES[shape]
    inputs, outputs = CodeAnalyzer.analyze_code_block(source)
    assert captured <= outputs
    assert not (captured & inputs)
    assert "data" in inputs


def test_a_capture_is_read_by_the_body_without_becoming_an_input():
    inputs, outputs = CodeAnalyzer.analyze_code_block(
        "match data:\n    case {'k': [a, *rest]} if a > low:\n        out = a + len(rest)"
    )
    assert inputs == {"data", "low", "len"}
    assert outputs == {"a", "rest", "out"}


def test_a_wildcard_and_a_value_pattern_bind_nothing():
    inputs, outputs = CodeAnalyzer.analyze_code_block(
        "match data:\n    case Color.RED:\n        pass\n    case _:\n        pass"
    )
    assert outputs == set()
    assert inputs == {"data", "Color"}


def test_the_name_of_an_except_clause_is_not_an_input():
    inputs, outputs = CodeAnalyzer.analyze_code_block(
        "try:\n    r = risky()\nexcept ValueError as e:\n    msg = str(e)"
    )
    assert "e" not in inputs
    # Python unbinds it when the handler ends: the statement leaves no `e`.
    assert outputs == {"r", "msg"}


def test_an_annotation_without_a_value_binds_nothing():
    inputs, outputs = CodeAnalyzer.analyze_code_block("x: int")
    assert outputs == set()
    assert CodeAnalyzer.analyze_code_block("x: int = 1")[1] == {"x"}


def test_an_annotated_assignment_reads_its_own_name_from_before():
    inputs, outputs = CodeAnalyzer.analyze_code_block("x: int = x + 1")
    assert "x" in inputs and outputs == {"x"}


def test_a_method_does_not_see_the_names_of_its_class_body():
    # `return scale` in a method reads the notebook's `scale`, not the
    # class attribute of the same name.
    inputs, _outputs = CodeAnalyzer.analyze_code_block(
        "class Model:\n    scale = 2\n    def run(self, v):\n        return v * scale"
    )
    assert "scale" in inputs
    # The class body itself reads its own name once it has bound it.
    inputs, _outputs = CodeAnalyzer.analyze_code_block("class Model:\n    scale = 2\n    double = scale * 2")
    assert "scale" not in inputs


def test_a_local_assigned_later_in_a_function_is_not_an_input():
    inputs, _outputs = CodeAnalyzer.analyze_code_block(
        "def outer():\n    def inner():\n        return late\n    late = 1\n    return inner"
    )
    assert "late" not in inputs


def test_what_an_augmented_attribute_assignment_reaches_through_is_read():
    inputs, _outputs = CodeAnalyzer.analyze_code_block("lookup(key).count += 1")
    assert {"lookup", "key"} <= inputs


def test_a_loop_reading_the_name_it_rebinds_reads_it():
    inputs, outputs = CodeAnalyzer.analyze_code_block("for row in row:\n    pass")
    assert "row" in inputs and "row" in outputs


def test_a_class_body_storing_through_global_writes_the_notebook_variable():
    _inputs, outputs = CodeAnalyzer.analyze_code_block("class C:\n    global g\n    g = 1")
    assert outputs == {"C", "g"}


def test_a_function_storing_through_global_writes_when_called_not_when_defined():
    inputs, outputs = CodeAnalyzer.analyze_code_block("def f():\n    global g\n    g = 1\n    return g")
    assert outputs == {"f"}
    assert "g" not in inputs


def test_a_reassigned_name_leaves_out_what_is_only_changed_in_place():
    assert CodeAnalyzer.reassigned_names("df = load()\nstats['n'] = len(df)") == {"df"}
    assert CodeAnalyzer.reassigned_names("match data:\n    case [head, *_]:\n        pass") == {"head"}


def test_a_class_named_in_a_hint_is_an_input_of_the_def():
    inputs, outputs = CodeAnalyzer.analyze_code_block(
        "def h(a: Pos, b=defb, *args: Var, c: Kw = defc, **kw: Kws) -> Ret:\n    pass"
    )
    assert inputs == {"Pos", "Var", "Kw", "Kws", "Ret", "defb", "defc"}
    assert outputs == {"h"}


def test_a_parameter_named_in_a_hint_of_an_inner_function_is_not_an_input():
    inputs, _outputs = CodeAnalyzer.analyze_code_block(
        "def outer(kind):\n    def inner(x: kind) -> kind:\n        return x\n    return inner"
    )
    assert inputs == set()


def test_the_names_in_a_quoted_hint_are_inputs():
    inputs, _outputs = CodeAnalyzer.analyze_code_block(
        "def f(x: 'Kind', y: Optional['Other'] = None) -> 'list[Kind]':\n    return x"
    )
    assert {"Kind", "Other", "Optional", "list"} <= inputs


def test_a_hint_that_is_not_python_is_no_input():
    inputs, _outputs = CodeAnalyzer.analyze_code_block("def f(x: 'not a name') -> 'a b':\n    return x")
    assert inputs == set()


def test_hints_are_inputs_under_the_future_import_too():
    inputs, _outputs = CodeAnalyzer.analyze_code_block(
        "from __future__ import annotations\ndef f(x: Kind) -> Kind:\n    return x"
    )
    assert "Kind" in inputs


@pytest.mark.skipif(sys.version_info < (3, 12), reason="type parameters are Python 3.12 syntax")
def test_the_bound_of_a_type_parameter_is_an_input_and_the_parameter_is_not():
    inputs, outputs = CodeAnalyzer.analyze_code_block("def ident[T: Kind](x: T) -> T:\n    return x")
    assert inputs == {"Kind"}
    assert outputs == {"ident"}
    inputs, outputs = CodeAnalyzer.analyze_code_block("class Box[T: Kind](Base):\n    item: T")
    assert inputs == {"Kind", "Base"}
    inputs, outputs = CodeAnalyzer.analyze_code_block("type Pair[T] = tuple[T, Other]")
    assert inputs == {"tuple", "Other"} and outputs == {"Pair"}
