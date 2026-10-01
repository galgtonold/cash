"""A function that writes a file is a writer whichever way the call is spelled.

Four export functions lived in the user's own project
module and were called as ``tl.export_corpus_summary(...)``. Every one was
cached. They deleted a deliverable from ``report/``, re-ran the cell, got
``CACHED (saved 0.00s)``, and the file did not come back -- for two sessions
their output folder was only accidentally right, because the files happened to
already be on disk.

``docs/how-it-works/safety.md`` promises exactly this case:

    "Source 3 also refuses a call to a function of yours -- defined in the
     notebook or your project, not an installed package -- whose body writes
     a file"

and it is the *your project* half that was not covered. ``user_callee_writing_files``
judges the function correctly and even follows the write one call further
down; it was simply never handed the callee, because
``statement_calls_user_writer`` only matched ``isinstance(node.func, ast.Name)``.
A function in your project is normally reached as ``mymodule.func(...)``, an
``ast.Attribute`` -- so the spelling the docs single out is the one that was
skipped, while a notebook-defined ``save(...)`` was covered all along.

The controls matter as much as the failing case: an installed package's method
(``df.to_csv``) must NOT become a reason to refuse caching every statement that
touches a DataFrame, and an append must stay a skippable effect.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import _writer_lib

from cash.analysis.annotations import CacheAnnotation
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability
from cash.analysis.namespace_effects import statement_calls_user_writer, user_callee_writing_files


@pytest.fixture
def ns():
    """A namespace holding the module, an alias of it, and a bare name."""
    return {
        "tl": _writer_lib,
        "_writer_lib": _writer_lib,
        "export_summary": _writer_lib.export_summary,
        "tidy": _writer_lib.tidy,
        "note": _writer_lib.note,
    }


def test_the_predicate_itself_was_never_the_problem():
    """Establishes the split: the analysis is right, the dispatch was not."""
    assert user_callee_writing_files(_writer_lib.export_summary) == "export_summary"
    assert user_callee_writing_files(_writer_lib.tidy) is None


def test_a_bare_name_writer_is_seen(ns):
    """The shape that always worked -- a function defined in the notebook."""
    assert statement_calls_user_writer("p = export_summary(d, 'out.json')", ns) == "export_summary"


def test_a_module_spelled_writer_is_seen(ns):
    """The reported shape, and the one the docs promise: a project module."""
    assert statement_calls_user_writer("p = tl.export_summary(d, 'out.json')", ns) == "export_summary"


def test_the_module_may_be_named_in_full(ns):
    """`import mypkg.helpers` then `mypkg.helpers.save(...)` is the same case."""
    assert statement_calls_user_writer("p = _writer_lib.save_chart(fig, 'chart.png')", ns) == "save_chart"


def test_a_module_function_that_writes_nothing_is_not_a_writer(ns):
    """The control that stops this refusing every module call."""
    assert statement_calls_user_writer("v = tl.tidy(values)", ns) is None


def test_an_append_through_a_module_is_still_not_a_writer(ns):
    """An append is the effect a cache hit is understood to skip."""
    assert statement_calls_user_writer("tl.note('hello')", ns) is None


def test_an_installed_packages_method_is_not_looked_into(ns):
    """`df.to_csv(path)` must not make every DataFrame statement uncacheable.

    Its source says nothing about what the call does with the user's files,
    and treating it as a writer here would refuse far more than it protects.
    """
    pd = pytest.importorskip("pandas")
    local = dict(ns)
    local["df"] = pd.DataFrame({"a": [1, 2]})
    assert statement_calls_user_writer("df.to_csv('out.csv')", local) is None


def test_an_unknown_name_is_not_a_writer(ns):
    """A module that is not in the namespace must not raise or guess."""
    assert statement_calls_user_writer("p = nope.save(x)", ns) is None
    assert statement_calls_user_writer("p = tl.no_such_function(x)", ns) is None


def test_a_write_one_module_further_down_is_seen(ns):
    """``tl.build_report`` hands the write to another project module as
    ``_writer_io.write_table(...)``: the helper's own call is spelled through a
    module too, and is followed like a bare-name call."""
    assert user_callee_writing_files(_writer_lib.build_report) == "write_table"
    assert statement_calls_user_writer("tl.build_report(rows, 'out.txt')", ns) == "write_table"


def _verdict(code: str, ns: dict, annotation=None) -> tuple[bool, list[str]]:
    tree = ast.parse(code)
    return decide_cacheability(
        code=code,
        tree=tree,
        inputs=set(),
        outputs=set(),
        annotation=annotation,
        analysis=analyze_statement(code, tree),
        user_ns=ns,
        variable_lineage={},
        is_stateful_call=lambda _name: False,
        scan_forbidden=lambda _code, _ns, _tree: [],
    )


@pytest.mark.parametrize(
    ("code", "writer"),
    [
        ("tl.export_summary(d, 'out.json')", "export_summary"),
        ("p = _writer_lib.save_chart(fig, 'chart.png')", "save_chart"),
        ("tl.build_report(rows, 'out.txt')", "write_table"),
        ("export_summary(d, 'out.json')", "export_summary"),  # ``from tl import export_summary``
    ],
)
def test_the_cacheability_decision_refuses_a_project_module_writer(ns, code, writer):
    """The decision that stores a statement is the one that must see it.

    ``tl.export_summary(...)`` was judged a writer by the upstream replay and
    the file-answer check, but the cacheability decision offered only
    bare-name callees to the writer check: the statement was stored, and the
    next run restored it without writing the file.
    """
    cacheable, reasons = _verdict(code, ns)
    assert cacheable is False
    assert len(reasons) == 1
    assert f"which writes files ({writer})" in reasons[0]


def test_the_decision_still_caches_module_calls_that_replace_no_file(ns):
    """Controls: no write, and an append, keep caching through a module."""
    assert _verdict("v = tl.tidy(values)", ns) == (True, [])
    assert _verdict("tl.note('hello')", ns) == (True, [])


def test_assume_safe_waives_a_project_module_writer(ns):
    """``# @cash:assume-safe`` waives the write however the call is spelled."""
    assert _verdict("tl.export_summary(d, 'out.json')", ns, CacheAnnotation(assume_safe=True)) == (True, [])
