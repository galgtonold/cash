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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import _writer_lib

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
