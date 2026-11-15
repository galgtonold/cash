"""A class named only to test objects against does not make its bases'
helpers part of the key; a class that runs does.

nilearn's logger defaults to ``(NilearnBaseEstimator,)`` and only tests
objects against it. The estimator inherits a documentation-link property
from a mixin, and that property calls a module function reading
``__version__``, which changes on every commit of a source checkout. Walking
the user bases of every class (so the functions an inherited method calls are
keyed) also walked them for the class only tested against, and the version
read moved the key of every cached function that logs.

The class only tested against is still keyed by its code and its bases'
code (an edit there can change what the test answers), and the same class
constructed on another path, whichever is met first, keeps its bases'
helpers keyed.
"""

from __future__ import annotations

import textwrap

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

VERSION = 'VERSION = "1.0.dev1"\n'

MODELS = """
    from version import VERSION


    def _doc_url():
        return "https://docs/" + VERSION


    def _scale(v):
        return v * 2


    class _DocMixin:
        @property
        def doc_link(self):
            return _doc_url()

        def scaled(self, v):
            return _scale(v)


    class Estimator(_DocMixin):
        def __init__(self):
            self.ready = True
"""

CACHED = """
    import cash
    from helpers import describe

    @cash.cache
    def f(x):
        print("ran")
        return describe(x)

    print(f(3))
"""

TESTED_ONLY = """
    from models import Estimator

    def describe(obj, classes=(Estimator,)):
        return "est" if isinstance(obj, classes) else type(obj).__name__
"""

#: The class tested against in one helper and constructed in another, in
#: both orders: its base's helpers decide the result.
ALSO_USED = {
    "isinstance helper written first": """
        from models import Estimator

        def _kind(obj, classes=(Estimator,)):
            return "est" if isinstance(obj, classes) else type(obj).__name__

        def _value(obj):
            return Estimator().scaled(obj)

        def describe(obj):
            return f"{_kind(obj)}:{_value(obj)}"
    """,
    "constructing helper written first": """
        from models import Estimator

        def _value(obj):
            return Estimator().scaled(obj)

        def _kind(obj, classes=(Estimator,)):
            return "est" if isinstance(obj, classes) else type(obj).__name__

        def describe(obj):
            return f"{_value(obj)}:{_kind(obj)}"
    """,
}


def _files(helpers: str) -> dict[str, str]:
    return {"main.py": CACHED, "helpers.py": helpers, "models.py": MODELS, "version.py": VERSION}


def test_a_global_a_base_helper_reads_does_not_move_the_key(tmp_path):
    first, after, uncached = edited_runs(tmp_path, _files(TESTED_ONLY), [("version.py", "dev1", "dev2")])
    assert first == "ran\nint"
    assert uncached == "ran\nint"
    assert after == "int", "a global read by a helper of a tested-only class's base moved the key"


def test_an_edit_to_the_base_of_a_class_only_tested_against_misses(tmp_path):
    # The class's own code and its bases' code can change what the test
    # answers: an edit to the mixin is seen.
    first, after, uncached = edited_runs(
        tmp_path, _files(TESTED_ONLY), [("models.py", "return _doc_url()", "return _doc_url() + '/'")]
    )
    assert first == "ran\nint"
    assert uncached == "ran\nint"
    assert after == "ran\nint", "an edit to the base class of a class tested against was not seen"


#: An edit to the code of a function the base's method calls, and to a
#: global it reads.
BASE_HELPER_EDITS = {
    "its code": ("models.py", "return v * MULT", "return v * MULT * 10"),
    "a global it reads": ("version.py", "MULT = 2", "MULT = 5"),
}


@pytest.mark.parametrize("edit", sorted(BASE_HELPER_EDITS))
@pytest.mark.parametrize("trace", ["0", "1"], ids=["trace_off", "trace_on"])
@pytest.mark.parametrize("order", sorted(ALSO_USED))
def test_the_class_constructed_elsewhere_keeps_its_bases_helpers(tmp_path, order, trace, edit):
    files = _files(ALSO_USED[order])
    files["models.py"] = MODELS.replace("import VERSION", "import VERSION, MULT").replace("v * 2", "v * MULT")
    files["version.py"] = VERSION + "MULT = 2\n"
    files["main.py"] = f"import os\nos.environ['CASH_TRACE'] = {trace!r}\n" + textwrap.dedent(CACHED)
    first, after, uncached = edited_runs(tmp_path, files, [BASE_HELPER_EDITS[edit]])
    assert first.startswith("ran")
    assert after == uncached
    assert after.startswith("ran"), "an edit to a helper of a constructed class's base was served stale"
    assert first != after
