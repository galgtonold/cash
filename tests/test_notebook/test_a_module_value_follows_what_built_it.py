"""A module-level value depends on what its right-hand side reads.

``ModuleInvalidator`` keeps the lineage of a name bound from a module when
none of the symbols it depends on changed ("granular preservation"). The
dependency map came from ``FunctionTracker._collect_intra_refs``, which only
followed references OUT OF functions and classes. So in

    def build_table(): ...
    TABLE = build_table()
    def lookup(k): return TABLE[k]

an edit to ``build_table`` expanded to nothing: ``TABLE`` and ``lookup`` both
counted as unchanged, and a lineage built on the old table could be kept.
Found while adding per-symbol keys (e371578), whose own closure walk
(``module_symbols``) has always followed these edges; the map is now built
from that same analysis, so the two cannot disagree about what reaches what.
"""

from cash.notebook.function_tracker import FunctionTracker

SOURCE = """
def build_table():
    return {"a": 1}

TABLE = build_table()

def lookup(k):
    return TABLE[k]

LIMIT: int = len(TABLE)
LO, HI = sorted(TABLE.values()) * 2

def unrelated():
    return 0
"""


def _deps(tmp_path, source=SOURCE):
    path = tmp_path / "lib.py"
    path.write_text(source, encoding="utf-8")
    return FunctionTracker.get_intra_module_call_deps(str(path))


def test_an_assignment_depends_on_the_function_that_builds_it(tmp_path):
    deps = _deps(tmp_path)
    assert "build_table" in deps.get("TABLE", set()), deps


def test_annotated_and_unpacked_assignments_are_followed_too(tmp_path):
    deps = _deps(tmp_path)
    assert "TABLE" in deps.get("LIMIT", set()), deps
    assert "TABLE" in deps.get("LO", set()) and "TABLE" in deps.get("HI", set()), deps


def test_an_edit_to_the_builder_reaches_every_reader(tmp_path):
    deps = _deps(tmp_path)
    changed = FunctionTracker.expand_changed_symbols_transitively({"build_table"}, deps)
    assert {"TABLE", "lookup", "LIMIT", "LO", "HI"} <= changed, changed
    assert "unrelated" not in changed, changed


def test_a_function_still_does_not_depend_on_itself(tmp_path):
    deps = _deps(tmp_path, "def f(n):\n    return f(n - 1) if n else 0\n")
    assert "f" not in deps.get("f", set()), deps
