"""``import tickets_lib as tl`` records which attributes a statement reads.

``_update_module_attribute_deps`` records, per variable, the attributes it
read from each tracked module, so that an edit to one function in the module
invalidates only the variables that read it. It checked the NAME the
statement read against ``_tracked_modules``, which holds real module names --
the same blind spot 9785293 fixed in the module's lineage. For
``tl.load(...)`` nothing was recorded and the invalidator fell back to
invalidating the variable on any edit to the module: safe, but every aliased
reader lost the granularity a plain ``import tickets_lib`` kept.

Recorded under the alias, because that is the name the invalidator looks it
up by (``_decide_symbol_action``'s ``input_var``).
"""
import types
from types import SimpleNamespace

from cash.notebook.function_tracker import FunctionTracker
from cash.notebook.statement.lineage import StatementLineageBuilder


def _record(code, user_ns, tracked):
    ft = FunctionTracker()
    ft._tracked_modules = set(tracked)
    state = SimpleNamespace(module_attribute_deps={})
    builder = SimpleNamespace(function_tracker=ft)
    StatementLineageBuilder._update_module_attribute_deps(builder, state, "corpus", code, user_ns)
    return state.module_attribute_deps


def test_an_aliased_module_records_the_attributes_read():
    mod = types.ModuleType("tickets_lib")
    deps = _record("corpus = tl.load_corpus(ROOT)", {"tl": mod}, {"tickets_lib"})
    assert deps == {"corpus": {"tl": {"load_corpus"}}}, deps


def test_a_plain_import_still_does():
    mod = types.ModuleType("tickets_lib")
    deps = _record("corpus = tickets_lib.load_corpus(ROOT)", {"tickets_lib": mod}, {"tickets_lib"})
    assert deps == {"corpus": {"tickets_lib": {"load_corpus"}}}, deps


def test_an_untracked_module_records_nothing():
    mod = types.ModuleType("numpy")
    deps = _record("corpus = np.load(ROOT)", {"np": mod}, {"tickets_lib"})
    assert deps == {}, deps
