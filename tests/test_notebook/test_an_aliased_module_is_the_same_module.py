"""``import lib as x`` is an import of ``lib``, in both places that forgot it.

Round 27, r27s2: a fix to a function in a project module had no effect in a
live kernel. The cell re-ran and returned the PRE-fix answer, and two exported
deliverables were computed from it. Only a kernel restart helped. Reproduced
3/3; three minimal repros written at the time did NOT reproduce, because all
three wrote ``import mylib`` -- and so does every module-reload test in this
suite. The tester's notebook wrote ``import tickets_lib as tl``.

Two independent places keyed on the name the cell bound rather than on the
module, and BOTH have to be fixed: with either one alone the integration twin
still fails.

1. :func:`~cash.notebook.lineage_formula.module_source_component` tested
   ``var_name in function_tracker._tracked_modules``, a set of real module
   names. For an alias that is false, so the module's source digest never
   entered the bound name's lineage and every statement reading it keyed
   identically across an edit.
2. :meth:`ModuleInvalidator._update_module_lineages` wrote the fresh lineage
   to ``variable_lineage[mod_name]`` only, leaving the alias holding the
   pre-edit hash -- which is the entry cache keys and downstream propagation
   actually read.

The integration twin is
``tests/test_notebook_integration/test_an_aliased_module_is_the_same_module.py``.
"""
import sys
import types

import pytest

from cash.notebook.lineage_formula import module_source_component


class _Tracker:
    """The two attributes ``module_source_component`` reads off a tracker."""

    def __init__(self, tracked, dep_parents=None):
        self._tracked_modules = set(tracked)
        self._dep_file_to_parents = dep_parents or {}


@pytest.fixture
def local_module(tmp_path):
    """A real, importable module on disk, and a way to rewrite it."""
    path = tmp_path / "aliased_probe_lib.py"

    def write(marker):
        path.write_text("VALUE = '%s'\n" % marker, encoding="utf-8")

    write("one")
    sys.path.insert(0, str(tmp_path))
    try:
        module = types.ModuleType("aliased_probe_lib")
        module.__file__ = str(path)
        yield module, write
    finally:
        sys.path.remove(str(tmp_path))


class TestTheLineageComponent:
    """Half one: the module's source has to reach the bound name's lineage."""

    def test_an_aliased_module_contributes_its_source(self, local_module):
        module, _write = local_module
        tracker = _Tracker({"aliased_probe_lib"})

        component = module_source_component(tracker, module, "apl", "import aliased_probe_lib as apl")

        assert component.startswith(":mod_src:"), (
            "the module is tracked and the cell bound it to 'apl'; returning "
            "nothing here is what let an edit to it change no key: %r"
            % (component,)
        )

    def test_and_that_source_moves_when_the_file_does(self, local_module):
        module, write = local_module
        tracker = _Tracker({"aliased_probe_lib"})
        args = (tracker, module, "apl", "import aliased_probe_lib as apl")

        before = module_source_component(*args)
        write("two")
        after = module_source_component(*args)

        assert before != after, (
            "same digest before and after the edit, so every statement "
            "reading the alias keys identically: %r" % (before,)
        )

    def test_an_unaliased_import_is_unchanged(self, local_module):
        """The control: the common spelling must key exactly as it did."""
        module, _write = local_module
        tracker = _Tracker({"aliased_probe_lib"})

        component = module_source_component(
            tracker, module, "aliased_probe_lib", "import aliased_probe_lib")

        assert component.startswith(":mod_src:"), component

    def test_a_module_nobody_tracks_still_contributes_nothing(self, local_module):
        """The control that keeps site-packages out of every key."""
        module, _write = local_module
        tracker = _Tracker(set())

        assert module_source_component(
            tracker, module, "apl", "import aliased_probe_lib as apl") == ""

    def test_a_dependency_file_is_found_under_either_name(self, local_module):
        """A tracked module's own dependencies must not go missing either."""
        module, _write = local_module
        dep = str(module.__file__)
        tracker = _Tracker({"aliased_probe_lib"}, {dep: {"aliased_probe_lib"}})

        assert module_source_component(
            tracker, module, "apl", "import aliased_probe_lib as apl"
        ).startswith(":mod_src:")


class TestTheInvalidator:
    """Half two: the alias is the entry cache keys actually read."""

    @staticmethod
    def _invalidator(user_ns):
        from cash.notebook.module_invalidator import ModuleInvalidator

        shell = types.SimpleNamespace(user_ns=user_ns)
        return ModuleInvalidator(shell)

    def test_every_name_bound_to_the_module_is_returned(self, local_module):
        module, _write = local_module
        sys.modules["aliased_probe_lib"] = module
        try:
            inv = self._invalidator({"apl": module, "other": 1})
            assert sorted(inv._names_bound_to("aliased_probe_lib")) == [
                "aliased_probe_lib", "apl",
            ]
        finally:
            del sys.modules["aliased_probe_lib"]

    def test_a_name_holding_a_different_module_is_left_alone(self, local_module):
        """Identity, not name: a same-named import of something else is not it."""
        module, _write = local_module
        impostor = types.ModuleType("aliased_probe_lib")
        sys.modules["aliased_probe_lib"] = module
        try:
            inv = self._invalidator({"apl": module, "fake": impostor})
            assert "fake" not in inv._names_bound_to("aliased_probe_lib")
        finally:
            del sys.modules["aliased_probe_lib"]

    def test_a_module_not_in_sys_modules_still_gets_its_own_name(self):
        """Degrade to today's behaviour rather than dropping the update."""
        inv = self._invalidator({})
        assert inv._names_bound_to("never_imported_anywhere") == [
            "never_imported_anywhere",
        ]
