"""CAS-260 at the call unit: identifying, keying on, and restoring the globals
a cached callee writes.

The statement path handles ``x = compute(y)``; this covers the half CAS-243
widened the bug into — a call whose enclosing statement is skip-cached
(``sink.append(compute(y))``), where the intercepted call is the only thing
serving a hit and therefore the only thing that can put the write back.
"""
import types

import pytest

from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import CallSite
from cash.notebook.call_unit import (
    _UNWRAP_FAILED,
    CallUnit,
    _unwrap_callee_globals,
    call_cache_key,
    callee_mutated_globals,
)


def _make(source, name, extra_globals=None):
    """Compile *source* in a fresh globals dict and return the function.

    A real dict rather than a mock: ``callee_mutated_globals`` filters against
    ``fn.__globals__`` for real, and ``inspect.getsource`` has to be able to
    read the body -- so this writes through ``exec`` on a named temp module
    only where the test needs source, and otherwise relies on the module-level
    helpers below.
    """
    ns = dict(extra_globals or {})
    exec(compile(source, __file__, "exec"), ns)
    return ns[name], ns


CALLS: list = []
COUNTER: dict = {"n": 0}
AAA: list = []
ZZZ: list = []
SOME_MODULE = types


def appends_to_two_globals(v):
    ZZZ.append(v)
    AAA.append(v)
    return v


def sets_an_attribute_on_a_module(v):
    SOME_MODULE.probe_attr = v
    return v


def appends_to_a_global(v):
    CALLS.append(v)
    return v * 10


def increments_a_global_subscript(v):
    COUNTER["n"] += 1
    return COUNTER["n"]


def mutates_its_argument(d):
    d["k"] = 1
    return len(d)


def mutates_a_fresh_local(v):
    acc = []
    acc.append(v)
    return acc


def is_pure(v):
    return v * 10


class TestIdentification:
    def test_a_write_method_on_a_global_is_named(self):
        assert callee_mutated_globals(appends_to_a_global) == ("CALLS",)

    def test_a_subscript_write_on_a_global_is_named(self):
        assert callee_mutated_globals(increments_a_global_subscript) == ("COUNTER",)

    def test_an_argument_mutation_is_not_a_global(self):
        # Owned by `_hash_args` / `function_arg_mutations`. Claiming it here
        # would double-count it and capture the caller's object as if it were
        # module state.
        assert callee_mutated_globals(mutates_its_argument) == ()

    def test_a_fresh_local_is_not_a_global(self):
        assert callee_mutated_globals(mutates_a_fresh_local) == ()

    def test_a_pure_callee_names_nothing(self):
        assert callee_mutated_globals(is_pure) == ()

    def test_redefining_the_callee_is_not_served_the_old_verdict(self):
        """The property the code-object memo rests on.

        Editing a cell rebuilds the function, and a rebuilt function has a NEW
        code object -- so a verdict cached under the old one can never be
        returned for the new body. Written against a real file so
        ``inspect.getsource`` can read both versions, since that is the path
        the memo is skipping.
        """
        import importlib
        import sys
        import textwrap
        from pathlib import Path

        tmp = Path(__file__).parent / "_cas260_redef_probe.py"
        try:
            tmp.write_text(textwrap.dedent("""
                MARK = []
                def edited(v):
                    MARK.append(v)
                    return v
            """), encoding="utf-8")
            sys.path.insert(0, str(tmp.parent))
            mod = importlib.import_module("_cas260_redef_probe")
            assert callee_mutated_globals(mod.edited) == ("MARK",)

            tmp.write_text(textwrap.dedent("""
                MARK = []
                def edited(v):
                    return v
            """), encoding="utf-8")
            importlib.invalidate_caches()
            import linecache
            linecache.clearcache()
            mod = importlib.reload(mod)
            assert callee_mutated_globals(mod.edited) == (), (
                "the memo served the pre-edit verdict for a rebuilt function"
            )
        finally:
            sys.modules.pop("_cas260_redef_probe", None)
            if str(tmp.parent) in sys.path:
                sys.path.remove(str(tmp.parent))
            tmp.unlink(missing_ok=True)

    def test_a_sourceless_callee_is_silent_rather_than_raising(self):
        # A C builtin has no readable source. Fail OPEN: a callee cash cannot
        # read is not made safer by refusing to cache it, and this must never
        # be why user code breaks.
        assert callee_mutated_globals(len) == ()
        assert callee_mutated_globals(object()) == ()

    def test_the_result_is_sorted(self):
        """Source order is ZZZ then AAA. The key component built from this is a
        join, so an unstable order would mint two keys for one call."""
        assert callee_mutated_globals(appends_to_two_globals) == ("AAA", "ZZZ")

    def test_a_module_valued_name_is_never_captured(self):
        """Mirrors ``_classify_method_mutations`` and
        ``_function_arg_mutation_receivers``: a module is never a value to
        serialise. The name IS found by the analysis (an attribute write on a
        free variable) and dropped by the live-namespace filter, so this is a
        real exclusion rather than an unreadable-source no-op."""
        assert callee_mutated_globals(sets_an_attribute_on_a_module) == ()


class TestKeying:
    """The pre-call state must pin the key, or a restored absolute post-state
    lands over a prefix that never produced it."""

    def _site(self):
        return CallSite(
            source="compute(x)",
            free_names=frozenset(("compute", "x")),
            occurrence_index=0,
            computed_arg_positions=(),
            stmt_identity="",
        )

    def _key(self, global_digests):
        return call_cache_key(
            self._site(),
            ctx=CacheKeyContext(variable_lineage={"x": "aaa"}, user_ns={"x": 1, "compute": len}),
            arg_digests=[],
            loop_vars={},
            global_digests=global_digests,
        )

    def test_a_different_pre_state_gives_a_different_key(self):
        assert self._key({"CALLS": "empty"}) != self._key({"CALLS": "one-item"})

    def test_the_same_pre_state_gives_the_same_key(self):
        assert self._key({"CALLS": "empty"}) == self._key({"CALLS": "empty"})

    def test_no_globals_keys_exactly_as_before_the_parameter_existed(self):
        # An entry written by an earlier version must still be found. Both
        # spellings of "nothing to say" have to agree with the old key.
        assert self._key(None) == self._key({})

    def test_a_global_cannot_collide_with_a_loop_variable_slot(self):
        """Both components go through the same ``|``-join, so a global named
        like a loop var must not be able to occupy that var's slot."""
        site = self._site()
        ctx = CacheKeyContext(variable_lineage={"x": "aaa"}, user_ns={"x": 1, "compute": len})
        as_global = call_cache_key(
            site, ctx=ctx, arg_digests=[], loop_vars={}, global_digests={"t": "d"},
        )
        as_loop_var = call_cache_key(
            site, ctx=ctx, arg_digests=[], loop_vars={"t": object()},
            loop_var_digests={"t": "d"},
        )
        assert as_global != as_loop_var


class TestCaptureAndRestore:
    def test_capture_returns_the_post_call_values(self):
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": [1, 2]})
        assert CallUnit._capture_globals(fn, ("CALLS",)) == {"CALLS": [1, 2]}

    def test_capture_of_nothing_is_an_empty_dict_not_a_refusal(self):
        # `None` means refuse; `{}` means "nothing to capture, store normally".
        # Conflating them would stop every ordinary call being cached.
        assert CallUnit._capture_globals(is_pure, ()) == {}

    def test_capture_refuses_a_name_that_vanished(self):
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": []})
        del ns["CALLS"]
        assert CallUnit._capture_globals(fn, ("CALLS",)) is None

    def test_capture_refuses_an_unpicklable_value(self):
        """``compute_hash`` falls back to ``sha256(id(obj))`` for anything that
        will not pickle. ``id`` is invariant across an in-place mutation, so a
        later pre-state comparison on such a name is blind -- always, for the
        whole class. Storing an entry whose key cannot discriminate is the
        partial-accumulator hazard, so this fails closed."""
        import threading
        fn, _ = _make("def f():\n    pass\n", "f", {"LOCK": threading.Lock()})
        assert CallUnit._capture_globals(fn, ("LOCK",)) is None

    def test_restore_writes_the_recorded_value_back(self):
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": []})
        unit = CallUnit.__new__(CallUnit)
        unit._restore_globals(fn, ("CALLS",), {"CALLS": [1, 2, 3]})
        assert ns["CALLS"] == [1, 2, 3]

    def test_restore_ignores_a_name_the_current_callee_no_longer_writes(self):
        """The entry can outlive an edit to the callee. Honouring a stale name
        would resurrect a variable the current source never mentions."""
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": []})
        unit = CallUnit.__new__(CallUnit)
        unit._restore_globals(fn, (), {"GONE": [9]})
        assert "GONE" not in ns

    def test_capture_snapshots_rather_than_referencing(self):
        """The captured value must not be a live handle on the object.

        This capture exists to record a POST-CALL state, and the object it
        records is by construction one that gets mutated in place -- so keeping
        a reference records nothing. The RAM tier stores metadata as given, so
        a later call mutating the same object silently rewrites an
        already-stored entry. Measured before this was fixed::

            cell 3   a = next_seq()            stores N -> [1]
            cell 5   seen.append(next_seq())   mutates the SAME list to [2]
            rerun    cell 3 hits, restores N -> [2]   (not [1])

        which re-keyed cell 5's call against a pre-state that never existed, so
        it missed forever and the two spellings diverged -- caught by
        ``test_impure_callee_is_spelling_consistent``, a whole integration gate
        later.
        """
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": [1]})
        captured = CallUnit._capture_globals(fn, ("CALLS",))
        ns["CALLS"].append(2)
        assert captured == {"CALLS": [1]}, (
            "the capture aliases the live object, so a later mutation rewrites "
            "an entry that was already stored"
        )

    def test_restore_hands_back_a_copy(self):
        """The mirror: restoring the stored object itself would make the live
        variable and the cache entry the same object, so the next call would
        rewrite the entry it was just served from."""
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": []})
        entry = {"CALLS": [1]}
        unit = CallUnit.__new__(CallUnit)
        unit._restore_globals(fn, ("CALLS",), entry)
        ns["CALLS"].append(2)
        assert entry["CALLS"] == [1], (
            "mutating the restored variable reached back into the cache entry"
        )

    def test_capture_refuses_a_value_that_cannot_be_copied(self):
        class NoCopy:
            def __deepcopy__(self, memo):
                raise RuntimeError("nope")

        fn, _ = _make("def f():\n    pass\n", "f", {"ODD": NoCopy()})
        assert CallUnit._capture_globals(fn, ("ODD",)) is None

    def test_restore_without_a_recorded_entry_is_a_no_op(self):
        fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": [7]})
        unit = CallUnit.__new__(CallUnit)
        unit._restore_globals(fn, ("CALLS",), None)
        assert ns["CALLS"] == [7]


def test_digests_use_the_full_hash_not_the_sampled_one():
    """Two long lists agreeing on head and tail must key DIFFERENTLY.

    ``compute_hash`` reduces a collection over 200 elements to its first and
    last five. An accumulator is exactly the shape that grows in the middle, so
    a sampled digest would collapse two genuinely different pre-states onto one
    key and serve the first's end state for the second -- wrong on the first
    run, no pre-existing cache required.
    """
    a = list(range(300))
    b = list(range(300))
    b[150] = -1
    fn, ns = _make("def f():\n    pass\n", "f", {"ACC": a})
    first = CallUnit._global_digests(fn, ("ACC",))
    ns["ACC"] = b
    second = CallUnit._global_digests(fn, ("ACC",))
    assert first != second, (
        "the pre-state digest is sampled; two accumulators differing only in "
        "the middle share a key"
    )


@pytest.mark.parametrize("bad", [None, 42, "text"])
def test_restore_tolerates_a_malformed_entry(bad):
    fn, ns = _make("def f():\n    pass\n", "f", {"CALLS": [7]})
    unit = CallUnit.__new__(CallUnit)
    unit._restore_globals(fn, ("CALLS",), bad)
    assert ns["CALLS"] == [7]


class TestPayloadLivesOnTheValueNotInMetadata:
    """CAS-260: the captured globals ride on the VALUE, with only a plain bool
    in metadata.

    Metadata is unpickled for every entry in the directory the first time
    eviction runs (``FileBackend._ensure_metadata_loaded``) and held for the
    process lifetime, so a user
    object there is deserialised whether or not it is used, and kept forever. A
    survey of 5859 real metadata files found 31 of 33 fields are plain
    builtins; this must not become the third exception.
    """

    def test_an_ordinary_entry_is_unwrapped_untouched(self):
        assert _unwrap_callee_globals(42, {}) == (42, None)

    def test_a_wrapped_entry_splits_into_result_and_globals(self):
        result, globs = _unwrap_callee_globals((42, {"CALLS": [1]}),
                                               {"has_callee_globals": True})
        assert result == 42
        assert globs == {"CALLS": [1]}

    def test_a_cached_value_that_is_itself_a_2_tuple_is_not_mistaken_for_a_wrapper(self):
        """Why the flag exists rather than sniffing the value's shape: a call
        legitimately returning ``(x, {...})`` is indistinguishable from a
        wrapped entry by type alone."""
        value = (42, {"CALLS": [1]})
        assert _unwrap_callee_globals(value, {}) == (value, None)

    def test_a_flagged_entry_with_the_wrong_shape_is_refused(self):
        """Corrupt or hand-edited. Handing the tuple back as the result would
        be a silently wrong value; a miss merely costs a recompute."""
        assert _unwrap_callee_globals(42, {"has_callee_globals": True})[0] is _UNWRAP_FAILED
        assert _unwrap_callee_globals((1, 2, 3), {"has_callee_globals": True})[0] is _UNWRAP_FAILED
        assert _unwrap_callee_globals((1, "no"), {"has_callee_globals": True})[0] is _UNWRAP_FAILED

    def test_none_is_a_legitimate_cached_value_not_a_failure(self):
        """``_UNWRAP_FAILED`` is a unique sentinel precisely so a stored
        ``None`` stays distinguishable from a broken entry."""
        assert _unwrap_callee_globals(None, {}) == (None, None)
        result, globs = _unwrap_callee_globals((None, {"C": [1]}),
                                               {"has_callee_globals": True})
        assert result is None and globs == {"C": [1]}
