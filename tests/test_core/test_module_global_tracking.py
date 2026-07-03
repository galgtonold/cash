"""CAS-107: mutable module-level state read by a cached function must track.

A cached function reading a module data global (a config constant) or dispatching
through a dict of callables returned stale results with no warning when that
global changed. These assert the read globals fold into the cache key — while a
global the function WRITES (an accumulator) stays excluded so it doesn't cause a
miss on every call.
"""
import os
import sys
import tempfile

from cash import Cash


def _write(dirpath, name, body):
    path = os.path.join(dirpath, f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


class TestModuleGlobalTracking:
    def test_data_global_change_invalidates(self, tmp_path):
        sys.path.insert(0, str(tmp_path))
        try:
            _write(str(tmp_path), "cas107t1",
                   "CONST = 5\ndef times_const(x):\n    return x * CONST\n")
            import importlib
            mod = importlib.import_module("cas107t1")
            c = Cash()
            f = c.cache(mod.times_const)
            assert f(2) == 10
            assert f.explain(2).reason == "hit"
            mod.CONST = 7
            assert f.explain(2).reason != "hit"     # config change seen
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("cas107t1", None)

    def test_dict_dispatch_swap_invalidates(self, tmp_path):
        sys.path.insert(0, str(tmp_path))
        try:
            _write(str(tmp_path), "cas107t2",
                   "OPS = {'double': (lambda x: x * 2)}\n"
                   "def apply_op(x):\n    return OPS['double'](x)\n")
            import importlib
            mod = importlib.import_module("cas107t2")
            c = Cash()
            f = c.cache(mod.apply_op)
            assert f(3) == 6
            assert f.explain(3).reason == "hit"
            mod.OPS["double"] = lambda x: x * 5
            assert f.explain(3).reason != "hit"     # dispatch swap seen
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("cas107t2", None)

    def test_rebound_global_accumulator_still_caches(self):
        # A function that REBINDS a module global (STORE_GLOBAL counter) must NOT
        # fold it, or every call would miss. The CAS-104 lesson for globals.
        assert _rebind_counter(1) == 1
        assert _rebind_counter.explain(1).reason == "hit"   # same arg -> hit

    def test_inplace_mutated_global_still_caches(self):
        # A function that mutates a global container IN PLACE (no STORE_GLOBAL)
        # must also be excluded from folding.
        assert _inplace_counter(1) == 1
        assert _inplace_counter.explain(1).reason == "hit"

    def test_pure_function_reading_no_globals_caches(self):
        c = Cash()

        @c.cache
        def pure(x):
            return x + 1

        assert pure(1) == 2
        assert pure.explain(1).reason == "hit"


# Module-level accumulator fixtures for the written-global tests.
_n = 0
_runs = {"n": 0}
_C = Cash()


@_C.cache
def _rebind_counter(x):
    global _n
    _n += 1             # STORE_GLOBAL -> excluded from folding
    return x


@_C.cache
def _inplace_counter(x):
    _runs["n"] += 1     # in-place mutation of a global -> excluded from folding
    return x
