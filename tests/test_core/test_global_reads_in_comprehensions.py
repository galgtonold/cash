"""CAS-128: globals read inside a nested scope must fold into the cache key.

A generator expression, comprehension, or ``lambda`` compiles to its OWN code
object, so a global referenced only in there never appeared in the outer
function's ``co_names`` — and ``_read_global_data_names`` collected only that.
Editing such a global served STALE results silently.

The controls matter as much as the repros: the nested ``STORE_GLOBAL`` of a
walrus accumulator lives in the genexp's own code object, so the write
exclusion had to learn the same recursion or a counter mutated inside a
comprehension would be folded and miss on every call.

Note on versions: CPython 3.12+ inlines list/set/dict comprehensions into the
enclosing scope (PEP 709), so those names already reached ``co_names`` there;
generator expressions and lambdas get their own scope on every version. The
comprehension cases below are kept as regression cover regardless of which
side of that inlining the interpreter falls.
"""
import importlib
import os
import sys

import pytest

from cash import Cash


def _write_module(dirpath, name, body):
    path = os.path.join(dirpath, f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


@pytest.fixture
def make_module(tmp_path):
    """Import a throwaway module from *tmp_path*, cleaning sys.path/modules."""
    created = []
    sys.path.insert(0, str(tmp_path))

    def _make(name, body):
        _write_module(str(tmp_path), name, body)
        mod = importlib.import_module(name)
        created.append(name)
        return mod

    try:
        yield _make
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        for name in created:
            sys.modules.pop(name, None)


class TestGlobalReadInNestedScope:
    """Editing a global read only inside a nested scope must invalidate."""

    @pytest.mark.parametrize(
        "name,body,arg",
        [
            (
                "genexp",
                "def f(v):\n    return sum(x > THRESHOLD for x in v)\n",
                (1, 5, 20, 30),
            ),
            (
                "listcomp",
                "def f(v):\n    return len([x for x in v if x > THRESHOLD])\n",
                (1, 5, 20, 30),
            ),
            (
                "dictcomp",
                "def f(v):\n    return {x: x > THRESHOLD for x in v}\n",
                (1, 20),
            ),
            (
                "setcomp",
                "def f(v):\n    return frozenset({x > THRESHOLD for x in v})\n",
                (1, 20),
            ),
            (
                "lambda",
                "def f(v):\n"
                "    is_big = lambda x: x > THRESHOLD\n"
                "    return sum(map(is_big, v))\n",
                (1, 5, 20, 30),
            ),
        ],
    )
    def test_global_read_in_nested_scope_invalidates(self, make_module, name, body, arg):
        mod = make_module(f"cas128_{name}", f"THRESHOLD = 10\n{body}")
        c = Cash()
        f = c.cache(mod.f)

        f(arg)
        assert f.explain(arg).reason == "hit"

        mod.THRESHOLD = 1000
        assert f.explain(arg).reason != "hit", (
            f"THRESHOLD read inside a {name} did not invalidate — stale result served"
        )

    def test_nested_genexp_two_levels_deep_invalidates(self, make_module):
        # Two levels of nesting: the global lives in the INNER genexp's code
        # object, reachable only by recursing through the outer one.
        mod = make_module(
            "cas128_nested",
            "THRESHOLD = 10\n"
            "def f(rows):\n"
            "    return sum(sum(y > THRESHOLD for y in row) for row in rows)\n",
        )
        c = Cash()
        f = c.cache(mod.f)
        arg = ((1, 20), (30, 2))

        f(arg)
        assert f.explain(arg).reason == "hit"

        mod.THRESHOLD = 1000
        assert f.explain(arg).reason != "hit"

    def test_listcomp_of_genexp_invalidates(self, make_module):
        # Mixed nesting: an inlined (3.12+) comprehension wrapping a genexp
        # that owns its own scope.
        mod = make_module(
            "cas128_mixed",
            "THRESHOLD = 10\n"
            "def f(rows):\n"
            "    return [sum(y > THRESHOLD for y in row) for row in rows]\n",
        )
        c = Cash()
        f = c.cache(mod.f)
        arg = ((1, 20), (30, 2))

        f(arg)
        assert f.explain(arg).reason == "hit"

        mod.THRESHOLD = 1000
        assert f.explain(arg).reason != "hit"


class TestNoOverInvalidation:
    """The write/mutate exclusions must survive the recursion (CAS-104 lesson)."""

    def test_global_mutated_in_comprehension_still_hits(self, make_module):
        # ACC.append(...) inside a genexp: an in-place accumulator. Folding it
        # would drift the key every call -> permanent miss. The AST-based
        # exclusion already sees comprehension bodies; assert it still does.
        mod = make_module(
            "cas128_mutacc",
            "ACC = []\n"
            "def f(v):\n"
            "    return len(list(ACC.append(x) for x in v))\n",
        )
        c = Cash()
        f = c.cache(mod.f)

        f((1, 2))
        assert f.explain((1, 2)).reason == "hit"
        f((1, 2))
        assert f.explain((1, 2)).reason == "hit", "mutated global folded -> permanent miss"

    def test_global_rebound_by_walrus_in_genexp_still_hits(self, make_module):
        # The STORE_GLOBAL for COUNTER lives in the GENEXP's code object, not
        # the outer one. Recursing reads without recursing writes would fold a
        # drifting counter and miss forever. This is the regression that pins
        # the two channels together.
        mod = make_module(
            "cas128_walrus",
            "COUNTER = 0\n"
            "def f(v):\n"
            "    global COUNTER\n"
            "    return len(list((COUNTER := COUNTER + x) for x in v))\n",
        )
        c = Cash()
        f = c.cache(mod.f)

        f((1, 2))
        assert f.explain((1, 2)).reason == "hit"
        f((1, 2))
        assert f.explain((1, 2)).reason == "hit", (
            "walrus-rebound global folded -> permanent miss"
        )

    def test_function_reading_no_globals_has_stable_key(self, make_module):
        # No key churn: a genexp that touches no global must produce a
        # byte-identical cache key before and after the fix's extra walk.
        mod = make_module(
            "cas128_noglobals",
            "def f(v):\n    return sum(x * 2 for x in v)\n",
        )
        c = Cash()
        f = c.cache(mod.f)

        assert c._read_global_data_names(mod.f) == ()
        f((1, 2, 3))
        key_before = f.explain((1, 2, 3)).cache_key
        assert f.explain((1, 2, 3)).reason == "hit"
        key_after = f.explain((1, 2, 3)).cache_key
        assert key_before == key_after


class TestReadGlobalNamesDetection:
    """Unit-level cover on the detector itself."""

    def test_nested_scope_names_are_collected(self, make_module):
        mod = make_module(
            "cas128_detect",
            "THRESHOLD = 10\n"
            "OTHER = 3\n"
            "def f(v):\n"
            "    return sum(x > THRESHOLD for x in v) + OTHER\n",
        )
        c = Cash()
        assert c._read_global_data_names(mod.f) == ("OTHER", "THRESHOLD")

    def test_detection_is_memoized_per_code_object(self, make_module):
        mod = make_module(
            "cas128_memo",
            "THRESHOLD = 10\ndef f(v):\n    return sum(x > THRESHOLD for x in v)\n",
        )
        c = Cash()
        first = c._read_global_data_names(mod.f)
        assert mod.f.__code__ in c._global_read_cache
        assert c._read_global_data_names(mod.f) is first
