"""CAS-110: ``depends_on=[plain_function]`` must contribute to the cache key.

The docstring promises declared function/DataSource deps fold into the key. For
a non-decorated function dependency, only a graph edge was added and the state
hasher contributed the empty string, so the declared dep was inert. These assert
that editing a declared plain-callable dep on disk (+ reload) invalidates.
"""
import importlib
import os
import sys
import tempfile
import time
import warnings

from cash import Cash


def _write_mod(dirpath, name, body):
    path = os.path.join(dirpath, f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


class TestDependsOnPlainFunction:
    def test_plain_function_dep_edit_invalidates(self, tmp_path):
        sys.path.insert(0, str(tmp_path))
        try:
            _write_mod(str(tmp_path), "cas110mod", "def g_proxy(x):\n    return x + 10\n")
            mod = importlib.import_module("cas110mod")

            c = Cash()

            def standalone(x):
                return x * 2

            f = c.cache(standalone, depends_on=[mod.g_proxy])
            assert f(2) == 4
            assert f.explain(2).reason == "hit"          # warm

            time.sleep(0.02)
            _write_mod(str(tmp_path), "cas110mod", "def g_proxy(x):\n    return x + 999\n")
            importlib.reload(mod)

            # The declared proxy dep changed on disk → key must change → not a hit.
            assert f.explain(2).reason != "hit"
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("cas110mod", None)

    def test_unchanged_dep_still_hits(self, tmp_path):
        sys.path.insert(0, str(tmp_path))
        try:
            _write_mod(str(tmp_path), "cas110mod2", "def g_proxy(x):\n    return x + 1\n")
            mod = importlib.import_module("cas110mod2")
            c = Cash()

            def standalone(x):
                return x * 2

            f = c.cache(standalone, depends_on=[mod.g_proxy])
            f(5)
            assert f.explain(5).reason == "hit"          # no edit → still hit
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("cas110mod2", None)

    def test_builtin_dep_does_not_crash(self):
        # A builtin has no readable source; it must not raise and must still
        # produce a stable key (coarse but never silently wrong).
        c = Cash()

        def s(x):
            return x

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = c.cache(s, depends_on=[len])
        assert f(3) == 3
        assert f.explain(3).reason == "hit"
