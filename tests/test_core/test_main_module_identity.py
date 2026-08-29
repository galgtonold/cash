"""Running a script and importing it must share one cache.

A function defined in the script you ran belongs to module ``__main__``. So
``python model.py`` keyed it ``__main__.work`` while ``import model`` keyed the
same function, same source, same argument as ``model.work``: two entries, one
computation. The shape that hits it is ordinary -- develop behind an
``if __name__ == "__main__"`` block, run it while testing, then import it from
a driver and recompute everything you already had.

Two places had to agree, which is what made the first attempt only half work.
``Cash._get_func_key`` names the entry, and the purity analyzer's
``_qualname_of`` names each helper in ``helper_source_hashes``, which is folded
into the state hash as ``helper:{qual}:{digest}``. Normalising one left the
function name matching and the state hash not, so the entry still missed --
with byte-identical digests either side. ``test_the_state_hash_agrees_too``
is the arm that catches that specific half-fix.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import cash
from cash.utils import resolve_main_module


def _script(tmp_path, name, body):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _run(path, tmp_path):
    return subprocess.run([sys.executable, str(path)], capture_output=True,
                          text=True, cwd=str(tmp_path), encoding="utf-8",
                          errors="replace")


WORKER = """
    import cash
    c = cash.Cash(cache_dir="cache")

    @c.cache(assume_safe=True)
    def work(n):
        print("COMPUTED")
        import time
        time.sleep(0.3)          # clear the persist floor so it reaches disk
        return n + 1

    if __name__ == "__main__":
        work(1)
"""


def test_running_then_importing_reuses_the_result(tmp_path):
    """The headline, on real processes -- the only oracle that cannot be fooled."""
    _script(tmp_path, "worker", WORKER)
    driver = _script(tmp_path, "driver", """
        import worker
        worker.work(1)
    """)

    first = _run(tmp_path / "worker.py", tmp_path)
    assert first.returncode == 0, first.stderr
    assert "COMPUTED" in first.stdout, "the priming run should have computed"

    second = _run(driver, tmp_path)
    assert second.returncode == 0, second.stderr
    assert "COMPUTED" not in second.stdout, (
        "importing the script recomputed what running it had already cached"
    )


def test_the_state_hash_agrees_too(tmp_path):
    """Not just the function name -- the whole key.

    The first version of this fix normalised ``_get_func_key`` alone, so both
    runs agreed on ``worker.work`` and still missed, because the analyzer kept
    naming the helper ``__main__.work`` inside the state hash.
    """
    _script(tmp_path, "worker", WORKER)
    dump = _script(tmp_path, "dump", """
        import json, sys
        target = sys.argv[1]
        if target == "direct":
            import runpy
            ns = runpy.run_path("worker.py", run_name="__main__")
            fn = ns["work"].__wrapped__
            c = ns["c"]
        else:
            import worker
            fn = worker.work.__wrapped__
            c = worker.c
        name = c._get_func_key(fn)
        c._analyze_dependencies(fn)
        state = c._state_hasher.compute(name, own_source_override=c._pin_own_source(fn))
        print(json.dumps({"name": name, "state": state}))
    """)
    import json
    direct = json.loads(subprocess.run(
        [sys.executable, str(dump), "direct"], capture_output=True, text=True,
        cwd=str(tmp_path), encoding="utf-8").stdout.strip().splitlines()[-1])
    imported = json.loads(subprocess.run(
        [sys.executable, str(dump), "import"], capture_output=True, text=True,
        cwd=str(tmp_path), encoding="utf-8").stdout.strip().splitlines()[-1])

    assert direct["name"] == imported["name"], "the function name still disagrees"
    assert direct["state"] == imported["state"], (
        "the function name agrees but the state hash does not -- the analyzer's "
        "helper qualnames are not being normalised"
    )


def test_two_scripts_with_different_names_stay_apart(tmp_path):
    """The module qualifier still has to separate unrelated scripts."""
    for name, value in (("alpha", 10), ("beta", 999)):
        _script(tmp_path, name, f"""
            import cash
            c = cash.Cash(cache_dir="cache")

            @c.cache(assume_safe=True)
            def F(n):
                import time
                time.sleep(0.3)
                return n * {value}

            print(F(1))
        """)
    assert _run(tmp_path / "alpha.py", tmp_path).stdout.strip() == "10"
    assert _run(tmp_path / "beta.py", tmp_path).stdout.strip() == "999"


# ---------------------------------------------------------------------------
# The decorator's ordinary promises, for a function defined in __main__
# ---------------------------------------------------------------------------
#
# Renaming a function renames the key into the dependency graph, the purity
# reports, ``helper_resolution_paths``, ``source_hashes`` and
# ``_purity_modes``. The rest of the suite defines its functions inside test
# modules, so none of it exercises those lookups under ``__main__`` -- the
# whole suite passed while this path was unchecked.
#
# The sharpest risk is ``helper_resolution_paths``, now KEYED by the
# normalised name while its module VALUE stays ``__main__`` because it is a
# ``sys.modules`` lookup. A mismatch there fails re-resolution quietly, and an
# edited helper stops invalidating -- a stale answer, not a miss. Hence
# ``[a helper edited]``, and hence every arm asserting the VALUE and not just
# whether it ran.

PROGRAM = """
    import time
    import cash

    c = cash.Cash(cache_dir="cache")

    THRESHOLD = {threshold}

    def helper(n):
        return n + {helper_const}

    @c.cache(assume_safe=True)
    def work(n):{extra}
        time.sleep(0.3)
        print("COMPUTED")
        return helper(n) + THRESHOLD + {body_const}

    if __name__ == "__main__":
        print("RESULT", work(1))
"""


def _program(tmp_path, **kw):
    params = {"threshold": 0, "helper_const": 1, "body_const": 0, "extra": ""}
    params.update(kw)
    _script(tmp_path, "prog", PROGRAM.format(**params))
    done = _run(tmp_path / "prog.py", tmp_path)
    assert done.returncode == 0, done.stderr
    value = next(line.split()[1] for line in done.stdout.splitlines()
                 if line.startswith("RESULT"))
    return "COMPUTED" in done.stdout, value


# work(1) == helper(1) + THRESHOLD + body_const == (1 + helper_const) + ...
@pytest.mark.parametrize(("label", "second", "recomputes", "value"), [
    ("nothing changed", {}, False, "2"),
    ("a comment added", {"extra": "\n        # a note"}, False, "2"),
    ("the body edited", {"body_const": 5}, True, "7"),
    ("a helper edited", {"helper_const": 9}, True, "10"),
    ("a read global changed", {"threshold": 7}, True, "9"),
])
def test_the_decorator_keeps_its_promises_under_main(
        tmp_path, label, second, recomputes, value):
    ran, _ = _program(tmp_path)
    assert ran, "the priming run should have computed"

    ran, got = _program(tmp_path, **second)
    assert ran is recomputes, (
        f"{label}: {'recomputed' if ran else 'restored'}, wanted "
        f"{'recompute' if recomputes else 'restore'}"
    )
    assert got == value, f"{label}: wrong answer ({got}, wanted {value})"


# ---------------------------------------------------------------------------
# The resolver itself
# ---------------------------------------------------------------------------


def test_a_function_from_a_real_module_is_untouched():
    assert resolve_main_module(test_a_function_from_a_real_module_is_untouched) \
        != "__main__"


def test_no_file_falls_back_to_main():
    """A REPL, ``python -c``, a frozen app, a Jupyter kernel.

    There is no import for those to agree with, so inventing a name would be
    worse than leaving them as they are.
    """
    namespace: dict = {}
    exec("def f(): pass", namespace)          # noqa: S102 - the case under test
    assert resolve_main_module(namespace["f"]) == "__main__"


def test_it_reads_the_function_s_own_globals_not_the_entry_point():
    """Under ``runpy``/``exec`` the entry point and the defining file differ.

    Taking ``sys.modules['__main__'].__file__`` names the function after a file
    it was not defined in; the function's own globals are always right.
    """
    namespace = {"__file__": "/somewhere/else/defining_file.py"}
    exec("def f(): pass", namespace)          # noqa: S102 - the case under test
    assert resolve_main_module(namespace["f"]) == "defining_file"


def test_a_notebook_style_namespace_stays_main():
    """Jupyter's ``__main__`` is a user namespace, not a file."""
    c = cash.Cash(cache_dir=None)
    namespace: dict = {"__name__": "__main__"}
    exec("def f(n): return n", namespace)     # noqa: S102 - the case under test
    assert c._get_func_key(namespace["f"]).startswith("__main__.")
