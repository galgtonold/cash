"""A user module imported only inside a cached function is not run by cash.

To key the helper such an import binds, the analysis imported the module on
the first call, before the body ran. That ran the module's code even where
the body never imports it (a branch not taken), so a module that writes a
file or sets state at import did so in a run where plain Python never
loaded it. The module is now found and read without running it: its source
and that of the user modules it imports stand for what it binds. A module
whose top level computes a value from outside its source (the environment,
a file) cannot be vouched for that way: the call runs with a warning, and is
not served from the cache.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


def _run(proj, env=None):
    p = run_python("job.py", cwd=proj, env=env)
    return p.stdout.strip(), "[RUN]" in p.stderr, p.stderr


def _write(proj, name, text):
    path = proj / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def test_a_module_the_body_does_not_reach_is_not_imported(tmp_path):
    _write(
        tmp_path,
        "helpmod.py",
        """
        import os
        open(os.path.join(os.path.dirname(__file__), "side.txt"), "w").write("helpmod ran")

        def scale(x):
            return x * 2
        """,
    )
    _write(
        tmp_path,
        "job.py",
        """
        import os, sys
        import cash

        @cash.cache
        def total(values, scaled):
            if scaled:
                from helpmod import scale
                return sum(scale(v) for v in values)
            return sum(values)

        print(total([1, 2, 3], False), "helpmod" in sys.modules, os.path.exists("side.txt"))
        """,
    )
    assert _run(tmp_path)[0] == "6 False False"
    assert _run(tmp_path)[0] == "6 False False", "a cache hit imported the module"


def test_a_module_whose_top_level_reads_the_environment_runs_with_a_warning(tmp_path):
    _write(tmp_path, "helpmod.py", 'import os\nK = int(os.environ.get("K", "2"))\n\ndef scale(x):\n    return x * K\n')
    _write(
        tmp_path,
        "job.py",
        """
        import sys
        import cash

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            from helpmod import scale
            return sum(scale(v) for v in values)

        print(total([1, 2, 3]))
        """,
    )
    out, ran, err = _run(tmp_path, env={"K": "2"})
    assert (out, ran) == ("12", True)
    assert "cannot read without running it" in err, err[-2000:]
    out, ran, _ = _run(tmp_path, env={"K": "3"})
    assert (out, ran) == ("18", True), "a value read from the environment was served stale"


def test_an_edit_to_a_module_the_helper_module_imports_invalidates(tmp_path):
    _write(tmp_path, "factors.py", "def factor():\n    return 2\n")
    _write(tmp_path, "helpmod.py", "from factors import factor\n\ndef scale(x):\n    return x * factor()\n")
    _write(
        tmp_path,
        "job.py",
        """
        import sys
        import cash

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            from helpmod import scale
            return sum(scale(v) for v in values)

        print(total([1, 2, 3]))
        """,
    )
    assert _run(tmp_path)[:2] == ("12", True)
    assert _run(tmp_path)[:2] == ("12", False), "an unedited second run did not hit"
    _write(tmp_path, "factors.py", "def factor():\n    return 3\n")
    assert _run(tmp_path)[:2] == ("18", True), "an edit two imports away was served stale"


def test_a_helper_whose_body_imports_the_module_is_keyed_by_it(tmp_path):
    _write(tmp_path, "helpmod.py", "def scale(x):\n    return x * 2\n")
    _write(
        tmp_path,
        "job.py",
        """
        import sys
        import cash

        def scaled(v):
            from helpmod import scale
            return scale(v)

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return sum(scaled(v) for v in values)

        print(total([1, 2, 3]))
        """,
    )
    assert _run(tmp_path)[:2] == ("12", True)
    assert _run(tmp_path)[:2] == ("12", False), "an unedited second run did not hit"
    _write(tmp_path, "helpmod.py", "def scale(x):\n    return x * 3\n")
    assert _run(tmp_path)[:2] == ("18", True), "the edited helper was served stale"


def test_a_submodule_imported_from_a_loaded_package_is_keyed(tmp_path):
    """`from pkg import sub` where `pkg` is loaded and `sub` is not: the
    package holds no `sub` yet, so the binding was not found at all."""
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/sub.py", "def scale(x):\n    return x * 2\n")
    _write(
        tmp_path,
        "job.py",
        """
        import sys
        import cash
        import pkg

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            from pkg import sub
            return sum(sub.scale(v) for v in values)

        print(total([1, 2, 3]))
        """,
    )
    assert _run(tmp_path)[:2] == ("12", True)
    assert _run(tmp_path)[:2] == ("12", False), "an unedited second run did not hit"
    _write(tmp_path, "pkg/sub.py", "def scale(x):\n    return x * 3\n")
    assert _run(tmp_path)[:2] == ("18", True), "the edited submodule was served stale"


def test_a_decorator_of_the_users_that_reads_the_environment_runs_with_a_warning(tmp_path):
    """A user function called as the module is imported (a decorator here)
    vouches for what it returns only when its own body does."""
    _write(
        tmp_path,
        "helpmod.py",
        """
        import os

        def pick(fn):
            return fn if os.environ.get("DOUBLE") else (lambda x: x)

        @pick
        def scale(x):
            return x * 2
        """,
    )
    _write(
        tmp_path,
        "job.py",
        """
        import sys
        import cash

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            from helpmod import scale
            return sum(scale(v) for v in values)

        print(total([1, 2, 3]))
        """,
    )
    out, ran, err = _run(tmp_path, env={"DOUBLE": "1"})
    assert (out, ran) == ("12", True)
    assert "cannot read without running it" in err, err[-2000:]
    out, ran, _ = _run(tmp_path, env={"DOUBLE": None})
    assert (out, ran) == ("6", True), "what the decorator picked by the environment was served stale"
