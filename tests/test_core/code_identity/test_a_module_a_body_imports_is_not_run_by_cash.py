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


@pytest.mark.parametrize(
    "spelled",
    [
        "from nsp.mod import scale",
        "from nsp import mod\nscale = mod.scale",
        "import nsp.mod\nscale = nsp.mod.scale",
    ],
    ids=["from_the_module", "from_the_folder", "import_the_module"],
)
def test_a_module_in_a_folder_with_no_init_is_keyed_by_its_source(tmp_path, spelled):
    """A folder with no ``__init__.py`` (a namespace package) has no file of
    its own: the reading stopped there, and the module in it was keyed by
    the digest of nothing, so an edit to it was served stale."""
    _write(tmp_path, "nsp/mod.py", "def scale(x):\n    return x * 2\n")
    _write(
        tmp_path,
        "job.py",
        f"""
        import sys
        import cash

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            {spelled.replace(chr(10), chr(10) + " " * 12)}
            return sum(scale(v) for v in values)

        print(total([1, 2, 3]))
        """,
    )
    assert _run(tmp_path)[:2] == ("12", True)
    assert _run(tmp_path)[:2] == ("12", False), "an unedited second run did not hit"
    _write(tmp_path, "nsp/mod.py", "def scale(x):\n    return x * 5\n")
    assert _run(tmp_path)[:2] == ("30", True), "the edited module was served stale"


@pytest.mark.parametrize(
    "top_level",
    [
        'K = max(map(eval, [\'int(__import__("os").environ["K"])\']))',
        "def build():\n    return sys.flags.optimize * 10 + 2\n\nK = build()",
        'K = max(map(__builtins__["eval"], [\'int(__import__("os").environ["K"])\']))',
        "import builtins\n"
        "F = max(map(object.__getattribute__, [builtins], ['__import__']))\n"
        "E = max(map(F, ['os']))\n"
        "K = int(max(map(object.__getattribute__, [E], ['environ']))['K'])",
        "import builtins\n\ndef _unused(getattr):\n    return getattr\n\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import operator, builtins\n"
        "E = max(map(operator.attrgetter('__import__'), [builtins]))\n"
        "K = int(max(map(E, ['os'])).environ['K'])",
        "import operator, builtins\n"
        "E = max(map(operator.methodcaller('__import__', 'os'), [builtins]))\n"
        "K = int(E.environ['K'])",
        "import builtins\ntry:\n    pass\nexcept Exception as getattr:\n    pass\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\nif False:\n    getattr = 1\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])\n"
        "getattr = abs",
        "import builtins\n"
        "E = max(map(max(map(builtins.getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(builtins.getattr, [E], ['environ']))['K'])",
        "import builtins as b\n"
        "E = max(map(max(map(b.getattr, [b], ['__import__'])), ['os']))\n"
        "K = int(max(map(b.getattr, [E], ['environ']))['K'])",
        "import builtins\nfrom builtins import getattr\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\nfrom builtins import getattr as ga\n"
        "E = max(map(max(map(ga, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(ga, [E], ['environ']))['K'])",
        "import builtins\ngetattr = abs\ndef _drop():\n    global getattr\n    del getattr\n_drop()\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\ngetattr = abs\nif True:\n    from builtins import getattr\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\ngetattr = abs\nfrom builtins import *\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\nclass C:\n    getattr = abs\n"
        "    E = [max(map(max(map(getattr, [builtins], ['__import__'])), ['os'])) for _ in [0]][0]\n"
        "    K = [int(max(map(getattr, [E], ['environ']))['K']) for E in [E]][0]\nK = C.K",
        "import builtins\ndef _u():\n    from functools import reduce as getattr\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\ndef _u():\n    import builtins as getattr\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\nclass C:\n    from functools import reduce as getattr\n"
        "E = max(map(max(map(getattr, [builtins], ['__import__'])), ['os']))\n"
        "K = int(max(map(getattr, [E], ['environ']))['K'])",
        "import builtins\nclass C:\n    from functools import reduce as getattr\n"
        "    E = [max(map(max(map(getattr, [builtins], ['__import__'])), ['os'])) for _ in [0]][0]\n"
        "    K = [int(max(map(getattr, [E], ['environ']))['K']) for E in [E]][0]\nK = C.K",
    ],
    ids=[
        "eval_handed_to_map",
        "attribute_read_in_a_function_called_at_import",
        "eval_taken_from_dunder_builtins",
        "a_method_under_an_allowed_builtin",
        "getattr_a_parameter_elsewhere_names",
        "an_attrgetter_of_a_dunder",
        "a_methodcaller_of_a_dunder",
        "getattr_an_except_handler_names",
        "getattr_bound_on_a_path_that_does_not_run",
        "getattr_bound_below_its_use",
        "getattr_read_from_the_builtins_module",
        "getattr_read_from_the_builtins_module_by_another_name",
        "getattr_imported_from_builtins",
        "getattr_imported_from_builtins_by_another_name",
        "getattr_deleted_by_a_function_the_top_level_calls",
        "getattr_imported_again_under_an_if",
        "getattr_imported_again_by_a_star_import",
        "getattr_in_a_comprehension_of_a_class_body",
        "getattr_imported_in_a_function_as_another_function",
        "getattr_imported_in_a_function_as_the_builtins_module",
        "getattr_imported_in_a_class_body",
        "getattr_imported_in_a_class_body_read_in_its_comprehension",
    ],
)
def test_a_value_read_outside_the_source_without_a_call_runs_with_a_warning(tmp_path, top_level):
    """``map(eval, ...)`` hands ``eval`` on as a value, and a function called
    at import may read ``sys.flags`` without calling anything: both read
    outside the source, so the module cannot be vouched for by it. So does
    ``eval`` taken from ``__builtins__``, ``object.__getattribute__``
    (allowed as ``object`` was), and ``getattr`` at the top level, which a
    parameter of another function in the module was taken to rebind, an
    ``except ... as getattr`` unbinds as its handler ends, or an ``if False:``
    never binds, or ``getattr = abs`` below the use, or read from the
    ``builtins`` module an import names (``builtins.getattr``, ``from
    builtins import getattr``), which is no rebinding, or an import in a function or class binds, which leaves the top level's the builtin; and ``operator.attrgetter`` or ``methodcaller`` given
    ``'__import__'``, ``getattr`` by another name."""
    _write(tmp_path, "helpmod.py", f"import sys\n\n{top_level}\n\ndef scale(x):\n    return x * K\n")
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
    out, ran, err = _run(tmp_path, env={"K": "2", "PYTHONOPTIMIZE": None})
    assert (out, ran) == ("12", True)
    assert "cannot read without running it" in err, err[-2000:]
    out, ran, _ = _run(tmp_path, env={"K": "12", "PYTHONOPTIMIZE": "1"})
    assert (out, ran) == ("72", True), "a value read outside the source was served stale"


def test_a_module_with_only_its_bytecode_runs_with_a_warning(tmp_path):
    """A ``.pyc`` with no ``.py`` beside it has no source to stand for it: it
    was taken for a library and not keyed at all, so an edit was served
    stale: before the body imported it, and after."""
    import py_compile

    def compiled(factor):
        _write(tmp_path, "src/pycmod.py", f"def scale(x):\n    return x * {factor}\n")
        py_compile.compile(str(tmp_path / "src/pycmod.py"), cfile=str(tmp_path / "pycmod.pyc"), doraise=True)

    compiled(2)
    _write(
        tmp_path,
        "job.py",
        """
        import sys
        import cash

        @cash.cache
        def total(values):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            from pycmod import scale
            return sum(scale(v) for v in values)

        print(total([1, 2, 3]), total([1, 2, 3]))
        """,
    )
    out, ran, err = _run(tmp_path)
    assert (out, ran) == ("12 12", True)
    assert "cannot read without running it" in err, err[-2000:]
    compiled(5)
    assert _run(tmp_path)[:2] == ("30 30", True), "the edited module was served stale"


@pytest.mark.parametrize(
    "top_level",
    [
        "K = max(map(str.upper, ['k']))",
        "def pick(getattr):\n    return max(map(getattr, [1], ['real']))\n\nK = pick(lambda value, name: 3)",
        "import operator\nK = max(map(operator.attrgetter('real'), [3]))",
        "getattr = abs\nK = max(map(getattr, [-3]))",
        "import numpy as np\nW = np.zeros(4)\nK = int(np.max([3]))",
    ],
    ids=[
        "a_str_method_handed_to_map",
        "a_parameter_named_like_a_builtin",
        "an_attrgetter_of_a_public_name",
        "a_builtin_name_the_module_always_binds",
        "numpy_arrays_made_from_their_arguments",
    ],
)
def test_a_source_only_top_level_is_still_vouched_for(tmp_path, monkeypatch, top_level):
    """A public method of ``str`` handed on, and a parameter named
    ``getattr`` used in its own function's body, follow from the source,
    as do ``np.zeros(4)`` and ``np.max([3])``."""
    from cash.analysis.unloaded_modules import source_digest

    _write(tmp_path, "plainmod.py", f"{top_level}\n\ndef scale(x):\n    return x\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert source_digest("plainmod", None) is not None
