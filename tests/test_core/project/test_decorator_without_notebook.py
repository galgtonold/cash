"""`@cash.cache` runs with the notebook package unimportable.

The decorator shares file tracking, randomness detection, static analysis and
object hashing with the notebook path. That shared code lives outside
``cash.notebook``, and only the magics loaders import the notebook package, so
a cached function must work end to end (a miss, an in-memory hit, a disk hit in
a new process) while any import of ``cash.notebook`` fails.
"""

from __future__ import annotations

import ast
import subprocess
import textwrap
from pathlib import Path

import cash
from tests._scripts import run_python

_BLOCKER = """
import sys

class _BlockNotebook:
    def find_spec(self, name, path=None, target=None):
        if name == "cash.notebook" or name.startswith("cash.notebook."):
            raise ImportError(f"blocked by the test: {name}")
        return None

sys.meta_path.insert(0, _BlockNotebook())
"""

_SCRIPT = """
import os
import random
import sys
import time

try:
    import cash.notebook  # noqa: F401 - imported for its side effects, if present
except ImportError:
    pass
else:
    raise SystemExit("the blocker let cash.notebook through")

from cash import Cash, FileBackend

c = Cash(backend=FileBackend(cache_dir=sys.argv[1]), register_magic=False)
DATA = sys.argv[2]
RUNS = sys.argv[3]


@c.cache
def load(scale):
    fd = os.open(RUNS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x\\n")
    os.close(fd)
    random.seed(scale)
    with open(DATA) as f:
        return [scale * int(v) + random.randint(0, 0) for v in f.read().split()]


first = load(3)
second = load(3)
assert first == second == [3, 6, 9], (first, second)
leaked = sorted(m for m in sys.modules if m == "cash.notebook" or m.startswith("cash.notebook."))
assert not leaked, leaked
print("OK")
"""


def _run(tmp_path: Path, script: Path) -> subprocess.CompletedProcess:
    return run_python(script, tmp_path / "cache", tmp_path / "data.txt", tmp_path / "runs", cwd=tmp_path, check=False)


def test_a_cached_function_runs_without_the_notebook_package(tmp_path):
    (tmp_path / "data.txt").write_text("1 2 3\n", encoding="utf-8")
    script = tmp_path / "use_cash.py"
    script.write_text(
        textwrap.dedent(_BLOCKER) + textwrap.dedent(_SCRIPT),
        encoding="utf-8",
    )

    first = _run(tmp_path, script)
    assert first.returncode == 0 and "OK" in first.stdout, first.stderr
    # A new process restores from disk: the body ran once across both.
    second = _run(tmp_path, script)
    assert second.returncode == 0 and "OK" in second.stdout, second.stderr
    assert (tmp_path / "runs").read_text(encoding="utf-8") == "x\n"


def _cash_imports() -> set[tuple[str, str]]:
    """``(importing module, imported cash module)`` for every import in the
    source, at the top of a file or inside a function."""
    root = Path(cash.__file__).resolve().parent
    found: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        parts = ("cash", *path.relative_to(root).with_suffix("").parts)
        package = parts[:-1]
        if path.name == "__init__.py":
            parts = package
        module = ".".join(parts)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = list(package[: len(package) - (node.level - 1)]) if node.level else []
                home = ".".join(base + ([node.module] if node.module else []))
                # ``from .x import f`` names a module's name, or a package's module.
                inside = root.parent.joinpath(*home.split("."))
                targets = [
                    f"{home}.{alias.name}"
                    if (inside / f"{alias.name}.py").exists() or (inside / alias.name).is_dir()
                    else home
                    for alias in node.names
                ]
            else:
                continue
            found.update((module, target) for target in targets if target.split(".")[0] == "cash")
    return found


def _layer(module: str) -> str:
    if module == "cash.notebook" or module.startswith("cash.notebook."):
        return "notebook"
    if module == "cash.core" or module == "cash.decorator" or module.startswith("cash.decorator."):
        return "decorator"
    return "shared"


#: Code outside the notebook package that imports it. The two loaders are the
#: design; each other line is a helper still filed under ``notebook/`` that
#: the decorator path needs. Take a line out when its import goes; add none.
_INTO_THE_NOTEBOOK = {
    ("cash.core", "cash.notebook.ipython.magics"),  # Cash.register_magic
    ("cash.nbconvert", "cash.notebook.ipython.magics"),  # the nbconvert loader
    ("cash.diagnostics", "cash.notebook.ipython.cell_count"),
    ("cash.decorator.purity_checks", "cash.notebook.statement.identity_holders"),
    ("cash.analysis.cacheability_decision", "cash.notebook.shared_objects"),
    ("cash.analysis.cacheability_decision", "cash.notebook.lineage_formula"),
    ("cash.analysis.handed_callables", "cash.notebook.held_functions"),
}

#: Shared code that imports the decorator package, each inside a function.
#: The first five are entry points, which are there to reach `Cash`.
_INTO_THE_DECORATOR = {
    ("cash", "cash.core"),
    ("cash", "cash.decorator.arg_key"),
    ("cash.__main__", "cash.decorator.explain"),
    ("cash._active", "cash.core"),
    ("cash.reconfigure", "cash.core"),
    ("cash.ui.explorer", "cash.core"),
    ("cash.analysis.handed_callables", "cash.decorator.closure_fold"),
    ("cash.analysis.report_store", "cash.decorator.library_versions"),
    ("cash.argument_sharing", "cash.decorator.user_code"),
    ("cash.config.schema", "cash.decorator.state_memo"),
    ("cash.effect_observer", "cash.decorator.class_writes"),
    ("cash.kept_state", "cash.decorator.function_identity"),
    ("cash.tracking.trace_items", "cash.decorator.class_memo"),
    ("cash.tracking.trace_items", "cash.decorator.loop_leftovers"),
    ("cash.value_hash", "cash.decorator.arg_hashing"),
}


def test_no_new_code_outside_the_notebook_package_imports_it():
    """The rule is "the notebook imports the decorator and the shared layer,
    never the reverse", for an import inside a function as for one at the top
    of a file: a decorator-only script ran notebook code that way."""
    found = {
        (module, target)
        for module, target in _cash_imports()
        if _layer(module) != "notebook" and _layer(target) == "notebook"
    }
    assert found - _INTO_THE_NOTEBOOK == set(), "a new import of cash.notebook from outside it"
    assert _INTO_THE_NOTEBOOK - found == set(), "an import that is gone: take it off the list"


def test_no_new_shared_code_imports_the_decorator_package():
    found = {
        (module, target)
        for module, target in _cash_imports()
        if _layer(module) == "shared" and _layer(target) == "decorator"
    }
    assert found - _INTO_THE_DECORATOR == set(), "a new import of the decorator from the shared layer"
    assert _INTO_THE_DECORATOR - found == set(), "an import that is gone: take it off the list"


def test_code_that_looks_a_name_up_by_a_string_is_traced_without_the_notebook_package(tmp_path):
    """The trace asks whether a function's code looks a global up by a string;
    that check was a notebook module, so this call loaded the package."""
    script = tmp_path / "by_string.py"
    script.write_text(
        textwrap.dedent(_BLOCKER)
        + textwrap.dedent(
            """
            import sys

            from cash import Cash, FileBackend

            c = Cash(backend=FileBackend(cache_dir=sys.argv[1]), register_magic=False)
            FACTOR = 3


            @c.cache
            def scaled(x):
                return globals()["FACTOR"] * x


            assert scaled(2) == 6 and scaled(2) == 6
            assert "cash.string_lookups" in sys.modules
            leaked = sorted(m for m in sys.modules if m.startswith("cash.notebook"))
            assert not leaked, leaked
            print("OK")
            """
        ),
        encoding="utf-8",
    )
    done = run_python(script, tmp_path / "cache", cwd=tmp_path, check=False)
    assert done.returncode == 0 and "OK" in done.stdout, done.stderr
