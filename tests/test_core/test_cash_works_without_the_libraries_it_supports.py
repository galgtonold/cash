"""``import cash`` must work without the libraries cash only works *with*.

``pip install cash-lib`` installs cash's own dependencies (psutil, ipynbname,
and tomli on Python 3.10) and nothing it merely supports: pandas, polars,
pyarrow, numpy, matplotlib, ipywidgets, redis and boto3 are there only when
the user already uses them. IPython arrives with ipynbname's kernel stack, but
the decorator must not need it either (the notebook layering rule), so it is
blocked too. It broke once -- ``statement/restore.py`` grew a module-level
``from IPython.display import ...`` that sits on the ``import cash`` chain
(``__init__`` → ``core`` → ``notebook`` → ``upstream`` → ``statement`` →
``restore``), and a plain script raised::

    >>> import cash
    ModuleNotFoundError: No module named 'IPython'

The rest of the suite runs with all of them importable (the dev env has
them), so no other test could catch this. This one can.

**Why a subprocess + meta_path finder**, rather than poking ``sys.modules``:
``cash`` — and very likely IPython, numpy, and friends — are already imported
by the time this test runs, so an in-process hack would be testing an already
warm module graph. A fresh interpreter with an import blocker installed
*before* the first ``import cash`` is the only way to genuinely exercise the
cold path.

**Why block more than IPython:** blocking the whole set means any future
stray top-level import of numpy/pandas/ipywidgets/... is caught by the same
guard rather than needing a new test each time.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# The import names of the libraries cash supports but does not install, plus
# IPython (see the module docstring).
BLOCKED_MODULES = [
    "IPython",
    "ipywidgets",
    "matplotlib",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "redis",
    "boto3",
]

_SCRIPT = textwrap.dedent(
    '''
    """Run in a fresh interpreter with the libraries cash only works with made
    unimportable. Prints PHASE markers that the parent test asserts on."""
    import io
    import sys
    import tempfile

    BLOCKED = set({blocked!r})


    class _BlockSupportedLibraries:
        """meta_path finder that makes the blocked libraries look uninstalled.

        Raising ModuleNotFoundError (an ImportError subclass) keeps genuine
        `try: import numpy / except ImportError:` feature-detection working,
        while a stray *unguarded* top-level import blows up loudly.
        """

        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root in BLOCKED:
                raise ModuleNotFoundError(
                    "No module named {{!r}} (blocked by the "
                    "bare-install guard)".format(root),
                    name=fullname,
                )
            return None


    sys.meta_path.insert(0, _BlockSupportedLibraries())
    # Drop anything already pulled in by site/sitecustomize before we start.
    for _name in list(sys.modules):
        if _name.split(".")[0] in BLOCKED:
            del sys.modules[_name]

    for _name in sorted(BLOCKED):
        try:
            __import__(_name)
        except ImportError:
            pass
        else:  # pragma: no cover - the blocker is broken
            raise AssertionError("blocker failed to block " + _name)

    # --- Phase 1: the README's headline command -------------------------
    import cash

    print("PHASE1_IMPORT_OK", cash.__version__)

    # --- Phase 2: a real decorator cache round-trip ----------------------
    from cash.backends import FileBackend

    calls = {{"n": 0}}
    c = cash.Cash(backend=FileBackend(tempfile.mkdtemp()))

    @c.cache(assume_safe=True)  # audited: the counter is the point
    def add(a, b):
        calls["n"] += 1
        return a + b

    first = add(2, 3)
    second = add(2, 3)
    info = add.cache_info()
    print("PHASE2_ROUNDTRIP", first, second, calls["n"], info["hits"])

    # --- Phase 3: the `cash` console-script entry point ------------------
    # [project.scripts] maps `cash` -> cash.__main__:main, so call exactly that.
    import cash.__main__

    _buf = io.StringIO()
    _real_stdout = sys.stdout
    sys.stdout = _buf
    try:
        sys.argv = ["cash", "version"]
        try:
            cash.__main__.main()
        except SystemExit:
            pass
    finally:
        sys.stdout = _real_stdout
    print("PHASE3_CLI", _buf.getvalue().strip())
    '''
)


@pytest.fixture(scope="module")
def bare_install_run(tmp_path_factory):
    """Run the blocked-libraries script once; hand its output to each test."""
    script = tmp_path_factory.mktemp("bare_install") / "bare_install_check.py"
    script.write_text(_SCRIPT.format(blocked=sorted(BLOCKED_MODULES)), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(script.parent),
    )
    if proc.returncode != 0:
        pytest.fail(
            "cash is not usable without the libraries it only works with "
            "(a plain `pip install cash-lib` would be broken).\n"
            f"--- exit code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc


def test_import_cash_without_supported_libraries(bare_install_run):
    """`import cash` must work without any of the libraries cash only works with."""
    assert "PHASE1_IMPORT_OK" in bare_install_run.stdout, bare_install_run.stdout


def test_decorator_round_trip_without_supported_libraries(bare_install_run):
    """The decorator must actually cache — 2nd call is a HIT, not a recompute."""
    line = next(line for line in bare_install_run.stdout.splitlines() if line.startswith("PHASE2_ROUNDTRIP"))
    _, first, second, n_calls, hits = line.split()

    assert first == second == "5", line  # same answer both times
    assert n_calls == "1", f"2nd call recomputed instead of hitting cache: {line}"
    assert hits == "1", f"cash did not record a cache hit: {line}"


def test_cli_entry_point_without_supported_libraries(bare_install_run):
    """The `cash` console script (cash.__main__:main) must run."""
    line = next(line for line in bare_install_run.stdout.splitlines() if line.startswith("PHASE3_CLI"))
    assert "cash" in line and line.strip() != "PHASE3_CLI", line
