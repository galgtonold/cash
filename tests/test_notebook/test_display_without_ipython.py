"""Rich output without IPython must fail LOUDLY, never silently (CAS-132).

Two rules are in tension here, and this file pins both:

1. **The modules must IMPORT without IPython** — base ``cash`` declares
   ``dependencies = []`` and ``statement/`` sits on the ``import cash`` chain,
   so an unguarded module-level IPython import breaks a bare
   ``pip install cash-lib`` (the CAS-129 P0, guarded by
   ``tests/test_core/test_bare_install_no_optional_deps.py``).
2. **A genuine display attempt must RAISE** — ``processor.py`` used to satisfy
   rule 1 with module-level ``try/except ImportError`` stubs where
   ``publish_display_data`` was a silent no-op. Rich output would then vanish
   with the cell still reporting success: the worst failure mode a display call
   has. ``restore.py`` already did the opposite (CAS-129, function-local
   import); CAS-132 made ``processor.py`` match.

Satisfying 1 by breaking 2 is exactly the regression this file exists to catch,
so both are asserted from the same run.

**Why a subprocess + meta_path finder:** IPython is imported long before this
test runs (the dev env has it, and most of the suite needs it), so an in-process
``sys.modules`` hack would both test an already-warm module graph and pollute
the worker for whatever xdist schedules next — the very contamination
``.github/copilot-instructions.md`` forbids reintroducing. A fresh interpreter
with the blocker installed before the first import is the honest test.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    '''
    """Fresh interpreter with IPython made unimportable. Prints PHASE markers."""
    import sys


    class _BlockIPython:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "IPython":
                raise ModuleNotFoundError(
                    "No module named 'IPython' (blocked by the CAS-132 guard)",
                    name=fullname,
                )
            return None


    sys.meta_path.insert(0, _BlockIPython())
    for _name in list(sys.modules):
        if _name.split(".")[0] == "IPython":
            del sys.modules[_name]

    try:
        import IPython
    except ImportError:
        pass
    else:  # pragma: no cover - the blocker is broken
        raise AssertionError("blocker failed to block IPython")

    # --- Phase 1: both modules must still IMPORT (the CAS-129 P0) --------
    from cash.notebook.statement.processor import StatementProcessor
    from cash.notebook.statement.restore import StatementRestorer

    print("PHASE1_IMPORT_OK")

    # --- Phase 2: processor's rich-output replay must RAISE, not no-op ---
    # Unbound call: _publish_rich_outputs touches no instance state, so `None`
    # for `self` keeps this about the import and nothing else.
    try:
        StatementProcessor._publish_rich_outputs(
            None, [{"data": {"text/plain": "should not vanish"}}]
        )
    except ImportError as e:
        print("PHASE2_PROCESSOR_RAISED", type(e).__name__, str(e))
    else:
        print("PHASE2_PROCESSOR_SILENT")

    # --- Phase 3: restore.py's replay must RAISE too (CAS-129 parity) ----
    try:
        StatementRestorer._replay_cached_outputs(
            None, "", "", [{"data": {"text/plain": "should not vanish"}}]
        )
    except ImportError as e:
        print("PHASE3_RESTORE_RAISED", type(e).__name__)
    else:
        print("PHASE3_RESTORE_SILENT")

    # --- Phase 4: NO rich output => no import => must not raise ----------
    # The `if not outputs` guard keeps the hot path off IPython entirely; a
    # statement that produced no rich output is not a display attempt.
    StatementProcessor._publish_rich_outputs(None, [])
    StatementRestorer._replay_cached_outputs(None, "", "", [])
    print("PHASE4_EMPTY_OK")
    '''
)


@pytest.fixture(scope="module")
def no_ipython_run(tmp_path_factory):
    """Run the IPython-blocked script once; hand its output to each test."""
    script = tmp_path_factory.mktemp("no_ipython") / "no_ipython_check.py"
    script.write_text(_SCRIPT, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(script.parent),
    )
    if proc.returncode != 0:
        pytest.fail(
            "the IPython-blocked probe crashed\n"
            f"--- exit code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc


def test_statement_modules_import_without_ipython(no_ipython_run):
    """Rule 1: importing them must NOT need IPython (do not re-break CAS-129)."""
    assert "PHASE1_IMPORT_OK" in no_ipython_run.stdout, no_ipython_run.stdout


def test_processor_rich_output_raises_without_ipython(no_ipython_run):
    """Rule 2: a real display attempt raises — it must never silently vanish."""
    assert "PHASE2_PROCESSOR_SILENT" not in no_ipython_run.stdout, (
        "processor._publish_rich_outputs silently dropped rich output instead "
        "of raising — the CAS-132 no-op stub is back:\n" + no_ipython_run.stdout
    )
    line = next(
        line for line in no_ipython_run.stdout.splitlines()
        if line.startswith("PHASE2_PROCESSOR_RAISED")
    )
    assert "IPython" in line, f"error must name the missing dep: {line}"


def test_restore_rich_output_raises_without_ipython(no_ipython_run):
    """The CAS-129 sibling behaves the same way — the two modules agree."""
    assert "PHASE3_RESTORE_RAISED" in no_ipython_run.stdout, no_ipython_run.stdout


def test_no_rich_output_does_not_need_ipython(no_ipython_run):
    """No display attempt => no import => no error (the guard still guards)."""
    assert "PHASE4_EMPTY_OK" in no_ipython_run.stdout, no_ipython_run.stdout
