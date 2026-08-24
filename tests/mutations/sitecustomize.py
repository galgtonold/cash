"""Apply a named mutation inside the notebook kernel.

The integration suite drives a REAL Jupyter kernel in a separate process, so a
monkeypatch applied in the pytest process never reaches the code under test.
That is not a theoretical gap: patching ``_backward_scan_pass`` from a pytest
plugin and watching 57 upstream integration tests pass reads exactly like "the
suite does not cover upstream", and is in fact "the patch was never there".
Every conclusion drawn that way is worthless.

Python imports ``sitecustomize`` automatically at interpreter startup if it is
importable, and ``KernelManager`` hands the kernel the parent environment -- so
putting this directory on ``PYTHONPATH`` gets code into the kernel with no
change to cash itself. Test hooks do not belong in ``src/``; the only
``CASH_*`` variable production code reads is a real configuration knob.

Usage::

    CASH_MUTATION=upstream-dead \\
    CASH_MUTATION_MARKER=/tmp/mut.json \\
    PYTHONPATH=tests/mutations \\
    pytest tests/test_notebook_integration/test_upstream_x.py

Inert unless ``CASH_MUTATION`` is set: no import of cash, no meta_path entry,
nothing. A stray PYTHONPATH cannot silently change a normal run.

The marker file is the point of the whole exercise. It records that the
mutation was applied AND how many times the broken code was actually called.
"Nothing failed" means one of two opposite things -- the suite tolerates a
broken engine, or the engine was never exercised -- and only the call count
tells them apart.
"""
from __future__ import annotations

import json
import os
import sys
import threading

_NAME = os.environ.get("CASH_MUTATION")
_MARKER = os.environ.get("CASH_MUTATION_MARKER")


class _Applier:
    """Patch *mutation*'s target as soon as that module is fully loaded.

    Registered on ``sys.meta_path``, which every import consults -- so this
    gets a chance to run on each import without ever claiming one (``find_spec``
    always returns None). Importing cash directly from ``sitecustomize`` would
    be simpler and is a bad idea: it runs during site initialisation, long
    before the interpreter is in a state where dragging in a large package is
    safe.

    The ``probe`` check matters. A module appears in ``sys.modules`` when its
    import STARTS, so patching on presence alone can land mid-body, before the
    class being patched exists. Waiting for the attribute means the module has
    got far enough to have it.
    """

    def __init__(self, mutation) -> None:
        self._mutation = mutation
        self._done = False
        self._lock = threading.Lock()
        self._calls = 0

    def _record(self) -> None:
        with self._lock:
            self._calls += 1
        self._write()

    def _marker_path(self) -> str:
        """One marker per PROCESS.

        A single shared path is last-writer-wins, and a mutation run spawns a
        kernel per test plus N xdist workers -- so the surviving file was
        whichever process happened to finish last, which reported calls=0 while
        47 tests were failing from the mutation. Per-pid files make the
        evidence add up instead of overwrite.
        """
        base, _, ext = _MARKER.rpartition(".")
        return f"{base or _MARKER}.{os.getpid()}.{ext or 'json'}"

    def _write(self) -> None:
        if not _MARKER:
            return
        try:
            with open(self._marker_path(), "w", encoding="utf-8") as fh:
                json.dump({
                    "mutation": self._mutation.name,
                    "applied": self._done,
                    "calls": self._calls,
                    "pid": os.getpid(),
                }, fh)
        except OSError:
            pass          # a marker we cannot write must not break the kernel

    def find_spec(self, name, path=None, target=None):
        if self._done:
            return None
        mod = sys.modules.get(self._mutation.target)
        if mod is None or not hasattr(mod, self._mutation.probe):
            return None
        self._done = True
        try:
            self._mutation.apply(mod, self._record)
            self._write()
        except Exception as exc:      # noqa: BLE001 - report, never mask
            sys.stderr.write(f"[cash-mutation] {self._mutation.name} failed: {exc!r}\n")
        finally:
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
        return None       # never claims an import


if _NAME:
    try:
        from _catalogue import CATALOGUE
    except ImportError:                                   # pragma: no cover
        sys.stderr.write("[cash-mutation] catalogue not importable\n")
        CATALOGUE = {}
    _mutation = CATALOGUE.get(_NAME)
    if _mutation is None:
        sys.stderr.write(
            f"[cash-mutation] unknown mutation {_NAME!r}; "
            f"known: {', '.join(sorted(CATALOGUE)) or '(none)'}\n"
        )
    else:
        _applier = _Applier(_mutation)
        _applier._write()          # 'applied: false' until the target loads
        sys.meta_path.insert(0, _applier)
