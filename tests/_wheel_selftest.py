"""The C extension's own tests, run by cibuildwheel against each platform
wheel it builds (``test-command`` in pyproject.toml's ``[tool.cibuildwheel]``).

Importing the extension is not enough: a bug only one platform's compiler
shows (a 32-bit ``long`` on Windows) loads fine and gives wrong digests. So
every wheel, on its own OS, architecture and Python, runs:

1. the watcher tests, ``tests/test_core/watchers``;
2. a short run of the differential fuzz (``_fuzz.py``): random values,
   random changes, the watched digest against a fresh hash after each;
3. the notebook tests of watched variables with ``CASH_CWATCH_VERIFY=1``
   and no size threshold, so every digest the watchers reuse is hashed
   again and compared.

cibuildwheel copies ``tests`` and ``pyproject.toml`` (``test-sources``) to a
folder of their own and runs ``python -m tests._wheel_selftest`` there, so
``cash`` is the installed wheel, never ``src``. The test dependencies are
``test-requires``: pytest, pytest-timeout, IPython (the root conftest builds
the magics) and networkx; not xdist or the rerun plugin, which is why
``addopts`` is emptied.

A free-threaded Python (3.14t) reports the extension unusable: the tests
skip there, and the fuzz and the verify check are left out. On any other
Python the fast path must be on, or the wheel is broken.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTEST = [sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", "-rs"]
VERIFY_ENV = {"CASH_CWATCH_VERIFY": "1", "CASH_CWATCH_MIN_BYTES": "0"}
NOTEBOOK_SMOKE = "tests/test_notebook/values/test_a_watched_variable_changed_in_place_is_seen.py"


def _run(step: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n== {step}\n$ {' '.join(cmd)}", flush=True)
    full = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    full.update(env or {})
    code = subprocess.run(cmd, cwd=ROOT, env=full, check=False).returncode
    if code:
        raise SystemExit(f"{step}: exit {code}")


def _fast_path_on() -> bool:
    import cash
    from cash import _native

    print(f"cash {cash.__version__} from {Path(cash.__file__).parent}")
    print(f"Python {sys.version}")
    print(_native.status_line(), flush=True)
    if Path(cash.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"cash is imported from the source tree ({cash.__file__}), not the installed wheel")
    if _native.AVAILABLE:
        return True
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    raise SystemExit(f"the fast path is off on a Python with a GIL: {_native.REASON}")


def _verify_smoke() -> None:
    """In this process, so it can count what verify mode checked."""
    import pytest

    from cash import value_watch

    assert value_watch.VERIFY, "run with CASH_CWATCH_VERIFY=1"
    code = pytest.main(PYTEST[3:] + [NOTEBOOK_SMOKE])
    if code:
        raise SystemExit(f"verify smoke: exit {code}")
    if value_watch.enabled() and not value_watch.VERIFIED[0]:
        raise SystemExit("verify smoke: the watchers vouched for no digest, so nothing was verified")
    print(f"verify mode checked {value_watch.VERIFIED[0]} reused digests")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz-seconds", type=float, default=float(os.environ.get("CASH_WHEEL_FUZZ_SECONDS", 40)))
    ap.add_argument("--verify-smoke", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.verify_smoke:
        os.chdir(ROOT)
        _verify_smoke()
        return
    on = _fast_path_on()
    _run("watcher tests", [*PYTEST, "tests/test_core/watchers"])
    if on:
        fuzz = [sys.executable, "-m", "tests.test_core.watchers._fuzz", "--seconds", str(args.fuzz_seconds)]
        _run("differential fuzz", fuzz)
    else:
        print("\n== differential fuzz: left out, the fast path is off on this free-threaded Python")
    _run("verify-mode notebook tests", [sys.executable, "-m", "tests._wheel_selftest", "--verify-smoke"], VERIFY_ENV)


if __name__ == "__main__":
    main()
