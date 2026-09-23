"""The cache folder the unit suite sets for module imports must not outlive them.

``tests/conftest.py`` points ``CASH_CACHE_DIR`` at a per-process folder while a
unit-test module is imported, and puts the old value back afterwards. It used to
put it back from ``pytest_collectreport``, and pytest does not send that report
for a module that is only on the way to a test named by node id
(``file.py::test``). The core set, CI shards and every rerun select tests that
way. The folder then stayed in ``os.environ`` for the rest of the session, and
each warm notebook kernel booted after it inherited it. All the integration
tests on that worker then shared one cache: a repeat of
``test_computed_time_matches_saved_time`` found its DataFrame already cached and
had no EXECUTED row to read.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().relative_to(_ROOT).as_posix()

_PROBE = textwrap.dedent(
    """
    import os

    def pytest_collection_finish(session):
        print("CASH_CACHE_DIR_AFTER_COLLECTION=" + repr(os.environ.get("CASH_CACHE_DIR")))
    """
)


def _env_after_collecting(selection: str, tmp_path: Path) -> str:
    (tmp_path / "cache_dir_probe.py").write_text(_PROBE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "CASH_CACHE_DIR"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-n", "0", "-p", "no:cacheprovider"]
        + ["-p", "cache_dir_probe", selection],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("CASH_CACHE_DIR_AFTER_COLLECTION="):
            return line.split("=", 1)[1]
    raise AssertionError(f"the probe did not report; pytest said:\n{proc.stdout}\n{proc.stderr}")


@pytest.mark.parametrize(
    "selection",
    [_HERE, f"{_HERE}::test_the_folder_is_gone_once_collection_ends"],
    ids=["by-file", "by-node-id"],
)
def test_the_folder_is_gone_once_collection_ends(selection, tmp_path):
    assert _env_after_collecting(selection, tmp_path) == "None", (
        "CASH_CACHE_DIR set for the import of an in-process test module is still "
        "set after collection; every notebook kernel booted later shares it"
    )
