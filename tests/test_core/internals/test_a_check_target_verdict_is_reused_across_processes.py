"""Whether a helper names a class only to test objects against it
(`check_targets.function_only_tests_against`) is read back in a fresh
process from what an earlier one worked out (`text_store`), only for the
same text, class name and builtin tests, and the key is the one working it
out gives.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from cash.analysis import text_store
    counts = {"loaded": 0, "computed": 0}
    kept = text_store.kept

    def counting(kind, text, extra, compute, encode, decode):
        ran = []

        def counted():
            ran.append(1)
            return compute()

        answer = kept(kind, text, extra, counted, encode, decode)
        if kind == "check-target-only":
            counts["computed" if ran else "loaded"] += 1
        return answer

    text_store.kept = counting
    import cash, mod
    key = cash.cache(mod.top).explain(2).cache_key
    print(json.dumps({"counts": counts, "key": key}))
    """
)

_MODULE = """
LIMIT = 3


class Kind:
    def size(self):
        return LIMIT


def helper(x):
    return 1 if isinstance(x, Kind) else 0


def top(k: "int") -> int:
    return helper(k) + k
"""

#: The same helper text, with ``isinstance`` rebound in its globals: no
#: longer the builtin test, so the class is read as data.
_REBOUND = _MODULE.replace("LIMIT = 3\n", "LIMIT = 3\n\n\ndef isinstance(x, kind):\n    return kind().size() > 2\n")


def _run(code_dir: Path, *, store: bool = True) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0"}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def code_dir(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "mod.py").write_text(_MODULE, encoding="utf-8")
    (code / ".cash").mkdir()
    return code


def test_a_second_process_reads_the_verdict_back(code_dir):
    first = _run(code_dir)
    assert first["counts"]["computed"] > 0
    second = _run(code_dir)
    assert second["counts"] == {"loaded": first["counts"]["computed"], "computed": 0}
    assert second["key"] == first["key"] == _run(code_dir, store=False)["key"]


def test_a_rebound_test_is_worked_out_again(code_dir):
    _run(code_dir)
    _run(code_dir)
    (code_dir / "mod.py").write_text(_REBOUND, encoding="utf-8")
    rebound = _run(code_dir)
    assert rebound["counts"]["computed"] > 0
    assert rebound["key"] == _run(code_dir, store=False)["key"]
    # LIMIT is data the rebound test reads: an edit to it moves the key.
    (code_dir / "mod.py").write_text(_REBOUND.replace("LIMIT = 3", "LIMIT = 4"), encoding="utf-8")
    assert _run(code_dir)["key"] != rebound["key"]
