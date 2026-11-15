"""What a fresh process works out from a helper's text alone is read back
from disk (`cash.analysis.text_store`), only for the same text, and the key
is the one working it out gives.

The answers a waiver can change are never stored: a ``# @cash:assume-safe``
comment is read from the file, beside the text, and a ``with`` block may be
the waiver imported under another name.
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
    import cash, mod
    from cash.analysis import text_store
    key = cash.cache(mod.top).explain(2).cache_key
    print(json.dumps({"stats": text_store.status(), "key": key}))
    """
)

_MODULE = """
import contextlib

TABLE = [1, 2, 3]
LEDGER = []


def helper(k):
    from json import dumps
    return TABLE[k] + len(dumps(k))


def top(k: "int") -> int:
    return helper(k) + sum(TABLE)
"""


def _run(code_dir: Path, *, store: bool = True, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0", "CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def code_dir(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "mod.py").write_text(_MODULE, encoding="utf-8")
    (code / ".cash").mkdir()
    return code


def test_a_second_process_works_out_no_text_answer_again(code_dir):
    first = _run(code_dir)
    assert first["stats"]["computed"] > 0
    second = _run(code_dir, verify=True)
    assert second["stats"]["computed"] == 0 and second["stats"]["loaded"] > 0
    assert second["stats"]["verified"] == second["stats"]["loaded"]
    assert second["key"] == first["key"]
    plain = _run(code_dir, store=False)
    assert plain["stats"]["loaded"] == plain["stats"]["computed"] == 0
    assert plain["key"] == first["key"]


def test_an_edited_helper_is_worked_out_again(code_dir):
    before = _run(code_dir)
    _run(code_dir)
    mod = code_dir / "mod.py"
    mod.write_text(_MODULE.replace("TABLE[k] +", "TABLE[k] * 2 +"), encoding="utf-8")
    after = _run(code_dir, verify=True)
    assert after["stats"]["computed"] > 0 and after["stats"]["loaded"] > 0
    assert after["key"] != before["key"]
    assert after["key"] == _run(code_dir, store=False)["key"]


@pytest.mark.parametrize("corrupt", ["truncated", "not json", "wrong shape"])
def test_a_damaged_store_file_is_a_miss(code_dir, corrupt):
    first = _run(code_dir)
    files = [p for p in (code_dir / ".cash" / "analysis").rglob("*.json") if '"text-answers"' in p.read_text("utf-8")]
    assert files
    for p in files:
        text = p.read_text("utf-8")
        if corrupt == "truncated":
            text = text[: len(text) // 2]
        elif corrupt == "not json":
            text = "\x00garbage"
        else:
            data = json.loads(text)
            data["answers"] = {
                d: "not a digest" if isinstance(a, str) else {"not": "names"} for d, a in data["answers"].items()
            }
            text = json.dumps(data)
        p.write_text(text, encoding="utf-8")
    again = _run(code_dir)
    assert again["key"] == first["key"]


@pytest.mark.parametrize(
    "body",
    [
        "    LEDGER.append(k)  # @cash:assume-safe\n    return TABLE[k] + len(LEDGER)\n",
        "    with contextlib.nullcontext():\n        LEDGER.append(k)\n    return TABLE[k] + len(LEDGER)\n",
    ],
)
def test_globals_a_waiver_may_reach_are_not_stored(code_dir, body):
    source = _MODULE.replace(
        "    from json import dumps\n    return TABLE[k] + len(dumps(k))\n",
        body,
    )
    (code_dir / "mod.py").write_text(source, encoding="utf-8")
    _run(code_dir)
    stored = [
        p
        for p in (code_dir / ".cash" / "analysis").rglob("*.json")
        if '"of": "mutated-globals"' in p.read_text("utf-8")
    ]
    answers = {}
    for p in stored:
        answers.update(json.loads(p.read_text("utf-8"))["answers"])
    # top's answer is stored, helper's is not: one answer, for top.
    assert len(answers) == 1
