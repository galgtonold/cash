"""A fresh process reuses what earlier analysis worked out from source text
alone, and only for the same text.

The first call of each ``@cash.cache`` function walks every helper it reaches.
A helper shared by many cached functions was parsed and scanned once per
cached function, and every module a helper lives in was parsed whole to find
the globals its functions change, in every new process. Now a helper's tree
and what follows from the tree alone are kept for the process
(`cash.analysis.tree_memo`), and a module's scan is kept on disk under a
digest of its exact text, the Python and cash's own code
(`cash.analysis.scan_store`). An edit -- to code, to a comment only, a
helper added or removed -- or a file moved elsewhere is scanned again;
an unchanged module is loaded. ``CASH_ANALYSIS_MEMO_VERIFY=1`` works every
reused answer out again and raises on a difference.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import cash
from cash.analysis import scan_store, tree_memo

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash
    import mod
    from cash.analysis import scan_store
    f = cash.cache(mod.top)
    with cash.assume_safe():
        f(2)
    print(json.dumps({"stats": scan_store.status()}))
    """
)

_MODULE = """
COUNTER = []
LIMIT = 3


def helper(x):
    return x + LIMIT


def top(x):
    return helper(x) * 2
"""


def _run(code_dir: Path, store: Path, *, verify: bool = False) -> dict:
    env = dict(os.environ)
    env["CASH_ANALYSIS_STORE"] = str(store)
    env["CASH_CACHE_DIR"] = str(code_dir / ".cash")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if verify:
        env["CASH_ANALYSIS_MEMO_VERIFY"] = "1"
    src_root = str(Path(cash.__file__).parent.parent)
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(code_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])["stats"]


def _write(code_dir: Path, text: str) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "mod.py").write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture
def layout(tmp_path):
    code_dir = tmp_path / "code"
    _write(code_dir, _MODULE)
    return code_dir, tmp_path / "store"


def test_an_unchanged_module_is_loaded_in_a_second_process(layout):
    code_dir, store = layout
    first = _run(code_dir, store)
    assert first["missed"] >= 1 and first["stored"] >= 1
    second = _run(code_dir, store, verify=True)
    assert second["missed"] == 0
    assert second["loaded"] >= 1
    assert second["verified"] == second["loaded"]


def _stored_names(store: Path, code_dir: Path) -> list[str]:
    """The names the store now holds for *code_dir*'s ``mod.py``."""
    digest = hashlib.blake2b((code_dir / "mod.py").read_text(encoding="utf-8").encode(), digest_size=20).hexdigest()
    found = [json.loads(p.read_text()) for p in store.rglob("*.json")]
    (entry,) = [e for e in found if e["digest"] == digest]
    return entry["names"]


# Each process's first lookup of a text it has not met misses; a later one in
# the same process may load what the first stored (a just-written file is not
# settled, so the in-process memo does not keep it).
@pytest.mark.parametrize(
    ("edited", "names"),
    [
        # A helper's code changed: it now changes a global.
        (_MODULE.replace("return x + LIMIT", "COUNTER.append(x)\n    return x + LIMIT"), ["COUNTER"]),
        # A helper added that changes a global.
        (_MODULE + "\n\ndef bump():\n    global LIMIT\n    LIMIT += 1\n", ["LIMIT"]),
        # A comment only: the code is the same, the text is not.
        (_MODULE.replace("def helper(x):", "# a note\ndef helper(x):"), []),
    ],
    ids=["helper-edited", "helper-added", "comment-only"],
)
def test_an_edited_module_is_scanned_again(layout, edited, names):
    code_dir, store = layout
    _run(code_dir, store)
    assert _stored_names(store, code_dir) == []
    _write(code_dir, edited)
    again = _run(code_dir, store, verify=True)
    assert again["missed"] >= 1
    assert _stored_names(store, code_dir) == names


def test_a_removed_helper_is_scanned_again(tmp_path):
    code_dir, store = tmp_path / "code", tmp_path / "store"
    with_bump = _MODULE + "\n\ndef bump():\n    global LIMIT\n    LIMIT += 1\n"
    _write(code_dir, with_bump)
    _run(code_dir, store)
    assert _stored_names(store, code_dir) == ["LIMIT"]
    _write(code_dir, _MODULE)
    again = _run(code_dir, store, verify=True)
    assert again["missed"] >= 1
    assert _stored_names(store, code_dir) == []


def test_a_moved_module_is_scanned_again(layout, tmp_path):
    code_dir, store = layout
    _run(code_dir, store)
    moved = tmp_path / "elsewhere"
    _write(moved, _MODULE)
    again = _run(moved, store, verify=True)
    assert again["missed"] >= 1
    assert len(list(store.rglob("*.json"))) == 2  # one per file


def test_the_scan_answers_follow_the_text(tmp_path, monkeypatch):
    monkeypatch.setenv("CASH_ANALYSIS_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(tree_memo, "VERIFY", False)  # it would compute each answer again
    calls = []

    def compute(source: str) -> frozenset[str]:
        calls.append(source)
        return frozenset({"A"}) if "A.append" in source else frozenset()

    plain = "def f():\n    return A\n"
    mutating = "def f():\n    A.append(1)\n"
    assert scan_store.scan("t", "/x/m.py", plain, compute) == frozenset()
    assert scan_store.scan("t", "/x/m.py", plain, compute) == frozenset()
    assert len(calls) == 1  # the second answer came from disk
    assert scan_store.scan("t", "/x/m.py", mutating, compute) == {"A"}
    assert len(calls) == 2


def test_the_verify_mode_raises_on_a_stored_scan_that_is_wrong(tmp_path, monkeypatch):
    monkeypatch.setenv("CASH_ANALYSIS_STORE", str(tmp_path / "store"))
    source = "def f():\n    A.append(1)\n"
    scan_store.scan("t", "/x/m.py", source, lambda s: frozenset({"A"}))
    (stored,) = list((tmp_path / "store").rglob("*.json"))
    data = json.loads(stored.read_text())
    data["names"] = []
    stored.write_text(json.dumps(data))
    monkeypatch.setattr(tree_memo, "VERIFY", True)
    with pytest.raises(scan_store.StoredScanMismatch):
        scan_store.scan("t", "/x/m.py", source, lambda s: frozenset({"A"}))


def test_a_scan_that_gave_up_is_not_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("CASH_ANALYSIS_STORE", str(tmp_path / "store"))
    assert scan_store.scan("t", "/x/m.py", "x = 1\n", lambda s: None) == frozenset()
    assert not list((tmp_path / "store").rglob("*.json"))


def test_a_shared_tree_is_never_changed_by_a_walk(monkeypatch):
    """The walk strips decorators off a copy; the kept tree still has them."""
    monkeypatch.setattr(tree_memo, "VERIFY", False)  # it would compute each answer again
    src = "@deco\ndef f(x):\n    y = [x]\n    return y\n"
    tree = tree_memo.parse(src)
    func_def = tree.body[0]
    stripped = tree_memo.stripped_def(func_def)
    assert stripped.decorator_list == []
    assert len(func_def.decorator_list) == 1
    assert tree_memo.parse(src) is tree
    # A fact asked of the copy is kept for the next copy of the same tree.
    seen = []

    def count(node):
        seen.append(node)
        return len(node.body)

    assert tree_memo.fact(stripped, "n", count) == 2
    assert tree_memo.fact(tree_memo.stripped_def(func_def), "n", count) == 2
    assert len(seen) == 1


def test_an_edited_helper_is_parsed_anew():
    before = "def f(x):\n    return x\n"
    after = "def f(x):\n    return x + 1\n"
    assert tree_memo.parse(before) is not tree_memo.parse(after)
