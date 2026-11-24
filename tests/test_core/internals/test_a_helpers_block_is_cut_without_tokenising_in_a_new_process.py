"""A fresh process cuts a helper's source block from its settled file by the
length an earlier process found (`source_reading._kept_block`), kept under a
digest of the file's lines and where the block starts, so it does not
tokenise every block again; what it reads is what ``inspect`` reads (``same``: every object's text
compared with ``inspect.getsource`` in the process that read it back).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import inspect, json, sys
    sys.path.insert(0, sys.argv[1])
    from cash import source_reading
    source_reading.SETTLED_SECONDS = -1.0  # every file has settled
    from cash.analysis import text_store
    counts = {"loaded": 0, "computed": 0}
    kept = text_store.kept

    def counting(kind, text, extra, compute, encode, decode):
        ran = []

        def counted():
            ran.append(1)
            return compute()

        answer = kept(kind, text, extra, counted, encode, decode)
        if kind == "source-block":
            counts["computed" if ran else "loaded"] += 1
        return answer

    text_store.kept = counting
    import cash, mod
    key = cash.cache(mod.top).explain(2).cache_key
    objs = [mod.helper, mod.top, mod.Box, mod.Box.size, mod.square, mod.outer, mod.outer()]
    same = all(source_reading.getsource(o) == inspect.getsource(o) for o in objs)
    print(json.dumps({"counts": counts, "key": key, "same": same,
                      "helper": source_reading.getsource(mod.helper)}))
    """
)

_MODULE = """
TABLE = [1, 2, 3]


def helper(k):
    # a comment inside the block
    return TABLE[k]


class Box:
    def size(self):
        return len(TABLE)


square = lambda v: v * v  # noqa: E731


def outer():
    def inner(j):
        return j + 1
    return inner


def top(k: "int") -> int:
    return helper(k) + Box().size() + square(k) + outer()(k)
"""


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


def test_a_second_process_cuts_each_block_by_the_kept_length(code_dir):
    first = _run(code_dir)
    assert first["same"] and first["counts"]["computed"] > 0
    second = _run(code_dir)
    assert second["same"]
    assert second["counts"]["computed"] == 0 and second["counts"]["loaded"] > 0
    assert second["key"] == first["key"]
    plain = _run(code_dir, store=False)
    assert plain["same"] and plain["counts"]["loaded"] == 0
    assert plain["key"] == first["key"]


def test_a_block_grown_below_the_same_first_line_is_cut_anew(code_dir):
    before = _run(code_dir)
    _run(code_dir)
    grown = _MODULE.replace("    return TABLE[k]\n", "    k = k + 1\n    k = k - 1\n    return TABLE[k] * 2\n")
    (code_dir / "mod.py").write_text(grown, encoding="utf-8")
    after = _run(code_dir)
    assert after["same"] and after["counts"]["computed"] > 0
    assert after["helper"].endswith("    return TABLE[k] * 2\n")
    assert after["key"] != before["key"]
    assert after["key"] == _run(code_dir, store=False)["key"]
