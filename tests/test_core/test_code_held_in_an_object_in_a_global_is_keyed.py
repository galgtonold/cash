"""A function held in an object's attribute inside a global is keyed by its code.

``CFG = {"b": Box(lambda x: x + 1)}`` read by a cached function: the dict and
a function in it were rewritten to the function's code, but an object was
left as it was, to be pickled. The lambda in it cannot be pickled, so the
global was dropped from the key with KEY-UNHASHABLE-GLOBAL and the call still
cached: editing the lambda served the old result. An object whose state
pickles as its ``__dict__`` is now rewritten like a dict, so what it holds
reaches the key by its code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

import cash
from cash.decorator.key_values import stabilize_for_global_hash

pytestmark = [pytest.mark.core]


def _write(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30  # saved a while before the run, like an ordinary edit
    os.utime(path, (past, past))


def _run(project: Path) -> tuple[str, int]:
    """Run ``job.py``: its answer, and how many times the body ran."""
    src = str(Path(cash.__file__).resolve().parents[1])  # the cash under test
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        CASH_CACHE_DIR=str(project / ".cash"),
        PYTHONPATH=os.pathsep.join([src, str(project)]),
    )
    p = subprocess.run(
        [sys.executable, "job.py"], cwd=str(project), env=env, capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout.strip().splitlines()[-1], p.stderr.count("[RUN]")


JOB = """
    import sys
    import time

    import cash
    from helper import CFG


    @cash.cache
    def run(x):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return {body}


    print(run(1))
"""

BOX_IN_A_DICT = """
    class Box:
        def __init__(self, fn):
            self.fn = fn

    CFG = {"b": Box(lambda x: x + 1)}
"""

BOX_AS_THE_GLOBAL = """
    class Box:
        def __init__(self, fn):
            self.fn = fn

    CFG = Box(lambda x: x + 1)
"""

NAMESPACE_IN_A_BOX = """
    from types import SimpleNamespace

    class Box:
        def __init__(self, inner):
            self.inner = inner

    CFG = Box(SimpleNamespace(steps=[lambda x: x + 1]))
"""

DATACLASS_IN_A_LIST = """
    from dataclasses import dataclass
    from typing import Callable

    @dataclass
    class Step:
        name: str
        fn: Callable

    CFG = [Step("inc", lambda x: x + 1)]
"""


@pytest.mark.parametrize(
    "helper, body",
    [
        pytest.param(BOX_IN_A_DICT, 'CFG["b"].fn(x)', id="object-in-a-dict"),
        pytest.param(BOX_AS_THE_GLOBAL, "CFG.fn(x)", id="object-as-the-global"),
        pytest.param(NAMESPACE_IN_A_BOX, "CFG.inner.steps[0](x)", id="namespace-in-an-object"),
        pytest.param(DATACLASS_IN_A_LIST, "CFG[0].fn(x)", id="dataclass-in-a-list"),
    ],
)
def test_editing_the_function_recomputes(tmp_path, helper, body):
    _write(tmp_path / "helper.py", helper)
    _write(tmp_path / "job.py", JOB.format(body=body))
    first = [_run(tmp_path), _run(tmp_path)]
    assert first == [("2", 1), ("2", 0)], "the unedited project was not cached"
    _write(tmp_path / "helper.py", helper.replace("x + 1", "x + 100"))
    assert _run(tmp_path) == ("101", 1), "the edited function was served the old result"


class _Plain:
    def __init__(self):
        self.rows = [1, 2]


@dataclass
class _Record:
    n: int


def test_an_object_that_holds_no_code_is_left_as_it_is():
    """Only an object with code inside is rewritten: the rest are keyed by
    their pickle, as before."""
    plain, record, ns = _Plain(), _Record(3), types.SimpleNamespace(a=1)
    out = stabilize_for_global_hash({"p": plain, "r": [record], "n": ns}, repr)
    assert out["p"] is plain and out["r"][0] is record and out["n"] is ns


def test_an_object_that_holds_itself_ends():
    box = _Plain()
    box.me = box
    box.fn = len
    out = stabilize_for_global_hash(box, repr)
    assert out[0] == "__cash_object__"
