"""A stored helper analysis outlives an edit to a function it never read,
and nothing else it took from that file.

A fresh process reads back the purity report of a cached function's whole
walk (`cash.analysis.report_store`). It was checked against the whole text
of every file a visited helper is in, so a commit touching any function of
such a file walked every cached function reaching it again. It is checked
now against the text of each callable it visited and what the walk took
from the rest of each file: the file with its function bodies cut out (the
imports, the globals, the conditions definitions sit under, a module
``__getattr__``), the text of each callable it read without visiting it,
and which globals the module's functions change. A finding keeps its line
from the start of the function holding it, numbered again when read back.

Each case runs plain Python and the cached function in a new process, with
``CASH_ANALYSIS_MEMO_VERIFY=1``, which walks the function again and raises
when the report read back differs from it.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_MAIN = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash
    from app import helpers as h

    def plain(k):
        return h.root(k) + h.dyn(k)

    cached = cash.cache(plain)
    value = cached(1)
    from cash.analysis import report_store
    from cash.analysis.purity_analyzer import get_analyzer
    # A hit may not ask for the analysis at all: ask for it.
    report = get_analyzer().analyze(plain)
    lines = sorted(i.line for i in report.issues if i.line)
    out = {"value": value, "plain": plain(1), "reports": report_store.STATS, "lines": lines}
    print(json.dumps(out))
    """
)

_HELPERS = textwrap.dedent(
    """
    from app.star import *

    SHIFT = 1
    TABLE = [1, 2, 3]
    SEEN = []
    FLAG = True


    def unrelated(x):
        return x + 0


    def lookup(k):
        return TABLE[k] + SHIFT


    def noisy(x):
        print("noisy", x)
        return x


    if FLAG:
        def pick(x):
            return x + 1
    else:
        def pick(x):
            return x + 2


    class Bag:
        def __getitem__(self, k):
            return k * 2


    BAG = Bag()


    def _one(x):
        return x + 10


    def _two(x):
        return x + 20


    class Model:
        KIND = staticmethod(_one)

        def run(self, x):
            return self.KIND(x)


    def root(k):
        return lookup(k) + noisy(k) + pick(k) + BAG[k] + scale(k) + step(k) + Model().run(k)


    step = _one


    def __getattr__(name):
        if name == "dyn":
            return _one
        raise AttributeError(name)
    """
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    code = tmp_path / "code"
    _write(code / "app" / "__init__.py", "")
    _write(code / "app" / "star.py", "def scale(x):\n    return x * 3\n")
    _write(code / "app" / "helpers.py", _HELPERS)
    _write(code / "main.py", _MAIN)
    (code / ".cash").mkdir()  # the stores live in an existing cache dir
    return code


def _run(code: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("main.py", str(code), cwd=code, env=env)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["value"] == result["plain"]
    return result


def _edit(code: Path, old: str, new: str) -> None:
    helpers = code / "app" / "helpers.py"
    text = helpers.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    helpers.write_text(text.replace(old, new), encoding="utf-8")


def _stored(code: Path) -> None:
    _run(code)
    again = _run(code)
    assert again["reports"]["loaded"] >= 1 and again["reports"]["stale"] == 0, again


def test_an_edit_to_a_function_the_walk_never_read_keeps_the_analysis(project):
    _stored(project)
    _edit(project, "return x + 0", "return x + 1  # edited")
    after = _run(project)
    assert after["reports"]["loaded"] >= 1 and after["reports"]["stale"] == 0, after
    assert _run(project, verify=True)["reports"]["verified"] >= 1


#: What the walk took from beyond the text of what it visited, each edited
#: in the file the helpers are in: the edit, whether the report must be
#: walked again (None: either way), and whether the value moves.
_FILE_LEVEL_EDITS = {
    # A function it never visits now changes a global a helper reads: the
    # module's global-change scan answers otherwise.
    "a function elsewhere changes a global read": (
        "return x + 0",
        "global SHIFT\n    SHIFT = 1\n    return x",
        True,
        False,
    ),
    # Lines added above a helper whose finding (the print) holds its line:
    # read back, the finding is numbered from where the helper starts now.
    "lines move under a finding": ("return x + 0", "x = x + 0\n    x = x + 0\n    return x", False, False),
    # A method read to decide whether to walk it, and not walked: it now
    # changes a global, and is walked.
    "a method read but not walked": ("return k * 2", "SEEN.append(k)\n        return k * 2", True, False),
    # The condition a definition sits under.
    "a conditional definition": ("FLAG = True", "FLAG = False", True, True),
    # A name a body reads, bound to another function further down.
    "a global rebound elsewhere in the file": ("step = _one", "step = _two", True, True),
    # A name a star import bound, shadowed by a definition further down.
    "a star import shadowed": ("step = _one", "step = _one\n\n\ndef scale(x):\n    return x * 100", True, True),
    # A class attribute a method reads.
    "a class attribute": ("KIND = staticmethod(_one)", "KIND = staticmethod(_two)", None, True),
    # What the module's ``__getattr__`` hands out for a name it lacks.
    "a module __getattr__": ("return _one\n    raise", "return _two\n    raise", None, True),
    # A global's value: the key reads it on every call, the report does not.
    "a global's value": ("SHIFT = 1", "SHIFT = 5", None, True),
}


@pytest.mark.parametrize("case", list(_FILE_LEVEL_EDITS))
def test_what_the_walk_took_from_the_file_still_counts(project, case):
    old, new, walked_again, value_moves = _FILE_LEVEL_EDITS[case]
    _stored(project)
    before = _run(project)["value"]
    _edit(project, old, new)
    after = _run(project, verify=True)
    if walked_again is not None:
        assert (after["reports"]["stale"] >= 1) == walked_again, after
    assert (after["value"] != before) == value_moves, after
    # The finding (the print) is numbered as the file numbers it now.
    printed = (project / "app" / "helpers.py").read_text(encoding="utf-8").splitlines().index('    print("noisy", x)')
    assert printed + 1 in after["lines"], after
    # Read back from then on, and still what walking it again gives.
    settled = _run(project, verify=True)
    assert settled["reports"]["stale"] == 0 and settled["value"] == after["value"], settled
