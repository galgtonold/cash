"""A cached function passed as an argument, or held in a table or a closure,
is keyed by everything it depends on: its code, its helpers and the globals
it reads.

Passed in, it was keyed by cash's own wrapper -- the same code for every
cached function -- so neither an edit to its body nor to a global it reads
moved the key; walking that wrapper's globals also warned
KEY-UNHASHABLE-GLOBAL about cash's internals. Held in a module dict or list,
or captured by a factory, its code was keyed but not the globals it reads.
"""

from __future__ import annotations

import pytest

from tests._scripts import run_python
from tests.test_core._edited_project import edited_runs

pytestmark = pytest.mark.core

STEPS = """
import cash

RATE = 2


@cash.cache
def scale(x):
    return x * RATE
"""

EDITS = {"a global it reads": ("RATE = 2", "RATE = 3"), "its body": ("x * RATE", "x * RATE * 10")}

PASSED = {
    "bare": "apply(steps.scale, 5)",
    "in a list": "apply_first([steps.scale], 5)",
    "in a dict": "apply_first({0: steps.scale}, 5)",
    "under a partial": "apply(functools.partial(steps.scale), 5)",
    "held by an object": "apply_held(Box(steps.scale), 5)",
}

APP = """
import cash


class Box:
    def __init__(self, fn):
        self.fn = fn


@cash.cache
def apply(fn, x):
    return fn(x) + 1


@cash.cache
def apply_first(fns, x):
    return fns[0](x) + 1


@cash.cache
def apply_held(box, x):
    return box.fn(x) + 1
"""


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("passed", list(PASSED))
def test_a_cached_function_passed_in_is_keyed_by_its_state(tmp_path, passed, edit):
    main = f"import functools\n\nimport app\nimport steps\nfrom app import Box\n\nprint(app.{PASSED[passed]})\n"
    files = {"steps.py": STEPS, "app.py": APP, "main.py": main}
    first, after, uncached = edited_runs(tmp_path, files, [("steps.py", *EDITS[edit])])
    assert first == "11"
    assert after == uncached != first


def test_passing_a_cached_function_warns_about_nothing_of_cash_s(tmp_path):
    (tmp_path / "steps.py").write_text(STEPS, encoding="utf-8")
    (tmp_path / "app.py").write_text(APP, encoding="utf-8")
    (tmp_path / "main.py").write_text("import app\nimport steps\nprint(app.apply(steps.scale, 5))\n", encoding="utf-8")
    out = run_python("main.py", cwd=tmp_path, cache_dir=tmp_path / "cache", check=False)
    assert out.stdout.strip() == "11", out.stderr
    assert "KEY-UNHASHABLE-GLOBAL" not in out.stderr


HOLDERS = {
    "a module dict": "from steps import scale\nSTEPS = {'scale': scale}\n\n\n@cash.cache\ndef run(x):\n"
    "    return STEPS['scale'](x) + 1\n",
    "a module list": "from steps import scale\nSTEPS = [scale]\n\n\n@cash.cache\ndef run(x):\n"
    "    return STEPS[0](x) + 1\n",
    "a factory's closure": "from steps import scale\n\n\ndef make(fn):\n    @cash.cache\n    def run(x):\n"
    "        return fn(x) + 1\n\n    return run\n\n\nrun = make(scale)\n",
}


@pytest.mark.parametrize("holder", list(HOLDERS))
def test_a_held_cached_function_is_keyed_by_the_globals_it_reads(tmp_path, holder):
    app = f"import cash\n\n{HOLDERS[holder]}"
    files = {"steps.py": STEPS, "app.py": app, "main.py": "import app\n\nprint(app.run(5))\n"}
    first, after, uncached = edited_runs(tmp_path, files, [("steps.py", "RATE = 2", "RATE = 4")])
    assert first == "11"
    assert after == uncached == "21"
