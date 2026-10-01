"""A cached function on another `Cash` instance is a dependency of its caller.

Each instance linked only the cached functions registered with itself, so a
caller on the default instance calling a function cached by ``app =
cash.Cash(...)`` -- or the other way round, or across two instances -- kept
its old result after that function's body or a global it reads changed.
"""

from __future__ import annotations

import pytest

from tests.test_core._edited_project import edited_runs

pytestmark = pytest.mark.core

DECORATORS = {
    "inner on an instance": ("app = cash.Cash()\n", "@app.cache", "", "@cash.cache"),
    "outer on an instance": ("", "@cash.cache", "app = cash.Cash()\n", "@app.cache"),
    "both on instances": ("app = cash.Cash()\n", "@app.cache", "other = cash.Cash()\n", "@other.cache"),
}


@pytest.mark.parametrize(
    "edit",
    [("n * G", "n * G * 15"), ("G = 2", "G = 30")],
    ids=["its body", "a global it reads"],
)
@pytest.mark.parametrize("decorators", list(DECORATORS))
def test_editing_the_other_instance_s_function_recomputes_the_caller(tmp_path, decorators, edit):
    inner_setup, inner_deco, outer_setup, outer_deco = DECORATORS[decorators]
    inner = f"import cash\n{inner_setup}G = 2\n\n\n{inner_deco}\ndef load(n):\n    return n * G\n"
    outer = f"import cash\nfrom inner import load\n{outer_setup}\n\n{outer_deco}\ndef top(n):\n    return load(n) + 1\n"
    files = {"inner.py": inner, "outer.py": outer, "main.py": "import outer\nprint(outer.top(2))\n"}
    first, after, uncached = edited_runs(tmp_path, files, [("inner.py", *edit)])
    assert first == "5"
    assert after == uncached == "61"
