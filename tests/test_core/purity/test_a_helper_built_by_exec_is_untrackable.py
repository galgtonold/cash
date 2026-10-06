"""A helper built by ``exec()`` / ``eval()`` from a string is an untrackable dependency.

``ns = {}; exec(open("rules.txt").read(), ns); score = ns["score"]``, or
``score = eval(cfg["score"], {"__builtins__": {}})``: the function has no
source file and no module, so it read as library code and was never keyed.
Editing the rules file served the old result with no word. Its text is data
the program read, so it is treated like ``exec`` / ``eval`` written in the
cached function's own body: ``CashImpureFunctionError`` unless waived.
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.exceptions import CashImpureFunctionError

_NS: dict = {}
exec("def score(x):\n    return x + 1\n", _NS)
SCORE = _NS["score"]
FORMULA = eval("lambda x: x * 2", {"__builtins__": {}})

# Built into this module's own namespace: a module to judge it by, keyed by
# its bytecode like any function without a source file.
exec("def in_module(x):\n    return x - 1\n")


def _cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def calls_exec_helper(x):
    return SCORE(x)


def calls_eval_lambda(x):
    return FORMULA(x)


def calls_waived(x):
    return SCORE(x)  # @cash:assume-safe


def calls_in_module(x):
    return in_module(x)  # noqa: F821 - defined by the exec above


@pytest.mark.parametrize("fn", [calls_exec_helper, calls_eval_lambda])
def test_calling_it_raises_and_names_the_line(tmp_path, fn):
    with pytest.raises(CashImpureFunctionError, match=r"untrackable_dep.*built by exec\(\)/eval\(\)"):
        _cash(tmp_path).cache(fn)(3)


def test_a_line_waiver_accepts_it(tmp_path):
    assert _cash(tmp_path).cache(calls_waived)(3) == 4


def test_assume_safe_accepts_it(tmp_path):
    assert _cash(tmp_path).cache(calls_exec_helper, assume_safe=True)(3) == 4


def test_one_built_into_a_module_namespace_is_keyed_as_before(tmp_path):
    assert _cash(tmp_path).cache(calls_in_module)(3) == 2


def test_one_the_cached_function_captures_is_keyed_by_its_code(tmp_path):
    c = _cash(tmp_path)

    def make(fn):
        @c.cache
        def run(x):
            return fn(x)

        return run

    add_one, add_two = eval("(lambda x: x + 1, lambda x: x + 2)", {})
    assert make(add_one)(1) == 2
    assert make(add_two)(1) == 3
