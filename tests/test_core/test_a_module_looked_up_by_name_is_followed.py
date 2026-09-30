"""A module or function looked up by a string is followed when the string is
written out, and reported when it is computed.

``sys.modules["helper"].g(x)``, ``globals()["helper"].g(x)``,
``vars(helper)["g"](x)``, ``helper.__dict__["g"](x)``,
``operator.attrgetter("g")(helper)(x)`` and
``operator.methodcaller("g", x)(helper)`` all call ``helper.g``, and none was
followed: an edit to ``g`` served the old result, without the warning
``getattr(helper, name)`` and ``importlib.import_module`` give.
"""

from __future__ import annotations

import warnings

import pytest

import cash
from cash.exceptions import CashImpureFunctionError, CashImpurityWarning
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S
from tests.test_core._edited_project import edited_runs

pytestmark = pytest.mark.core

CONSTANT = [
    "sys.modules['helper'].g(x)",
    "sys.modules.get('helper').g(x)",
    "globals()['helper'].g(x)",
    "vars(helper)['g'](x)",
    "helper.__dict__['g'](x)",
    "operator.attrgetter('g')(helper)(x)",
    "attrgetter('g')(helper)(x)",
    "operator.methodcaller('g', x)(helper)",
]


@pytest.mark.parametrize("call", CONSTANT)
def test_a_lookup_by_a_written_out_name_is_followed(tmp_path, call):
    main = f"""
import operator
import sys
import time
from operator import attrgetter

import cash
import helper


@cash.cache
def f(x):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return {call}


print(f(1))
"""
    files = {"main.py": main, "helper.py": "def g(x):\n    return x + 1\n"}
    first, after, uncached = edited_runs(tmp_path, files, [("helper.py", "x + 1", "x + 100")])
    assert first == "2"
    assert after == uncached == "101"


@pytest.mark.parametrize(
    "call",
    [
        "sys.modules[NAME].dumps(x)",
        "sys.modules.get(NAME).dumps(x)",
        "operator.attrgetter(ATTR)(json)(x)",
        "operator.methodcaller(ATTR, x)(json)",
    ],
)
def test_a_computed_module_or_attribute_is_refused_like_import_module(tmp_path, call):
    """What `importlib.import_module(name)` and `getattr(obj, name)(x)` get."""
    c = cash.Cash(cache_dir=str(tmp_path / "c"))
    f = c.cache(_function_from_source(tmp_path, f"def f(x):\n    return {call}\n"))
    with pytest.raises(CashImpureFunctionError, match="non-constant name"):
        f(1)


def test_a_call_through_a_computed_globals_lookup_is_warned_about(tmp_path):
    c = cash.Cash(cache_dir=str(tmp_path / "c"))
    f = c.cache(_function_from_source(tmp_path, "def f(x):\n    return globals()[NAME].dumps(x)\n"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(1) == "1"
    messages = [str(w.message) for w in caught if issubclass(w.category, CashImpurityWarning)]
    assert any("globals()[...] with a computed name" in m for m in messages), messages


def _function_from_source(tmp_path, body):
    """``f`` from a module file of its own, which reads ``NAME = "json"`` and
    ``ATTR = "dumps"`` as globals."""
    path = tmp_path / "dispatch_src.py"
    path.write_text(
        f"import json\nimport operator\nimport sys\n\nNAME = 'json'\nATTR = 'dumps'\n\n\n{body}", encoding="utf-8"
    )
    namespace: dict = {"__name__": "dispatch_src"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["f"]
