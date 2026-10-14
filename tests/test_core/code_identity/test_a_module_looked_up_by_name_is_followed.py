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
from tests.test_core.code_identity._edited_project import edited_runs

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
def test_a_computed_module_or_attribute_runs_uncached_like_import_module(tmp_path, call):
    """What `importlib.import_module(name)` and `getattr(obj, name)(x)` get:
    run uncached with KEY-UNTRACKABLE-DEP, and raise under strict=True."""
    c = cash.Cash(cache_dir=str(tmp_path / "c"))
    f = c.cache(_function_from_source(tmp_path, f"def f(x):\n    return {call}\n"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f(1)
    messages = [str(w.message) for w in caught if "KEY-UNTRACKABLE-DEP" in str(w.message)]
    assert messages and "non-constant name" in messages[0], [str(w.message) for w in caught]
    strict = cash.Cash(cache_dir=str(tmp_path / "s")).cache(strict=True)(
        _function_from_source(tmp_path, f"def f(x):\n    return {call}\n")
    )
    with pytest.raises(CashImpureFunctionError, match="non-constant name"):
        strict(1)


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


BY_NAME_MAIN = """
import cash

cfg = 1
NAME = "make"


def make(x):
    return x + cfg


@cash.cache
def f(x):
    {body}


print(f(1))
"""


@pytest.mark.parametrize(
    "body",
    [
        "return globals()[NAME](x)",
        "return globals().get(NAME)(x)",
        "fn = globals()[NAME]\n    return fn(x)",
        "fn = globals().get(NAME)\n    return fn(x)",
        "return globals()['make'](x)",
        "return make(x)",
    ],
)
def test_a_function_named_by_a_string_is_keyed_on_what_it_reads(tmp_path, body):
    """``globals()[NAME](x)`` calls ``make``, whose ``cfg`` was in no key:
    after ``cfg = 2`` the next run was served 2 where an uncached run gives
    3. A computed name now runs uncached, as ``getattr(obj, name)(x)``
    does; a written-out one and a plain call are keyed on ``cfg``."""
    files = {"main.py": BY_NAME_MAIN.replace("{body}", body)}
    first, after, uncached = edited_runs(tmp_path, files, [("main.py", "cfg = 1", "cfg = 2")])
    assert first == "2"
    assert after == uncached == "3"


@pytest.mark.parametrize("call", ["vars(json)[ATTR](x)", "json.__dict__[ATTR](x)", "globals().get(NAME).dumps(x)"])
def test_a_computed_namespace_lookup_is_reported(tmp_path, call):
    c = cash.Cash(cache_dir=str(tmp_path / "c"))
    f = c.cache(_function_from_source(tmp_path, f"def f(x):\n    return {call}\n"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(1) == "1"
    messages = [str(w.message) for w in caught]
    assert any("non-constant name" in m or "computed name" in m for m in messages), messages
