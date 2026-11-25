"""A decorator is the call it is, to every analysis that looks at calls.

``@reg.add('x')`` above a ``def`` is ``reg.add('x')(f)`` and ``@register`` is
``register(f)``: the method's receiver is a candidate for "changed in place"
like the receiver of the call standing alone, and a function used as a
decorator is a function the statement calls.
"""

from __future__ import annotations

import ast
import functools
import types

import pytest

from cash.analysis.ast_util import called_names
from cash.analysis.callee_effects import callee_global_mutations
from cash.analysis.mutation_effects import classify_receivers
from cash.analysis.mutations import top_level_calls


def _calls(source: str):
    return top_level_calls(ast.parse(source))


@pytest.mark.parametrize(
    "source, pair",
    [
        ("@reg.add(PREFIX + '1')\ndef h(): pass", ("reg", "add")),
        ("@reg.plain\ndef h(): pass", ("reg", "plain")),
        ("@show.register(Kind)\ndef _(x): pass", ("show", "register")),
        ("@app.route('/x', methods=['GET'])\nasync def page(): pass", ("app", "route")),
        ("@reg.add('K')\nclass K: pass", ("reg", "add")),
        ("@other\n@hub.listeners.on('tick')\ndef h(): pass", ("hub", "on")),
    ],
)
def test_the_receiver_of_a_decorators_method_is_a_candidate_like_a_bare_calls(source, pair):
    calls = _calls(source)
    assert pair in calls.method_calls
    assert pair[0] in calls.decorator_receivers
    # The same candidate the call standing alone gives.
    assert pair in _calls(f"{pair[0]}.{'listeners.' if pair[0] == 'hub' else ''}{pair[1]}(h)").method_calls


def test_a_known_mutating_method_used_as_a_decorator_changes_its_receiver():
    assert _calls("@handlers.append\ndef h(): pass").mutation_receivers == {"handlers"}


def test_a_method_call_in_a_decorators_argument_is_a_call_whose_result_is_kept():
    calls = _calls("@reg.add(names.pop())\ndef h(): pass")
    assert ("names", "pop") in calls.assigned_method_calls


def test_a_plain_name_decorator_has_no_receiver():
    calls = _calls("@deco\n@make(3)\ndef h(): pass")
    assert calls.method_calls == frozenset() and calls.decorator_receivers == frozenset()


def test_an_undecided_receiver_of_a_decorator_is_left_to_be_watched():
    class Reg:
        def enroll(self, name):
            return lambda f: f

    tree = ast.parse("@reg.enroll('x')\ndef h(): pass")
    classes = classify_receivers(tree, {"reg": Reg()}, lambda: None, arguments=())
    assert classes.unknown_receivers == {"reg"}
    # Once a run was seen changing it, it is changed.
    assert classify_receivers(tree, {"reg": Reg()}, lambda: {"reg"}, arguments=()).mutated == {"reg"}
    assert classify_receivers(tree, {"reg": Reg()}, lambda: set(), arguments=()).mutated == frozenset()


def test_a_decorator_of_a_module_changes_nothing_the_notebook_holds():
    tree = ast.parse("@functools.lru_cache(maxsize=None)\ndef sq(x): return x * x")
    classes = classify_receivers(tree, {"functools": functools}, lambda: None, arguments=())
    assert classes.mutated == frozenset() and classes.unknown_receivers == frozenset()


def test_cashs_own_decorator_on_an_object_of_the_notebook_is_not_a_change(cash_instance):
    assert not isinstance(cash_instance, types.ModuleType)
    tree = ast.parse("@c.cache\ndef f(x): return x\n")
    classes = classify_receivers(tree, {"c": cash_instance}, lambda: None, arguments=())
    assert classes.mutated == frozenset() and classes.unknown_receivers == frozenset()
    # A bare call on it is judged as before.
    bare = classify_receivers(ast.parse("c.clear_cache()"), {"c": cash_instance}, lambda: None, arguments=())
    assert bare.unknown_receivers == {"c"}


@pytest.mark.parametrize("scope", ["all", "top_level", "eager", "no_control_bodies"])
def test_a_name_used_as_a_decorator_is_a_name_the_statement_calls(scope):
    tree = ast.parse("@register\n@make(option)\nclass K:\n    @inner\n    def m(self): helper()")
    names = called_names(tree, scope)
    assert {"register", "make"} <= names
    # What runs only when the class body or the method runs follows the scope.
    assert ("helper" in names) == (scope in ("all", "no_control_bodies"))


def test_a_global_a_decorator_function_changes_is_changed_by_the_def():
    sources = {"register": "def register(f):\n    REGISTRY.append(f.__name__)\n    return f\n"}
    tree = ast.parse("@register\ndef h(): pass")
    assert callee_global_mutations(tree, sources.get) == {"REGISTRY"}
    # The decorated function's own body is not run by defining it.
    tree = ast.parse("@plain\ndef h():\n    LOG.append(1)")
    assert callee_global_mutations(tree, {"plain": "def plain(f):\n    return f\n"}.get) == frozenset()
