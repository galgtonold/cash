"""A ``@stateful`` function is refused whichever way the call is spelled.

After "Moving to a module" a stateful helper is called as
``helpers.announce(...)``, or ``pkg.helpers.announce(...)`` from a package.
Only bare names were looked up, so those spellings were cached and a re-run
skipped the call the marker exists to keep. The controls: a plain module
function still caches, and so does a ``@stateful`` name that is not called.
"""

from __future__ import annotations

import types

import pytest

from cash.purity import stateful


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def ns(mock_shell):
    @stateful
    def announce(text):
        return len(text)

    def measure(text):
        return len(text)

    helpers = _module("helpers", announce=announce, measure=measure)
    mock_shell.user_ns.update(
        helpers=helpers,
        pkg=_module("pkg", helpers=helpers),
        announce=announce,
    )
    return mock_shell.user_ns


def _reasons(statement_processor, code: str) -> list[str]:
    return statement_processor.process_statement(code).get("uncacheable_reasons", [])


@pytest.mark.parametrize(
    "code",
    [
        "receipt = announce('x')",
        "receipt = helpers.announce('x')",
        "receipt = pkg.helpers.announce('x')",
        "receipt = str(helpers.announce('x'))",
    ],
    ids=["bare-name", "module", "package", "nested-in-expression"],
)
def test_a_stateful_call_is_refused(statement_processor, ns, code):
    assert "Calls @stateful function" in _reasons(statement_processor, code)
    assert ns["receipt"] in (1, "1")


def test_a_plain_module_function_is_not_refused(statement_processor, ns):
    assert "Calls @stateful function" not in _reasons(statement_processor, "n = helpers.measure('abc')")


def test_naming_a_stateful_function_without_calling_it_is_not_refused(statement_processor, ns):
    assert "Calls @stateful function" not in _reasons(statement_processor, "f = helpers.announce")
