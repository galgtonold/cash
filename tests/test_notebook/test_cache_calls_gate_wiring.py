"""The production call site must apply the object-level gate too (CAS-243).

``wrap_eligible_calls``'s ``gate`` parameter (Task 4) was tested at the AST
level but never wired into ``statement/processor.py``'s
``_code_and_tree_for_execution`` (Task 4 landed the function, not the wiring —
confirmed by ``grep -rn "gate=" src/`` returning zero hits before this change).
Structural eligibility alone (the free-variable rule in
``eligible_call_nodes``) says nothing about whether a call is *substantively*
cacheable -- ``time.time()`` doesn't read the statement's own target, so it is
structurally eligible, but caching it would be exactly the kind of thing
``decide_cacheability``'s forbidden-function scan exists to refuse for a plain
statement. Without the gate wired in, an intercepted call skipped that
judgment entirely: the call still got wrapped into
``__cash_call__(time.time, 0)()`` (a harmless no-op here, since ``time.time``
is a builtin and ``CallCache.resolve``'s own type check passes it straight
through) -- but the processor believed it had found something cacheable when
it had not.

Interception is on by default now (CAS-243 task 10), so the earlier "warns
when nothing is eligible" signal (``test_cache_calls_noop_warning.py``, since
deleted -- silence is the ordinary case under default-on, not a mistake worth
a warning) no longer exists to assert on. The two tests below instead spy on
``wrap_eligible_calls`` and assert directly on the ``sites`` it returns: zero
sites for the forbidden call, one for the genuinely eligible one. That is a
more direct check of "did the gate actually refuse this" than the warning
ever was.

Not exercised via ``nb_runner`` (no real kernel needed): this is the
synchronous statement path, and ``CashMagics`` + a mock shell reaches
``_code_and_tree_for_execution`` directly.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend, cash
    backend.clear()
    shell.user_ns.clear()


def _spy_on_wrap_eligible_calls(monkeypatch):
    """Patch ``wrap_eligible_calls`` to pass through but record the sites it
    returned, so a test can assert on how many call sites survived the gate.
    """
    import cash.notebook.statement.processor as processor_module

    captured = {}
    real = processor_module.wrap_eligible_calls

    def _spy(tree, *, gate):
        rewritten, sites = real(tree, gate=gate)
        captured["sites"] = sites
        return rewritten, sites

    monkeypatch.setattr(processor_module, "wrap_eligible_calls", _spy)
    return captured


def test_a_forbidden_call_is_never_wrapped_even_though_it_is_structurally_eligible(magics_fixture, monkeypatch):
    """``time.time()`` reads nothing of ``out`` -- structurally eligible -- but
    ``decide_cacheability`` forbids it for an ordinary statement, and the gate
    must apply that same refusal here: no site survives.
    """
    captured = _spy_on_wrap_eligible_calls(monkeypatch)
    magics, shell, _, _ = magics_fixture
    magics.cash("", "import time\nout = []")

    magics.cash("", "out.append(time.time())")

    assert captured["sites"] == [], "a forbidden call must not survive the gate"


def test_an_ordinary_eligible_call_is_wrapped(magics_fixture, monkeypatch):
    """Positive control for the test above, using the SAME statement shape
    (``out.append(...)``) so only the callee's cacheability differs.
    """
    captured = _spy_on_wrap_eligible_calls(monkeypatch)
    magics, shell, _, _ = magics_fixture
    magics.cash("", "def compute(x):\n    return x + 1\nout = []\nx = 1")

    magics.cash("", "out.append(compute(x))")

    assert len(captured["sites"]) == 1, (
        f"a genuinely cacheable call was not wrapped: {captured['sites']!r}"
    )


def test_a_gate_exception_fails_closed_instead_of_crashing_the_cell(magics_fixture, monkeypatch):
    """CAS-243 review I1: the gate runs the full ``decide_cacheability`` /
    ``analyze_statement`` / ``scan_for_forbidden_functions`` stack against a
    bare ``ast.Expr(Call)`` sub-expression -- a shape that stack had never
    been run against before this task wired the gate in. The ``try`` around
    ``wrap_eligible_calls`` only ever guarded a copy-and-unparse, so it only
    catches ``(SyntaxError, ValueError, TypeError, AttributeError)``; an
    exception of any other type escaping the gate would surface as the
    user's OWN traceback, on their line, for a caching optimisation gone
    wrong. Monkeypatching ``call_site_is_cacheable`` to raise a type outside
    that tuple is the sharpest way to prove the gate itself fails closed
    rather than relying on that narrower, pre-existing except clause.
    """
    import cash.notebook.statement.processor as processor_module

    def _raise(*args, **kwargs):
        raise RuntimeError("boom from inside the gate")

    monkeypatch.setattr(processor_module, "call_site_is_cacheable", _raise)

    magics, shell, _, _ = magics_fixture
    magics.cash("", "def compute(x):\n    return x + 1\nout = []\nx = 1")

    # Must not raise -- a caching optimisation must never be why user code
    # fails, and the statement must still actually run.
    magics.cash("", "# @cash:cache-calls\nout.append(compute(x))")
    assert shell.user_ns["out"] == [2], "the statement did not run to completion"


def test_the_gate_is_given_variable_lineage(magics_fixture, monkeypatch):
    """CAS-243 review I2: ``call_site_is_cacheable``'s ``variable_lineage``
    parameter is optional and, per its own docstring, omitting it makes
    "missing lineage" an unreachable refusal reason -- correct for the
    AST-only rewrite-time case that docstring describes, but this call site
    has a REAL, populated lineage table in scope (``self.variable_lineage``,
    the same object already passed to the statement's own
    ``decide_cacheability`` call one line above). Asserting on the actual
    keyword arguments the gate is built with is more direct than trying to
    construct a statement whose gate verdict differs only by this one
    argument.
    """
    import cash.notebook.statement.processor as processor_module

    captured_kwargs = {}
    real = processor_module.call_site_is_cacheable

    def _spy(call, **kwargs):
        captured_kwargs.update(kwargs)
        return real(call, **kwargs)

    monkeypatch.setattr(processor_module, "call_site_is_cacheable", _spy)

    magics, shell, _, _ = magics_fixture
    magics.cash("", "def compute(x):\n    return x + 1\nout = []\nx = 1")
    magics.cash("", "# @cash:cache-calls\nout.append(compute(x))")

    assert "variable_lineage" in captured_kwargs, (
        "the gate never passes variable_lineage, even though a real lineage "
        "table is in scope at this call site"
    )
    assert captured_kwargs["variable_lineage"] is magics._statement_processor.variable_lineage, (
        "the gate passed SOME variable_lineage, but not the processor's own "
        "live table -- a fabricated stand-in would satisfy 'is it passed' "
        "without actually wiring in the real lineage"
    )
    # And it isn't an accidentally-empty table either: `x` was assigned above
    # and read by this very call, so its lineage must actually be in it.
    assert "x" in captured_kwargs["variable_lineage"]


def test_a_no_cache_statement_never_reaches_the_gate(magics_fixture, monkeypatch):
    """``# @cash:no-cache`` must short-circuit BEFORE the gate is ever built,
    not merely produce the same observable result because
    ``decide_cacheability``'s own no-cache short-circuit happens to refuse
    the call downstream too (CAS-243 task 10 review I3).

    There are genuinely two independent enforcement layers for "no-cache
    wins over interception": the outer clause in
    ``_code_and_tree_for_execution`` (this task's own code, checked here),
    and ``cacheability_decision.decide_cacheability``'s pre-existing
    ``annotation.no_cache`` short-circuit, reached transitively through
    ``call_site_is_cacheable``. An integration test that only counts
    ``compute()`` executions cannot tell them apart -- both produce the
    identical byte-for-byte outcome (nothing wrapped, nothing cached), so a
    regression that deletes JUST the outer clause is invisible to a call-count
    assertion; the inner layer silently absorbs it. Spying on
    ``call_site_is_cacheable`` itself is the only way to observe that the
    outer clause did its job by never letting ``wrap_eligible_calls`` reach
    the gate at all, rather than reaching it and having it refuse.
    """
    import cash.notebook.statement.processor as processor_module

    captured_calls = []
    real = processor_module.call_site_is_cacheable

    def _spy(call, **kwargs):
        captured_calls.append(call)
        return real(call, **kwargs)

    monkeypatch.setattr(processor_module, "call_site_is_cacheable", _spy)

    magics, shell, _, _ = magics_fixture
    magics.cash("", "def compute(x):\n    return x + 1\nout = []\nx = 1")
    magics.cash("", "# @cash:no-cache\nout.append(compute(x))")

    assert captured_calls == [], (
        "the gate was invoked for a no-cache statement -- "
        "_code_and_tree_for_execution's outer no_cache clause must return "
        "before wrap_eligible_calls ever calls the gate, not rely on "
        "decide_cacheability's downstream no-cache check to save it: "
        f"{captured_calls!r}"
    )


def test_identity_contract_holds_on_the_no_eligible_call_and_opt_out_branches(magics_fixture):
    """FINDING 7 (notebook-annotation-visibility Task 1, review round 1).

    ``process_statement``/``process_statement_async`` forward the cell's
    original text for compilation (``exec_source``) ONLY when this method
    made no change: ``_exec_source = exec_source if _exec_code is code else
    None``. That guard is correct only because every opt-out /
    no-eligible-call / failure branch below returns the SAME ``code`` object
    it was given, never an equal-but-freshly-built string -- ``is``, not
    ``==``. Removing the guard entirely fails 7 integration tests, so the
    guard's *effect* is covered, but nothing previously named this identity
    *contract* directly: a future refactor that returned, say,
    ``code.strip()`` or ``ast.unparse(tree)`` on one of these branches would
    be value-equal (breaking no existing assertion) while silently
    disabling ``exec_source`` forwarding for every ordinary statement in
    that branch, forever falling back to the unparsed form.

    Pinned directly at the two branches ``exec_source`` forwarding actually
    depends on, with a positive control proving this is not vacuous: the
    SAME statement, with the opt-out lifted, produces a genuinely NEW string
    once ``wrap_eligible_calls`` actually rewrites something.
    """
    import ast
    import types

    magics, shell, _, _ = magics_fixture
    magics.cash("", "out = []")
    processor = magics._statement_processor

    # Branch 1: no eligible call anywhere in the statement.
    code = "out.append(1)"
    tree = ast.parse(code)
    result_code, result_tree = processor._code_and_tree_for_execution(code, tree, None)
    assert result_code is code, "the no-eligible-call branch built a NEW string object"
    assert result_tree is tree, "the no-eligible-call branch built a NEW tree object"

    # Branch 2: the `no_cache_calls` opt-out, short-circuiting BEFORE
    # `wrap_eligible_calls` runs at all -- even though `compute(x)` below is
    # structurally eligible (proven by the control that follows).
    magics.cash("", "def compute(x):\n    return x + 1\nx = 1")
    code = "out.append(compute(x))"
    tree = ast.parse(code)
    opted_out = types.SimpleNamespace(no_cache_calls=True, no_cache=False)
    result_code, result_tree = processor._code_and_tree_for_execution(code, tree, opted_out)
    assert result_code is code, "the no_cache_calls opt-out branch built a NEW string object"
    assert result_tree is tree, "the no_cache_calls opt-out branch built a NEW tree object"

    # Control: the identical statement, opt-out lifted, DOES get a new
    # string -- proving the two assertions above are discriminating, not
    # trivially true because nothing here is ever eligible for rewriting.
    not_opted_out = types.SimpleNamespace(no_cache_calls=False, no_cache=False)
    control_code, _ = processor._code_and_tree_for_execution(code, tree, not_opted_out)
    assert control_code is not code, (
        "the control did not rewrite -- these statements are not actually "
        "exercising the branches this test claims to pin"
    )
