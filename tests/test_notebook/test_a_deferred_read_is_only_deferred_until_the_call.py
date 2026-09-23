"""What counts as a module-level read, for the forward-reference guard.

The guard refuses a cell that reads, at module level, a name
only a LATER cell binds -- and it was written on the premise that "a name
inside a function body resolves at call time, so it is not a module-level
read". That premise is true and it is not the condition. It only protects the
cell if the call happens AFTER the binding.

Two shapes slipped through, both from a real notebook, both of
which ran clean under cash and raise ``NameError`` from the top:

    out = s.map(lambda v: f(len(v)))     # the lambda is called by .map, NOW
    def use_it(): return f(21)
    print(use_it())                      # called in this very cell, NOW

So the rule is not "a function body is deferred" but "deferred only until the
call". A lambda handed to a call is invoked inside that statement; a function
defined and called in the same cell is invoked inside that cell.

The controls are the point of this file. The guard RAISES, so a false positive
stops a cell that works -- which already happened once: judging on the
statement's inputs (which include deferred free names) refused
``test_downward_function_dependency``'s notebook, which runs fine from the
top. A helper that names something bound below and is called later must stay
allowed, and so must a lambda that is stored rather than invoked.
"""

from __future__ import annotations

from cash.notebook.upstream.checker import UpstreamChecker

reads = UpstreamChecker._module_level_reads


class TestReadNow:
    """Shapes that really do read the name while the cell runs."""

    def test_a_bare_read(self):
        assert "f" in reads("x = f(1)")

    def test_a_comprehension(self):
        assert "f" in reads("xs = [f(i) for i in range(3)]")

    def test_a_decorator(self):
        assert "deco" in reads("@deco\ndef g():\n    return 1")

    def test_a_default_argument(self):
        assert "seed" in reads("def g(a=seed):\n    return a")

    def test_a_base_class(self):
        assert "Base" in reads("class C(Base):\n    pass")

    def test_a_lambda_passed_to_a_call(self):
        """The reported shape: `.map` invokes the lambda inside this statement."""
        assert "f" in reads("out = s.map(lambda v: f(v))")

    def test_a_lambda_passed_as_a_keyword(self):
        assert "rank" in reads("ys = sorted(xs, key=lambda v: rank(v))")

    def test_a_function_defined_and_called_in_the_same_cell(self):
        """The other reported shape: `use_it()` runs before the cell is over."""
        assert "f" in reads("def use_it():\n    return f(21)\nprint(use_it())")


class TestDeferred:
    """Shapes whose lookup really is postponed. Refusing these breaks
    notebooks that run correctly from the top."""

    def test_a_helper_that_is_not_called_here(self):
        """`test_downward_function_dependency`'s shape, which this guard
        already refused once. `b` is bound in a later cell; `a` is called in a
        later cell still, by which time `b` exists."""
        assert "b" not in reads("def a(n):\n    return b(n) * 2")

    def test_a_stored_lambda(self):
        """Put in a dict for later dispatch, not invoked now."""
        assert "f" not in reads("handlers = {'x': lambda: f()}")

    def test_a_lambda_bound_to_a_name(self):
        assert "f" not in reads("g = lambda: f()")

    def test_a_method_body(self):
        assert "helper" not in reads("class C:\n    def m(self):\n        return helper()")

    def test_a_function_called_only_from_another_uncalled_function(self):
        """`inner` is never reached while this cell runs."""
        assert "f" not in reads("def outer():\n    return inner()\ndef inner():\n    return f()")


class TestItNeverGuesses:
    def test_a_cell_that_does_not_parse_yields_none(self):
        assert reads("x = (") is None

    def test_a_lambdas_own_parameter_is_not_a_read(self):
        """`v` is bound by the lambda; only `f` comes from outside."""
        got = reads("out = s.map(lambda v: f(v))")
        assert "f" in got and "v" not in got

    def test_a_called_functions_own_parameter_is_not_a_read(self):
        got = reads("def g(n):\n    return f(n)\ng(1)")
        assert "f" in got and "n" not in got
