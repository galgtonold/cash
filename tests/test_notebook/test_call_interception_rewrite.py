"""Rewriting eligible calls into cached calls (CAS-243).

The transform wraps the **callee expression**, leaving the argument list
untouched::

    compute(x)  ->  __cash_call__(compute)(x)

That choice is the whole design. The obvious alternative — hoisting the
sub-expression into a temp before the statement — is wrong: it makes the call
run unconditionally, so ``f() or g()`` would evaluate ``g()`` even when ``f()``
is truthy, firing its side effects, raising exceptions that never occurred, and
paying its cost on a first run that previously skipped it.

Wrapping the callee is lazy and in position, so short-circuiting, ternaries,
comprehension scopes and argument unpacking all keep working for free — Python
decides whether the call is reached, exactly as before.
"""

import ast
import unittest

from cash.notebook.call_interception import CallSite, HELPER_NAME, wrap_eligible_calls


def _rewrite(src: str) -> tuple[str, list[CallSite]]:
    tree = ast.parse(src)
    new_tree, sites = wrap_eligible_calls(tree)
    return ast.unparse(new_tree), sites


class TestRewrite(unittest.TestCase):

    def test_helper_name_is_dunder_private(self):
        """The injected name must not collide with anything a user would write."""
        self.assertTrue(HELPER_NAME.startswith("__cash"))

    def test_accumulator_fold(self):
        out, sites = _rewrite("s += compute(x)")
        self.assertEqual(out, "s += __cash_call__(compute, 0)(x)")
        self.assertEqual(len(sites), 1)

    def test_append(self):
        out, sites = _rewrite("out.append(compute(x))")
        self.assertEqual(out, "out.append(__cash_call__(compute, 0)(x))")
        self.assertEqual(len(sites), 1)

    def test_ineligible_statement_is_untouched(self):
        out, sites = _rewrite("s = merge(s, x)")
        self.assertEqual(out, "s = merge(s, x)")
        self.assertEqual(len(sites), 0)

    def test_argument_unpacking_survives(self):
        """Arguments are not rewritten, so *args/**kwargs need no special case."""
        out, sites = _rewrite("out.append(compute(*xs, k=1, **kw))")
        self.assertEqual(out, "out.append(__cash_call__(compute, 0)(*xs, k=1, **kw))")
        self.assertEqual(len(sites), 1)

    def test_short_circuit_is_preserved(self):
        """The load-bearing case: `g` stays under the `or`, not hoisted above it."""
        out, sites = _rewrite("out.append(f() or g())")
        self.assertEqual(
            out, "out.append(__cash_call__(f, 0)() or __cash_call__(g, 1)())"
        )
        self.assertEqual(len(sites), 2)

    def test_method_call_callee_is_wrapped_whole(self):
        """A bound method is a callable like any other; wrap the whole callee."""
        out, sites = _rewrite("s += model.predict(x)")
        self.assertEqual(out, "s += __cash_call__(model.predict, 0)(x)")
        self.assertEqual(len(sites), 1)

    def test_original_tree_is_not_mutated(self):
        """The caller keeps using the original tree for analysis and keying.

        Rewriting in place would change the source the cache key is built from
        and desync the runtime from the upstream simulator.
        """
        tree = ast.parse("s += compute(x)")
        before = ast.unparse(tree)
        wrap_eligible_calls(tree)
        self.assertEqual(ast.unparse(tree), before)

    def test_rewritten_tree_compiles(self):
        """Locations must be fixed up or compile() rejects the new nodes."""
        tree = ast.parse("out.append(compute(x))")
        new_tree, _ = wrap_eligible_calls(tree)
        compile(new_tree, "<test>", "exec")  # must not raise


def test_rewrite_emits_site_index_and_table():
    tree = ast.parse("out.append(compute(x))")
    rewritten, sites = wrap_eligible_calls(tree)

    assert len(sites) == 1
    assert sites[0].source == "compute(x)"
    assert sites[0].free_names == frozenset({"compute", "x"})
    assert sites[0].occurrence_index == 0
    assert "__cash_call__(compute, 0)(x)" in ast.unparse(rewritten)


def test_identical_call_sites_get_distinct_occurrence_indices():
    tree = ast.parse("a = compute(x)\nb = compute(x)")
    _, sites = wrap_eligible_calls(tree)

    assert [s.occurrence_index for s in sites] == [0, 1]
    assert sites[0].source == sites[1].source == "compute(x)"


def test_distinct_sources_each_start_at_occurrence_zero():
    tree = ast.parse("a = compute(x)\nb = other(x)")
    _, sites = wrap_eligible_calls(tree)

    assert [s.occurrence_index for s in sites] == [0, 0]


if __name__ == "__main__":
    unittest.main()
