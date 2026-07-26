"""Which call nodes may be cached independently of their statement (CAS-243).

One structural rule decides it: **a call is eligible when its free variables do
not include the statement's assignment / mutation target.** If the call reads
the target, it *is* the fold and there is nothing order-independent to extract.

The rule is doing more work than it looks like. In ``out.append(compute(x))``
the *append* is itself a Call whose func reads ``out`` — the target — so it is
excluded by the same rule that admits ``compute(x)``. No special case needed
for "don't cache the mutation itself".
"""

import ast
import unittest

from cash.notebook.call_interception import eligible_call_nodes


def _calls(src: str) -> list[str]:
    """Return the source of each eligible call in the first statement of *src*."""
    stmt = ast.parse(src).body[0]
    return [ast.unparse(node) for node in eligible_call_nodes(stmt)]


class TestEligibility(unittest.TestCase):

    def test_accumulator_fold_extracts_the_call(self):
        """``s += compute(x)`` — compute reads x, not s."""
        self.assertEqual(_calls("s += compute(x)"), ["compute(x)"])

    def test_append_extracts_the_argument_not_the_append(self):
        """``out.append(compute(x))`` — the append reads `out` and is excluded."""
        self.assertEqual(_calls("out.append(compute(x))"), ["compute(x)"])

    def test_subscript_store_extracts_the_call(self):
        """``prices[t] = compute(t)`` — target base is `prices`."""
        self.assertEqual(_calls("prices[t] = compute(t)"), ["compute(t)"])

    def test_fold_reading_the_accumulator_is_excluded(self):
        """``s = merge(s, x)`` — the call reads `s`; it IS the fold."""
        self.assertEqual(_calls("s = merge(s, x)"), [])

    def test_inplace_method_on_the_target_is_excluded(self):
        """``df.sort_values(inplace=True)`` — the mutation is the expensive part."""
        self.assertEqual(_calls("df.sort_values(inplace=True)"), [])

    def test_plain_assignment_extracts_the_call(self):
        """``y = compute(x)`` is structurally eligible.

        Whether it is *worth* intercepting is the processor's call — statement
        caching already covers this shape. This unit answers eligibility only.
        """
        self.assertEqual(_calls("y = compute(x)"), ["compute(x)"])

    def test_only_the_outermost_eligible_call_is_returned(self):
        """Nested eligible calls must not both be intercepted.

        Intercepting `f(...)` already covers `g(x)` inside it; returning both
        would mint two keys for one piece of work and cache the inner result
        redundantly.
        """
        self.assertEqual(_calls("out.append(f(g(x)))"), ["f(g(x))"])

    def test_multiple_independent_calls_are_all_returned(self):
        """Siblings are separate work and each deserves its own entry."""
        self.assertEqual(
            _calls("out.append(f(x) + g(y))"), ["f(x)", "g(y)"]
        )

    def test_call_reading_the_target_deep_inside_is_excluded(self):
        """The target check must see through attributes and subscripts."""
        self.assertEqual(_calls("s += compute(s.total)"), [])
        self.assertEqual(_calls("s += compute(s[0])"), [])

    def test_no_calls_yields_nothing(self):
        self.assertEqual(_calls("s += 1"), [])

    def test_compound_statements_are_not_searched(self):
        """A whole loop or branch is out of scope, and that is a safety rule.

        Cash can execute a loop as ONE unit, in which case the statement handed
        here is the ``ast.For`` itself. Its target set is empty — there is no
        assignment target on a `for` node — so the free-variable rule excludes
        nothing and *every* call in the body would look eligible, including the
        side-effecting one the loop exists to perform:

            for x in xs:
                log_it(x)        # would be cached -> the log is skipped on re-run

        Per-iteration decomposition already hands each body statement here
        individually, with its own target, which is where the rule is sound. So
        compound statements are declined outright rather than searched with a
        rule that does not apply to them.
        """
        self.assertEqual(_calls("for x in xs:\n    log_it(x)"), [])
        self.assertEqual(_calls("for x in xs:\n    s += compute(x)"), [])
        self.assertEqual(_calls("if flag:\n    out.append(compute(x))"), [])
        self.assertEqual(_calls("while go:\n    emit(x)"), [])
        self.assertEqual(_calls("with open(p) as fh:\n    data = parse(fh)"), [])

    def test_function_and_class_definitions_are_not_searched(self):
        """A def's body runs later, under its own statement, not here."""
        self.assertEqual(_calls("def f():\n    return compute(1)"), [])
        self.assertEqual(_calls("class C:\n    x = compute(1)"), [])


if __name__ == "__main__":
    unittest.main()
