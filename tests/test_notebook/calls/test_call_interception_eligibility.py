"""Which call nodes may be cached independently of their statement.

One structural rule decides it: **a call is eligible unless its callee reads
the statement's assignment / mutation target.** A target the call is only
handed (``df.insert(1, "c", slow(df))``) is keyed on the value it holds when
the call runs, as a comprehension's variable is.

The rule is doing more work than it looks like. In ``out.append(compute(x))``
the *append* is itself a Call whose callee reads ``out`` — the target — so it
is excluded by the same rule that admits ``compute(x)``. No special case needed
for "don't cache the mutation itself".
"""

import ast

from cash.notebook.call_interception import _eligible_calls_in_scope, eligible_call_nodes


def _calls(src: str) -> list[str]:
    """Return the source of each eligible call in the first statement of *src*."""
    stmt = ast.parse(src).body[0]
    return [ast.unparse(node) for node in eligible_call_nodes(stmt)]


class TestEligibility:
    def test_accumulator_fold_extracts_the_call(self):
        """``s += compute(x)`` — compute reads x, not s."""
        assert _calls("s += compute(x)") == ["compute(x)"]

    def test_append_extracts_the_argument_not_the_append(self):
        """``out.append(compute(x))`` — the append reads `out` and is excluded."""
        assert _calls("out.append(compute(x))") == ["compute(x)"]

    def test_subscript_store_extracts_the_call(self):
        """``prices[t] = compute(t)`` — target base is `prices`."""
        assert _calls("prices[t] = compute(t)") == ["compute(t)"]

    def test_a_call_handed_the_target_is_keyed_on_its_value(self):
        """``s = merge(s, x)`` — `s` is handed in, so its value is in the key."""
        stmt = ast.parse("s = merge(s, x)").body[0]
        [(call, local)] = _eligible_calls_in_scope(stmt)
        assert ast.unparse(call) == "merge(s, x)"
        assert local == {"s"}

    def test_inplace_method_on_the_target_is_excluded(self):
        """``df.sort_values(inplace=True)`` — the mutation is the expensive part."""
        assert _calls("df.sort_values(inplace=True)") == []

    def test_plain_assignment_extracts_the_call(self):
        """``y = compute(x)`` is structurally eligible.

        Whether it is *worth* intercepting is the processor's call — statement
        caching already covers this shape. This unit answers eligibility only.
        """
        assert _calls("y = compute(x)") == ["compute(x)"]

    def test_a_nested_call_is_returned_after_the_outer_one(self):
        """An argument runs whether the outer call hits or not.

        Wrapping a call replaces its callee expression only, so ``g(x)``
        executes either way: taking it as well is the only way its work is
        ever reused (nine fits nested in another call once re-ran every
        time). The outer call comes first.
        """
        assert _calls("out.append(f(g(x)))") == ["f(g(x))", "g(x)"]

    def test_a_nested_call_handed_the_target_is_taken(self):
        assert _calls("out.append(f(g(out)))") == ["f(g(out))", "g(out)"]

    def test_a_call_whose_callee_reads_the_target_is_excluded(self):
        assert _calls("df['c'] = df.apply(f)") == []
        assert _calls("df['c'] = df['a'].map(f)") == []

    def test_multiple_independent_calls_are_all_returned(self):
        """Siblings are separate work and each deserves its own entry."""
        assert _calls("out.append(f(x) + g(y))") == ["f(x)", "g(y)"]

    def test_call_reading_the_target_deep_inside_keys_it_by_value(self):
        """The target check sees through attributes and subscripts."""
        for src in ("s += compute(s.total)", "s += compute(s[0])"):
            [(_call, local)] = _eligible_calls_in_scope(ast.parse(src).body[0])
            assert local == {"s"}

    def test_no_calls_yields_nothing(self):
        assert _calls("s += 1") == []

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
        assert _calls("for x in xs:\n    log_it(x)") == []
        assert _calls("for x in xs:\n    s += compute(x)") == []
        assert _calls("if flag:\n    out.append(compute(x))") == []
        assert _calls("while go:\n    emit(x)") == []
        assert _calls("with open(p) as fh:\n    data = parse(fh)") == []

    def test_function_and_class_definitions_are_not_searched(self):
        """A def's body runs later, under its own statement, not here."""
        assert _calls("def f():\n    return compute(1)") == []
        assert _calls("class C:\n    x = compute(1)") == []
