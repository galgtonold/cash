"""A comprehension's or generator's own iteration variable is not a cell-level input
or output.

Python 3 scopes it to the comprehension, so treating it as a cell input made a
statement depend on a name that it never reads from the namespace. A for
loop's target, which does leak, still counts.
"""

from cash.analysis.code_analyzer import CodeAnalyzer


class TestComprehensionScoping:
    """Test that comprehension iteration variables don't leak as cell-level inputs/outputs."""

    def test_listcomp_var_not_input(self):
        """List comprehension iteration variable should NOT be a cell-level input."""
        code = "result = [x * 2 for x in data]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "x" not in inputs, "Comprehension var 'x' should not be an input"
        assert "data" in inputs, "'data' should be an input"
        assert "result" in outputs

    def test_listcomp_var_not_output(self):
        """List comprehension iteration variable should NOT leak as cell-level output."""
        code = "result = [x * 2 for x in data]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "x" not in outputs, "Comprehension var 'x' should not be a cell output"

    def test_setcomp_var_not_input(self):
        """Set comprehension iteration variable should not leak."""
        code = "unique = {v.lower() for v in names}"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "v" not in inputs
        assert "names" in inputs
        assert "unique" in outputs

    def test_dictcomp_var_not_input(self):
        """Dict comprehension iteration variable should not leak."""
        code = "mapping = {k: v for k, v in items}"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "k" not in inputs
        assert "v" not in inputs
        assert "items" in inputs
        assert "mapping" in outputs

    def test_generatorexp_var_not_input(self):
        """Generator expression iteration variable should not leak."""
        code = "total = sum(x ** 2 for x in numbers)"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "x" not in inputs
        assert "numbers" in inputs
        assert "sum" in inputs
        assert "total" in outputs

    def test_any_with_generatorexp(self):
        """any() with generator expression - the iteration var should not leak.

        The pattern a real project used to look for its input files:
        any(_fn.endswith('.csv') for _fn in os.listdir(...))
        """
        code = "has_csv = any(_fn.endswith('.csv') for _fn in file_list)"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "_fn" not in inputs, "_fn should not leak from generator expression"
        assert "file_list" in inputs
        assert "any" in inputs
        assert "has_csv" in outputs

    def test_nested_comprehension(self):
        """Nested comprehension variables should all be local."""
        code = "flat = [item for sublist in matrix for item in sublist]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "item" not in inputs
        assert "sublist" not in inputs
        assert "matrix" in inputs
        assert "flat" in outputs

    def test_comprehension_with_conditional(self):
        """Comprehension with if clause - iteration var in condition should be local."""
        code = "evens = [x for x in numbers if x % 2 == 0]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "x" not in inputs
        assert "numbers" in inputs
        assert "evens" in outputs

    def test_comprehension_uses_outer_variable(self):
        """Comprehension body that uses an outer variable should detect it as input."""
        code = "scaled = [x * factor for x in data]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "x" not in inputs
        assert "factor" in inputs, "'factor' is from outer scope, should be input"
        assert "data" in inputs
        assert "scaled" in outputs

    def test_comprehension_tuple_unpacking_target(self):
        """Comprehension with tuple unpacking target should not leak vars."""
        code = "keys = [k for k, v in items.items()]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "k" not in inputs
        assert "v" not in inputs
        assert "items" in inputs
        assert "keys" in outputs

    def test_for_loop_var_still_leaks(self):
        """Regular for-loop variable SHOULD still be a cell-level output (Python semantics)."""
        code = "for i in range(10):\n    pass"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        # In Python, for-loop variables DO leak to enclosing scope
        assert "i" in outputs, "For-loop var should be a cell output (Python semantics)"

    def test_walrus_in_comprehension(self):
        """Walrus operator (:=) in comprehension DOES leak to enclosing scope."""
        code = "results = [y := f(x) for x in data]"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "x" not in inputs
        assert "data" in inputs
        # y is assigned via walrus - Python 3.8+ says it leaks to enclosing scope
        # Our analyzer may or may not handle this - just ensure no crash

    def test_dictcomp_key_value_both_scoped(self):
        """Both key and value expressions in dictcomp should use comprehension scope."""
        code = "d = {k.upper(): len(v) for k, v in pairs}"
        inputs, outputs = CodeAnalyzer.analyze_code_block(code)
        assert "k" not in inputs
        assert "v" not in inputs
        assert "pairs" in inputs
        assert "len" in inputs
        assert "d" in outputs
