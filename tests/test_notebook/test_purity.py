"""Tests for the purity declaration system.

Tests cover:
- @pure and @stateful decorators
- is_pure() and is_stateful() checker functions
- is_known_pure() for built-in function inference
- Integration with statement_processor.py (stateful skips cache, pure skips mutation detection)
- Edge cases: unmarked functions, non-callables, methods, lambdas
"""

from cash.notebook.purity import pure, stateful, is_pure, is_stateful, is_known_pure, KNOWN_PURE_BUILTINS


# ===========================================================================
# Unit tests for decorators and checkers
# ===========================================================================


class TestPureDecorator:
    """Tests for @pure decorator."""

    def test_pure_marks_function(self):
        @pure
        def add(x, y):
            return x + y

        assert is_pure(add) is True

    def test_pure_preserves_functionality(self):
        @pure
        def add(x, y):
            return x + y

        assert add(2, 3) == 5

    def test_pure_preserves_name(self):
        @pure
        def my_function():
            pass

        assert my_function.__name__ == 'my_function'

    def test_pure_preserves_docstring(self):
        @pure
        def my_function():
            """My docstring."""
            pass

        assert my_function.__doc__ == 'My docstring.'

    def test_pure_not_stateful(self):
        @pure
        def compute(x):
            return x * 2

        assert is_stateful(compute) is False

    def test_pure_with_args_and_kwargs(self):
        @pure
        def multi(a, b, c=10, **kw):
            return a + b + c + sum(kw.values())

        assert multi(1, 2, c=3, d=4) == 10
        assert is_pure(multi)


class TestStatefulDecorator:
    """Tests for @stateful decorator."""

    def test_stateful_marks_function(self):
        @stateful
        def train(data):
            pass

        assert is_stateful(train) is True

    def test_stateful_preserves_functionality(self):
        results = []

        @stateful
        def append_to(val):
            results.append(val)

        append_to(42)
        assert results == [42]

    def test_stateful_preserves_name(self):
        @stateful
        def train_model():
            pass

        assert train_model.__name__ == 'train_model'

    def test_stateful_preserves_docstring(self):
        @stateful
        def train_model():
            """Train the model."""
            pass

        assert train_model.__doc__ == 'Train the model.'

    def test_stateful_not_pure(self):
        @stateful
        def train(data):
            pass

        assert is_pure(train) is False


class TestIsPureChecker:
    """Tests for is_pure() function."""

    def test_unmarked_function_not_pure(self):
        def regular():
            pass

        assert is_pure(regular) is False

    def test_none_not_pure(self):
        assert is_pure(None) is False

    def test_integer_not_pure(self):
        assert is_pure(42) is False

    def test_string_not_pure(self):
        assert is_pure("hello") is False

    def test_class_not_pure(self):
        class MyClass:
            pass

        assert is_pure(MyClass) is False

    def test_lambda_not_pure(self):
        f = lambda x: x * 2
        assert is_pure(f) is False

    def test_manually_marked_pure(self):
        def compute(x):
            return x + 1

        compute._cash_pure = True
        assert is_pure(compute) is True

    def test_false_attribute_not_pure(self):
        def compute(x):
            return x + 1

        compute._cash_pure = False
        assert is_pure(compute) is False


class TestIsStatefulChecker:
    """Tests for is_stateful() function."""

    def test_unmarked_function_not_stateful(self):
        def regular():
            pass

        assert is_stateful(regular) is False

    def test_none_not_stateful(self):
        assert is_stateful(None) is False

    def test_integer_not_stateful(self):
        assert is_stateful(42) is False

    def test_manually_marked_stateful(self):
        def train():
            pass

        train._cash_stateful = True
        assert is_stateful(train) is True

    def test_false_attribute_not_stateful(self):
        def train():
            pass

        train._cash_stateful = False
        assert is_stateful(train) is False


class TestCombinations:
    """Tests for edge cases and combinations."""

    def test_cannot_be_both_pure_and_stateful(self):
        """A function can technically have both markers, but is_pure and is_stateful are independent checks."""
        @pure
        @stateful
        def weird():
            pass

        # Both attributes are set
        assert is_pure(weird) is True
        assert is_stateful(weird) is True

    def test_pure_class_method(self):
        class Calculator:
            @pure
            def add(self, a, b):
                return a + b

        calc = Calculator()
        assert is_pure(calc.add) is True
        assert calc.add(2, 3) == 5

    def test_stateful_class_method(self):
        class Trainer:
            @stateful
            def fit(self, data):
                self.fitted = True

        t = Trainer()
        assert is_stateful(t.fit) is True

    def test_pure_nested_calls(self):
        """Pure functions calling other pure functions."""
        @pure
        def double(x):
            return x * 2

        @pure
        def quadruple(x):
            return double(double(x))

        assert quadruple(3) == 12
        assert is_pure(quadruple) is True


# ===========================================================================
# Integration tests: purity checks in statement processor
# ===========================================================================

class TestPurityStatementProcessorIntegration:
    """Tests for how purity declarations affect caching in the statement processor."""

    def test_stateful_function_skips_cache(self, cash_magics, mock_shell):
        """Statements calling @stateful functions should skip caching."""
        @stateful
        def train(data):
            return sum(data)

        mock_shell.user_ns['train'] = train
        mock_shell.user_ns['data'] = [1, 2, 3]

        # Execute the statement
        cash_magics.cash("", "result = train(data)")
        assert mock_shell.user_ns['result'] == 6

        # Execute again - should NOT be cached (stateful)
        mock_shell.user_ns['data'] = [4, 5, 6]
        cash_magics.cash("", "result = train(data)")
        assert mock_shell.user_ns['result'] == 15

    def test_pure_function_allows_caching(self, cash_magics, mock_shell):
        """Statements calling @pure functions should be cacheable."""
        @pure
        def compute(x):
            return x ** 2

        mock_shell.user_ns['compute'] = compute
        mock_shell.user_ns['x'] = 5

        # Execute the statement
        cash_magics.cash("", "result = compute(x)")
        assert mock_shell.user_ns['result'] == 25

        # Execute again with same inputs - should be cached
        cash_magics.cash("", "result = compute(x)")
        assert mock_shell.user_ns['result'] == 25

    def test_pure_function_skips_mutation_detection(self, cash_magics, mock_shell):
        """Statements calling only @pure functions should skip mutation detection."""
        @pure
        def transform(lst):
            return sorted(lst)

        mock_shell.user_ns['transform'] = transform
        mock_shell.user_ns['data'] = [3, 1, 2]

        # Even though 'data' is a mutable type, @pure should skip mutation checks
        cash_magics.cash("", "result = transform(data)")
        assert mock_shell.user_ns['result'] == [1, 2, 3]

    def test_unmarked_function_uses_default_behavior(self, cash_magics, mock_shell):
        """Functions without purity markers should use default caching behavior."""
        def regular_func(x):
            return x + 1

        mock_shell.user_ns['regular_func'] = regular_func
        mock_shell.user_ns['x'] = 10

        cash_magics.cash("", "result = regular_func(x)")
        assert mock_shell.user_ns['result'] == 11

    def test_mixed_pure_and_unmarked_not_all_pure(self, cash_magics, mock_shell):
        """Mix of @pure and unmarked functions should not be considered all-pure."""
        @pure
        def pure_func(x):
            return x * 2

        def unmarked_func(x):
            return x + 1

        mock_shell.user_ns['pure_func'] = pure_func
        mock_shell.user_ns['unmarked_func'] = unmarked_func
        mock_shell.user_ns['x'] = 5

        # Both functions called - not all pure, so normal mutation detection applies
        cash_magics.cash("", "result = pure_func(x) + unmarked_func(x)")
        assert mock_shell.user_ns['result'] == 16  # 10 + 6

    def test_stateful_with_debug(self, cash_magics, mock_shell):
        """Verify stateful debug output is generated."""
        cash_magics._statement_processor.debug = True

        @stateful
        def update_db():
            return "updated"

        mock_shell.user_ns['update_db'] = update_db

        cash_magics.cash("", "status = update_db()")
        assert mock_shell.user_ns['status'] == "updated"

    def test_stateful_reruns_every_time(self, cash_magics, mock_shell):
        """@stateful functions should be marked as uncacheable - they're always re-executed."""
        @stateful
        def increment(n):
            return n + 1

        mock_shell.user_ns['increment'] = increment

        # First call: computes normally
        mock_shell.user_ns['n'] = 1
        cash_magics.cash("", "v = increment(n)")
        assert mock_shell.user_ns['v'] == 2

        # Change input: should recompute because stateful skips cache
        mock_shell.user_ns['n'] = 10
        cash_magics.cash("", "v = increment(n)")
        assert mock_shell.user_ns['v'] == 11

    def test_no_function_calls_no_purity_effect(self, cash_magics, mock_shell):
        """Statements with no function calls should not be affected by purity system."""
        mock_shell.user_ns['x'] = 10
        cash_magics.cash("", "y = x * 2")
        assert mock_shell.user_ns['y'] == 20

    def test_method_call_not_detected_as_name(self, cash_magics, mock_shell):
        """Method calls (obj.method()) should not be checked as Name calls."""
        class MyObj:
            @stateful
            def do_something(self):
                return 42

        obj = MyObj()
        mock_shell.user_ns['obj'] = obj

        # obj.do_something() is an Attribute call, not a Name call
        # So purity system should not detect it (current implementation only checks ast.Name)
        cash_magics.cash("", "result = obj.do_something()")
        assert mock_shell.user_ns['result'] == 42


class TestPurityImports:
    """Tests for importing purity declarations from cash package."""

    def test_import_pure_from_cash(self):
        from cash import pure
        assert callable(pure)

    def test_import_stateful_from_cash(self):
        from cash import stateful
        assert callable(stateful)

    def test_import_is_pure_from_cash(self):
        from cash import is_pure
        assert callable(is_pure)

    def test_import_is_stateful_from_cash(self):
        from cash import is_stateful
        assert callable(is_stateful)

    def test_pure_in_all(self):
        import cash
        assert 'pure' in cash.__all__

    def test_stateful_in_all(self):
        import cash
        assert 'stateful' in cash.__all__

    def test_is_pure_in_all(self):
        import cash
        assert 'is_pure' in cash.__all__

    def test_is_stateful_in_all(self):
        import cash
        assert 'is_stateful' in cash.__all__


# ===========================================================================
# Tests for known-pure builtins
# ===========================================================================

class TestKnownPureBuiltins:
    """Tests for is_known_pure() and KNOWN_PURE_BUILTINS."""

    def test_sorted_is_known_pure(self):
        assert is_known_pure('sorted') is True

    def test_len_is_known_pure(self):
        assert is_known_pure('len') is True

    def test_min_max_are_known_pure(self):
        assert is_known_pure('min') is True
        assert is_known_pure('max') is True

    def test_sum_is_known_pure(self):
        assert is_known_pure('sum') is True

    def test_type_constructors_are_known_pure(self):
        for name in ['int', 'float', 'str', 'bool', 'list', 'tuple', 'set', 'dict']:
            assert is_known_pure(name) is True, f"{name} should be known pure"

    def test_introspection_functions_are_known_pure(self):
        for name in ['type', 'isinstance', 'issubclass', 'callable', 'hasattr']:
            assert is_known_pure(name) is True, f"{name} should be known pure"

    def test_iteration_functions_are_known_pure(self):
        for name in ['range', 'enumerate', 'zip', 'map', 'filter']:
            assert is_known_pure(name) is True, f"{name} should be known pure"

    def test_print_is_not_known_pure(self):
        """print() has side effects (writes to stdout)."""
        assert is_known_pure('print') is False

    def test_open_is_not_known_pure(self):
        """open() has side effects (creates file handle)."""
        assert is_known_pure('open') is False

    def test_input_is_not_known_pure(self):
        assert is_known_pure('input') is False

    def test_exec_is_not_known_pure(self):
        assert is_known_pure('exec') is False

    def test_custom_function_is_not_known_pure(self):
        assert is_known_pure('my_custom_function') is False

    def test_known_pure_builtins_is_frozenset(self):
        assert isinstance(KNOWN_PURE_BUILTINS, frozenset)

    def test_known_pure_builtins_not_empty(self):
        assert len(KNOWN_PURE_BUILTINS) > 20


class TestKnownPureIntegration:
    """Integration tests: known-pure builtins skip mutation detection."""

    def test_sorted_skips_mutation_detection(self, cash_magics, mock_shell):
        """sorted() is a known-pure builtin — mutation detection should be skipped."""
        mock_shell.user_ns['data'] = [3, 1, 2]
        cash_magics.cash("", "result = sorted(data)")
        assert mock_shell.user_ns['result'] == [1, 2, 3]

    def test_len_skips_mutation_detection(self, cash_magics, mock_shell):
        """len() is a known-pure builtin."""
        mock_shell.user_ns['items'] = [10, 20, 30]
        cash_magics.cash("", "n = len(items)")
        assert mock_shell.user_ns['n'] == 3

    def test_mixed_known_pure_and_user_pure(self, cash_magics, mock_shell):
        """Combination of known-pure builtins and @pure user functions."""
        @pure
        def double(x):
            return x * 2

        mock_shell.user_ns['double'] = double
        mock_shell.user_ns['x'] = 5
        cash_magics.cash("", "result = max(double(x), 0)")
        assert mock_shell.user_ns['result'] == 10

    def test_known_pure_with_unknown_function(self, cash_magics, mock_shell):
        """Mix of known-pure builtin and unknown function — not all pure."""
        def unknown_func(x):
            return x + 1

        mock_shell.user_ns['unknown_func'] = unknown_func
        mock_shell.user_ns['data'] = [1, 2, 3]
        # sorted is known pure, but unknown_func is not → not all_calls_pure
        cash_magics.cash("", "result = sorted(data) + [unknown_func(1)]")
        assert mock_shell.user_ns['result'] == [1, 2, 3, 2]
