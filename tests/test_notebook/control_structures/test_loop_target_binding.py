"""A for loop's target binds per iteration however it is written: a single name,
a flat tuple, nested tuples, an enumerate pair or a list target.
"""

import ast


class TestNestedTupleUnpacking:
    """Test that nested tuple unpacking in for loops works correctly."""

    def test_bind_target_values_simple(self):
        """Simple single variable binding."""
        from cash.notebook.control_structures import bind_target_values

        code = "for i in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        ns = {}
        bindings = bind_target_values(target, 42, ns)
        assert bindings == {"i": 42}
        assert ns["i"] == 42

    def test_bind_target_values_flat_tuple(self):
        """Flat tuple unpacking: for a, b in data."""
        from cash.notebook.control_structures import bind_target_values

        code = "for a, b in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        ns = {}
        bindings = bind_target_values(target, (10, 20), ns)
        assert bindings == {"a": 10, "b": 20}
        assert ns["a"] == 10
        assert ns["b"] == 20

    def test_bind_target_values_nested_tuple(self):
        """Nested tuple unpacking: for a, (b, c) in data.

        The pattern that used to bind the wrong values.
        """
        from cash.notebook.control_structures import bind_target_values

        code = "for a, (b, c) in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        ns = {}
        bindings = bind_target_values(target, (1, (2, 3)), ns)
        assert bindings == {"a": 1, "b": 2, "c": 3}
        assert ns["a"] == 1
        assert ns["b"] == 2
        assert ns["c"] == 3

    def test_bind_target_values_deep_nested(self):
        """Deeply nested tuple unpacking: for a, (b, (c, d)) in data."""
        from cash.notebook.control_structures import bind_target_values

        code = "for a, (b, (c, d)) in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        ns = {}
        bindings = bind_target_values(target, ("x", ("y", ("z", "w"))), ns)
        assert bindings == {"a": "x", "b": "y", "c": "z", "d": "w"}
        assert ns["a"] == "x"
        assert ns["b"] == "y"
        assert ns["c"] == "z"
        assert ns["d"] == "w"

    def test_bind_target_values_enumerate_pattern(self):
        """Common enumerate pattern: for i, (k, v) in enumerate(items)."""
        from cash.notebook.control_structures import bind_target_values

        code = "for i, (k, v) in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        ns = {}
        bindings = bind_target_values(target, (0, ("key", "value")), ns)
        assert bindings == {"i": 0, "k": "key", "v": "value"}
        assert ns["i"] == 0
        assert ns["k"] == "key"
        assert ns["v"] == "value"

    def test_bind_target_values_with_list_target(self):
        """List unpacking target: for [a, b] in data."""
        from cash.notebook.control_structures import bind_target_values

        code = "for [a, b] in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        ns = {}
        bindings = bind_target_values(target, [10, 20], ns)
        assert bindings == {"a": 10, "b": 20}
        assert ns["a"] == 10
        assert ns["b"] == 20

    def test_extract_target_names_still_works(self):
        """Ensure extract_target_names is unchanged (backward compatibility)."""
        from cash.notebook.control_structures import extract_target_names

        code = "for a, (b, c) in data: pass"
        tree = ast.parse(code)
        target = tree.body[0].target

        names = extract_target_names(target)
        assert set(names) == {"a", "b", "c"}
