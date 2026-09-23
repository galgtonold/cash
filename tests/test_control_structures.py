"""Tests for control_structures helper functions."""

import ast

from cash.notebook.cache_status import CacheStatus
from cash.notebook.control_structures import (
    ControlStructureResult,
    build_iteration_context,
    compute_context_hash,
    contains_break_or_continue,
    extract_target_names,
    get_control_structure_type,
    is_control_structure,
)


class TestIsControlStructure:
    """Test control structure detection."""

    def test_for_loop(self):
        node = ast.parse("for x in items: pass").body[0]
        assert is_control_structure(node) is True

    def test_while_loop(self):
        node = ast.parse("while True: pass").body[0]
        assert is_control_structure(node) is True

    def test_if_statement(self):
        node = ast.parse("if x: pass").body[0]
        assert is_control_structure(node) is True

    def test_with_statement(self):
        node = ast.parse("with open('f'): pass").body[0]
        assert is_control_structure(node) is True

    def test_try_statement(self):
        node = ast.parse("try:\n pass\nexcept: pass").body[0]
        assert is_control_structure(node) is True

    def test_assignment_not_control(self):
        node = ast.parse("x = 1").body[0]
        assert is_control_structure(node) is False

    def test_expression_not_control(self):
        node = ast.parse("print(1)").body[0]
        assert is_control_structure(node) is False


class TestGetControlStructureType:
    """Test control structure type detection."""

    def test_for_type(self):
        node = ast.parse("for x in items: pass").body[0]
        assert get_control_structure_type(node) == "for"

    def test_while_type(self):
        node = ast.parse("while True: pass").body[0]
        assert get_control_structure_type(node) == "while"

    def test_if_type(self):
        node = ast.parse("if x: pass").body[0]
        assert get_control_structure_type(node) == "if"

    def test_with_type(self):
        node = ast.parse("with open('f'): pass").body[0]
        assert get_control_structure_type(node) == "with"

    def test_try_type(self):
        node = ast.parse("try:\n pass\nexcept: pass").body[0]
        assert get_control_structure_type(node) == "try"

    def test_none_for_other(self):
        node = ast.parse("x = 1").body[0]
        assert get_control_structure_type(node) is None


class TestContainsBreakOrContinue:
    """Test break/continue detection."""

    def test_no_break_continue(self):
        nodes = ast.parse("x = 1\ny = 2").body
        assert contains_break_or_continue(nodes) is False

    def test_with_break(self):
        nodes = ast.parse("if x:\n break").body
        assert contains_break_or_continue(nodes) is True

    def test_with_continue(self):
        nodes = ast.parse("if x:\n continue").body
        assert contains_break_or_continue(nodes) is True

    def test_nested_break(self):
        nodes = ast.parse("if x:\n if y:\n  break").body
        assert contains_break_or_continue(nodes) is True


class TestExtractTargetNames:
    """Test loop target name extraction."""

    def test_simple_name(self):
        node = ast.parse("for x in items: pass").body[0]
        assert extract_target_names(node.target) == ["x"]

    def test_tuple_unpack(self):
        node = ast.parse("for x, y in items: pass").body[0]
        assert extract_target_names(node.target) == ["x", "y"]

    def test_nested_tuple(self):
        node = ast.parse("for (a, (b, c)) in items: pass").body[0]
        names = extract_target_names(node.target)
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_list_target(self):
        node = ast.parse("for [a, b] in items: pass").body[0]
        assert extract_target_names(node.target) == ["a", "b"]

    def test_unsupported_target(self):
        """Non-name targets return empty list."""
        # Attribute targets like obj.attr
        target = ast.Attribute(value=ast.Name(id="obj"), attr="x")
        assert extract_target_names(target) == []


class TestBuildIterationContext:
    """Test iteration context building."""

    def test_simple_context(self):
        user_ns = {"x": 10, "y": 20}
        ctx = build_iteration_context(["x"], user_ns, None)
        assert ctx == {"x": 10}

    def test_multiple_vars(self):
        user_ns = {"x": 10, "y": 20}
        ctx = build_iteration_context(["x", "y"], user_ns, None)
        assert ctx == {"x": 10, "y": 20}

    def test_parent_context(self):
        user_ns = {"x": 10}
        parent = {"outer": 5}
        ctx = build_iteration_context(["x"], user_ns, parent)
        assert ctx == {"outer": 5, "x": 10}

    def test_unhashable_value(self):
        """Unhashable values are stored as a FULL-content hash.

        repr() truncates large numpy/pandas objects, so two iterations
        differing outside the repr window collided into one context hash.
        """
        user_ns = {"x": [1, 2, 3]}
        ctx = build_iteration_context(["x"], user_ns, None)
        assert set(ctx) == {"x"}
        assert isinstance(ctx["x"], str) and len(ctx["x"]) == 64
        # Distinct content -> distinct context entry, even where repr agrees.
        import numpy as np

        a = np.zeros(2000)
        b = np.zeros(2000)
        b[1000] = 5.0
        assert repr(a) == repr(b), "precondition: repr truncates the difference"
        ctx_a = build_iteration_context(["x"], {"x": a}, None)
        ctx_b = build_iteration_context(["x"], {"x": b}, None)
        assert ctx_a["x"] != ctx_b["x"]

    def test_missing_var(self):
        """Missing variables are skipped."""
        user_ns = {"y": 20}
        ctx = build_iteration_context(["x"], user_ns, None)
        assert ctx == {}


class TestComputeContextHash:
    """Test context hash computation."""

    def test_deterministic(self):
        ctx = {"x": 10, "y": 20}
        h1 = compute_context_hash(ctx)
        h2 = compute_context_hash(ctx)
        assert h1 == h2

    def test_different_context_different_hash(self):
        h1 = compute_context_hash({"x": 10})
        h2 = compute_context_hash({"x": 20})
        assert h1 != h2

    def test_order_independent(self):
        """Key ordering doesn't matter."""
        h1 = compute_context_hash({"a": 1, "b": 2})
        h2 = compute_context_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_length(self):
        h = compute_context_hash({"x": 10})
        assert len(h) == 16  # Truncated to 16 chars


class _Point:
    """A plain user object: hashable by identity, repr carries its address."""

    def __init__(self, x):
        self.x = x


class TestUserObjectLoopValue:
    """A user object's str() holds its memory address, so hashing the context
    by str() changed on every run and never on mutation."""

    def _hash(self, value, digests=None):
        return compute_context_hash(build_iteration_context(["p"], {"p": value}, None, digests))

    def test_equal_content_in_a_new_object_gives_the_same_hash(self):
        # A kernel restart builds a new object at a new address.
        assert self._hash(_Point(1)) == self._hash(_Point(1))

    def test_mutation_changes_the_hash(self):
        p = _Point(1)
        before = self._hash(p)
        p.x = 2
        assert self._hash(p) != before

    def test_the_live_object_stays_in_the_context(self):
        # loop_vars hands the object itself to call caching.
        p = _Point(1)
        ctx = build_iteration_context(["p"], {"p": p}, None)
        assert ctx["p"] is p

    def test_the_callers_digest_is_used(self, monkeypatch):
        import cash.notebook.object_hashing as object_hashing

        def _fail(value):
            raise AssertionError("recomputed a digest the caller already had")

        monkeypatch.setattr(object_hashing, "compute_hash_full", _fail)
        p = _Point(1)
        assert self._hash(p, {"p": "d1"}) != self._hash(p, {"p": "d2"})

    def test_nested_loop_carries_the_outer_digest(self):
        outer = build_iteration_context(["p"], {"p": _Point(1)}, None)
        inner_a = build_iteration_context(["i"], {"i": 0}, outer)
        other = build_iteration_context(["p"], {"p": _Point(2)}, None)
        inner_b = build_iteration_context(["i"], {"i": 0}, other)
        assert compute_context_hash(inner_a) != compute_context_hash(inner_b)

    def test_primitives_hash_as_before(self):
        import hashlib

        expected = hashlib.sha256(str([("i", 3), ("s", "a")]).encode()).hexdigest()[:16]
        assert compute_context_hash({"s": "a", "i": 3}) == expected


class TestControlStructureResult:
    """Test the ControlStructureResult dataclass."""

    def test_defaults(self):
        result = ControlStructureResult(success=True, metrics=[])
        assert result.success is True
        assert result.metrics == []
        assert result.error is None
        assert result.total_iterations == 0

    def test_with_metrics(self):
        metrics = [{"status": CacheStatus.COMPUTED}, {"status": CacheStatus.RESTORED}]
        result = ControlStructureResult(
            success=True,
            metrics=metrics,
            total_iterations=2,
            cached_iterations=1,
            computed_iterations=1,
        )
        assert result.total_iterations == 2
        assert result.cached_iterations == 1
