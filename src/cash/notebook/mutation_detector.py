from __future__ import annotations

"""AST-based detection of in-place mutations that affect caching correctness."""

import ast
from dataclasses import dataclass

__all__ = ["MUTATING_METHODS", "PANDAS_INPLACE_METHODS", "MutationInfo", "MutationDetector"]

# Methods that mutate their receiver object in-place
MUTATING_METHODS = {
    # list methods
    'append', 'extend', 'insert', 'pop', 'remove', 'sort', 'reverse', 'clear',
    # dict methods
    'update', 'popitem', 'setdefault', # set methods
    'add', 'discard', 'intersection_update', 'difference_update', 'symmetric_difference_update',
}

# Pandas methods that accept inplace=True
PANDAS_INPLACE_METHODS = {
    'fillna', 'dropna', 'drop', 'rename', 'reset_index', 'set_index',
    'sort_values', 'sort_index', 'replace', 'clip', 'where', 'mask',
    'drop_duplicates', 'eval', 'query', 'astype',
}

@dataclass
class MutationInfo:
    """Information about a detected mutation."""
    variable: str
    method: str
    kind: str  # 'method_call', 'inplace_kwarg', 'augmented_assign', 'subscript_assign'
    line: int = 0

class _MutationVisitor(ast.NodeVisitor):
    """AST visitor that collects MutationInfo entries into ``self.mutations``."""

    def __init__(self) -> None:
        self.mutations: list[MutationInfo] = []

    def visit_Expr(self, node: ast.Expr) -> None:
        """Detect standalone method calls like lst.append(x)."""
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute):
                method_name = call.func.attr
                base = _extract_base_name(call.func.value)

                if base and method_name in MUTATING_METHODS:
                    self.mutations.append(MutationInfo(
                        variable=base, method=method_name,
                        kind='method_call', line=node.lineno,
                    ))

                if base and method_name in PANDAS_INPLACE_METHODS:
                    for kw in call.keywords:
                        if kw.arg == 'inplace' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            self.mutations.append(MutationInfo(
                                variable=base, method=method_name,
                                kind='inplace_kwarg', line=node.lineno,
                            ))
                            break
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Detect augmented assignments like x += 1, arr *= 2."""
        base = _extract_base_name(node.target)
        if base:
            op_name = type(node.op).__name__
            self.mutations.append(MutationInfo(
                variable=base, method=f'__i{op_name.lower()}__',
                kind='augmented_assign', line=node.lineno,
            ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect subscript/attribute assignments like d[key] = val, obj.attr = val."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method='__setitem__',
                        kind='subscript_assign', line=node.lineno,
                    ))
            elif isinstance(target, ast.Attribute):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method=f'__setattr__({target.attr})',
                        kind='attribute_assign', line=node.lineno,
                    ))
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Detect del d[key], del lst[0]."""
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                base = _extract_base_name(target.value)
                if base:
                    self.mutations.append(MutationInfo(
                        variable=base, method='__delitem__',
                        kind='subscript_delete', line=node.lineno,
                    ))
        self.generic_visit(node)


class MutationDetector:
    """
    AST-based detection of in-place mutations.

    Analyzes code to find operations that modify existing objects,
    which is critical for cache correctness — mutated objects may
    have stale lineage hashes if the mutation isn't tracked.
    """

    @staticmethod
    def detect_mutations(code: str, tree: ast.Module | None = None) -> list[MutationInfo]:
        """Analyze code and return all detected in-place mutations.

        Args:
            code: Python source code to analyze
            tree: Optional pre-parsed AST to avoid redundant parsing.
        """
        if tree is None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return []
        visitor = _MutationVisitor()
        visitor.visit(tree)
        return visitor.mutations

    @staticmethod
    def get_mutated_variables(code: str, tree: ast.Module | None = None) -> set[str]:
        """
        Return the set of variable names that are mutated in-place.

        This is a convenience method that extracts just the variable names
        from the full mutation analysis.
        """
        return {m.variable for m in MutationDetector.detect_mutations(code, tree=tree)}

    @staticmethod
    def get_top_level_mutated_variables(code: str, tree: ast.Module | None = None) -> set[str]:
        """
        Return variables mutated at the top level only (not inside class/function bodies).

        This is important for caching decisions: mutations inside class methods
        or function definitions (e.g., self.val = val) don't affect the caching
        of the statement that defines the class/function itself.

        Args:
            code: Python source code to analyze
            tree: Optional pre-parsed AST to avoid redundant parsing.
        """
        if tree is None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return set()

        # Only analyze top-level statements, skipping class/function bodies
        top_level_code_lines: list[str] = []
        source_lines = code.split('\n')

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip: mutations inside these are internal to the definition
                continue
            # For other top-level statements, we include them for mutation analysis
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                # Extract source lines for this node
                start = node.lineno - 1
                end = node.end_lineno
                top_level_code_lines.extend(source_lines[start:end])

        if not top_level_code_lines:
            return set()

        top_level_code = '\n'.join(top_level_code_lines)
        return MutationDetector.get_mutated_variables(top_level_code)

    @staticmethod
    def has_mutations(code: str) -> bool:
        """Quick check: does this code contain any in-place mutations?"""
        return len(MutationDetector.detect_mutations(code)) > 0

def _extract_base_name(node: ast.AST) -> str | None:
    """Extract the root variable name from a potentially nested AST node.

    Handles chained method calls like ``groups.setdefault(key, []).append(val)``
    by walking through Call nodes to reach the underlying variable.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        return _extract_base_name(node.value)
    if isinstance(node, ast.Call):
        # For chained calls like obj.method1().method2(), walk through the Call
        # to find the root variable.  e.g. groups.setdefault(key, []).append(val)
        # -> Call(func=Attr('append', value=Call(func=Attr('setdefault', value=Name('groups')))))
        if isinstance(node.func, ast.Attribute):
            return _extract_base_name(node.func.value)
        if isinstance(node.func, ast.Name):
            return node.func.id
    return None

