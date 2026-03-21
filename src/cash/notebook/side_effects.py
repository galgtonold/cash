from __future__ import annotations

"""Detection of I/O and system side effects that make statements uncacheable."""

import ast
from dataclasses import dataclass

__all__ = ["SideEffectInfo", "SideEffectDetector"]

@dataclass
class SideEffectInfo:
    """Information about a detected side effect."""
    kind: str  # 'file_write', 'network', 'system', 'global_state'
    description: str
    line: int = 0

# Function calls known to produce I/O side effects (writing to files, network, etc.)
# Format: (module_or_none, function_name) -> side_effect_kind
_IO_SIDE_EFFECT_FUNCTIONS = {
    # File writing
    ('', 'open'): 'file_write',  # open() with write modes detected separately
    # os module
    ('os', 'remove'): 'file_write',
    ('os', 'unlink'): 'file_write',
    ('os', 'rmdir'): 'file_write',
    ('os', 'mkdir'): 'file_write',
    ('os', 'makedirs'): 'file_write',
    ('os', 'rename'): 'file_write',
    ('os', 'replace'): 'file_write',
    ('os', 'symlink'): 'file_write',
    ('os', 'system'): 'system',
    # shutil
    ('shutil', 'copy'): 'file_write',
    ('shutil', 'copy2'): 'file_write',
    ('shutil', 'copytree'): 'file_write',
    ('shutil', 'rmtree'): 'file_write',
    ('shutil', 'move'): 'file_write',
    # subprocess
    ('subprocess', 'run'): 'system',
    ('subprocess', 'call'): 'system',
    ('subprocess', 'Popen'): 'system',
    ('subprocess', 'check_call'): 'system',
    ('subprocess', 'check_output'): 'system',
    # pandas write operations
    ('', 'to_csv'): 'file_write',
    ('', 'to_excel'): 'file_write',
    ('', 'to_parquet'): 'file_write',
    ('', 'to_json'): 'file_write',
    ('', 'to_pickle'): 'file_write',
    ('', 'to_hdf'): 'file_write',
    ('', 'to_feather'): 'file_write',
    ('', 'to_sql'): 'file_write',
    # json/pickle/csv module
    ('json', 'dump'): 'file_write',
    ('pickle', 'dump'): 'file_write',
    ('csv', 'writer'): 'file_write',
    # requests/urllib
    ('requests', 'post'): 'network',
    ('requests', 'put'): 'network',
    ('requests', 'delete'): 'network',
    ('requests', 'patch'): 'network',
}

# Method names that indicate writing (when called on any object)
_WRITE_METHODS = frozenset({
    'to_csv', 'to_excel', 'to_parquet', 'to_json', 'to_pickle',
    'to_hdf', 'to_feather', 'to_sql', 'to_stata', 'to_latex',
    'to_html', 'to_clipboard', 'to_gbq', 'to_markdown',
    'savefig',  # matplotlib
    'save',     # numpy, PIL, torch
    'write',    # file objects
    'writelines',
})

# File open modes that indicate writing
_WRITE_MODES = frozenset({'w', 'wb', 'a', 'ab', 'w+', 'wb+', 'a+', 'ab+', 'x', 'xb'})

class _SideEffectVisitor(ast.NodeVisitor):
    """Collects side-effect call sites from an AST."""

    def __init__(self) -> None:
        self.effects: list[SideEffectInfo] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Detect function/method calls with side effects."""
        # Case 1: Simple function calls like open(), os.remove()
        func_name = _get_call_name(node.func)
        module_name = _get_call_module(node.func)

        if func_name:
            # Check for known side-effect functions
            key = (module_name or '', func_name)
            if key in _IO_SIDE_EFFECT_FUNCTIONS:
                kind = _IO_SIDE_EFFECT_FUNCTIONS[key]
                # Special case: open() is only a side effect in write mode
                if func_name == 'open' and not module_name:
                    if _is_open_write_mode(node):
                        self.effects.append(SideEffectInfo(
                            kind='file_write',
                            description="open() with write mode",
                            line=getattr(node, 'lineno', 0)
                        ))
                else:
                    self.effects.append(SideEffectInfo(
                        kind=kind,
                        description=f"{module_name + '.' if module_name else ''}{func_name}()",
                        line=getattr(node, 'lineno', 0)
                    ))

            # Check for write methods called on objects
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in _WRITE_METHODS:
                    base = _get_base_name(node.func.value)
                    self.effects.append(SideEffectInfo(
                        kind='file_write',
                        description=f"{base + '.' if base else ''}{method}()",
                        line=getattr(node, 'lineno', 0)
                    ))

        self.generic_visit(node)


class SideEffectDetector:
    """
    AST-based detection of side effects in Python code.

    Identifies operations that produce effects beyond computing return values,
    such as file writes, network calls, and system commands.
    """

    @staticmethod
    def detect_side_effects(code: str, tree: ast.Module | None = None) -> list[SideEffectInfo]:
        """
        Analyze code and return all detected side effects.

        Args:
            code: Python source code to analyze
            tree: Optional pre-parsed AST to avoid redundant parsing.

        Returns:
            List of SideEffectInfo objects describing detected side effects
        """
        if tree is None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return []

        visitor = _SideEffectVisitor()
        visitor.visit(tree)
        return visitor.effects

    @staticmethod
    def has_side_effects(code: str) -> bool:
        """Quick check: does this code contain any side effects?"""
        return len(SideEffectDetector.detect_side_effects(code)) > 0

    @staticmethod
    def get_side_effect_reasons(code: str, tree: ast.Module | None = None) -> list[str]:
        """Return human-readable reasons for detected side effects."""
        effects = SideEffectDetector.detect_side_effects(code, tree=tree)
        return [f"Side effect: {e.description} ({e.kind})" for e in effects]

def _get_call_name(func_node: ast.AST) -> str | None:
    """Extract the function name from a call's func node."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return None

def _get_call_module(func_node: ast.AST) -> str | None:
    """Extract the module/object prefix from a call's func node."""
    if isinstance(func_node, ast.Attribute):
        if isinstance(func_node.value, ast.Name):
            return func_node.value.id
        if isinstance(func_node.value, ast.Attribute):
            # e.g., os.path.join -> module = 'os.path'
            parts = []
            node = func_node.value
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return '.'.join(reversed(parts))
    return None

def _get_base_name(node: ast.AST) -> str | None:
    """Extract a human-readable name for the object a method is called on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_base_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        base = _get_base_name(node.value)
        return f"{base}[...]" if base else None
    return None

def _is_open_write_mode(call_node: ast.Call) -> bool:
    """Check if an open() call uses a write mode."""
    # Check positional arg (2nd argument is mode)
    if len(call_node.args) >= 2:
        mode_arg = call_node.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            return mode_arg.value in _WRITE_MODES or any(c in mode_arg.value for c in 'wax')

    # Check keyword argument mode=...
    for kw in call_node.keywords:
        if kw.arg == 'mode' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value in _WRITE_MODES or any(c in kw.value.value for c in 'wax')

    return False

