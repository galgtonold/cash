"""Visualization and debugging tools for cache inspection."""

from __future__ import annotations

# Import only what doesn't cause circular dependencies
from .graph import DependencyGraph


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'CacheExplorer':
        from .explorer import CacheExplorer
        return CacheExplorer
    if name == 'visualize_notebook':
        from .visualizer import visualize_notebook
        return visualize_notebook
    if name == 'CacheDebugger':
        from .debugger import CacheDebugger
        return CacheDebugger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'CacheExplorer',
    'visualize_notebook',
    'CacheDebugger',
    'DependencyGraph',
]
