from __future__ import annotations

"""IPython/Jupyter notebook integration for statement-level caching."""

# Import only what doesn't cause circular dependencies
from .analysis import CodeAnalyzer
from .cache_status import CacheStatus, ExecutionResult
from .upstream import UpstreamChecker


# Lazy imports to break circular dependency chain:
#   ipython.magics → statement.processor → core → notebook/__init__ → ipython.magics
# CashMagics and StatementProcessor are deferred to first access.
def __getattr__(name):
    if name == 'CashMagics':
        from .ipython import CashMagics
        return CashMagics
    if name == 'StatementProcessor':
        from .statement import StatementProcessor
        return StatementProcessor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'CacheStatus',
    'CashMagics',
    'CodeAnalyzer',
    'ExecutionResult',
    'UpstreamChecker',
    'StatementProcessor',
]

