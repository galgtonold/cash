"""IPython adapter for the notebook caching subsystem.

The four files in this package — `magics`, `admin`, `cell_executor`,
`error_display` — make up the adapter that wires Cash's caching pipeline
into IPython's `Magics` system: `%cash_on`, `%%cash`, `%cash_status`,
`%cash_stats`, and so on.

Public surface:
    - :class:`CashMagics` — the `Magics` subclass that IPython registers.

Everything else (`CashAdminMagicsMixin`, `CellExecutor`, the value types
`TimingBreakdown` / `StatementSummary` / `CellMetrics` / `CashSession`,
`show_clean_error`, the internal pipeline sentinels) is package-internal.
See ADR-013 for the package-extraction rationale.
"""

from __future__ import annotations

from .magics import CashMagics

__all__ = ["CashMagics"]
