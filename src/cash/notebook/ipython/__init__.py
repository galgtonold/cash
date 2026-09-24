"""IPython adapter for the notebook caching subsystem.

The modules in this package — `magics`, `inspection`, `cell_executor`, `badges`,
`error_display` — make up the adapter that wires Cash's caching pipeline
into IPython's `Magics` system: `%cash_on`, `%cash_status`,
`%cash_stats`, and so on.

Public surface:
    - :class:`CashMagics` — the `Magics` subclass that IPython registers.

Everything else (`InspectionMagicsMixin`, `CellExecutor`, `BadgePresenter`, the value types
`TimingBreakdown` / `StatementSummary` / `CellMetrics` / `CashSession`,
`show_clean_error`, the internal pipeline sentinels) is package-internal:
code outside the package imports only `CashMagics`.
"""

from __future__ import annotations

from .magics import CashMagics

__all__ = ["CashMagics"]
