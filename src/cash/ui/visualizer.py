"""Notebook cache visualization utilities."""

from __future__ import annotations

__all__ = ["visualize_notebook", "visualize_cache"]


try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from IPython.display import display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from ..utils import get_notebook_cells, get_notebook_path
from .debugger import CacheDebugger

# ── Status display constants ──────────────────────────────────────────────────

STATUS_HIT = "HIT"
STATUS_MISS = "MISS"

_STATUS_STYLES = {
    STATUS_HIT: "color: green; font-weight: bold",
    STATUS_MISS: "color: red; font-weight: bold",
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_time(seconds: float | None) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds is None:
        return "N/A"
    if seconds < 0.001:
        return "< 1ms"
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def format_memory(bytes_size: int | None) -> str:
    """Format byte count into a human-readable size string."""
    if bytes_size is None:
        return "N/A"
    if bytes_size == 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB']
    unit_index = 0
    size = float(bytes_size)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


# ── Shared rendering logic ────────────────────────────────────────────────────

def _build_results_dataframe(
    results: list[dict],
    include_cell_column: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """Build a styled DataFrame from debug results.

    Args:
        results: List of statement analysis dicts from CacheDebugger.debug().
        include_cell_column: If True, include a 'Cell' column with cell numbers.

    Returns:
        Tuple of (DataFrame, hit_count, total_count).
    """
    var_to_stmt: dict[str, int] = {}
    data: list[dict] = []
    stmt_counter = 1

    for res in results:
        if 'error' in res:
            row: dict = {'#': stmt_counter, 'Code': 'ERROR', 'Status': res['error'],
                         'Depends On': '', 'Outputs': ''}
            if include_cell_column:
                row['Cell'] = res.get('cell_number', '?')
            data.append(row)
            stmt_counter += 1
            continue

        # Find which previous statements this statement depends on
        dependencies = [
            f"#{var_to_stmt[input_var]}"
            for input_var in res['inputs']
            if input_var in var_to_stmt
        ]

        # Track which variables this statement produces
        for output_var in res['outputs']:
            var_to_stmt[output_var] = stmt_counter

        deps_str = ", ".join(dependencies) if dependencies else "None"
        code_display = res['code'][:60] + "..." if len(res['code']) > 60 else res['code']

        row = {
            '#': stmt_counter,
            'Code': code_display,
            'Status': res['status'],
            'Exec Time': format_time(res.get('execution_time')),
            'Memory': format_memory(res.get('memory_size')),
            'Depends On': deps_str,
            'Outputs': ", ".join(res['outputs']),
        }
        if include_cell_column:
            row['Cell'] = res.get('cell_number', '?')
        data.append(row)
        stmt_counter += 1

    df = pd.DataFrame(data)

    # Reorder columns so Cell comes after # if present
    if include_cell_column and 'Cell' in df.columns:
        cols = list(df.columns)
        if '#' in cols and 'Cell' in cols:
            cols.remove('Cell')
            cols.insert(cols.index('#') + 1, 'Cell')
            df = df[cols]

    hits = sum(1 for r in results if 'status' in r and r['status'] == STATUS_HIT)
    total = len([r for r in results if 'error' not in r])

    return df, hits, total


def _display_styled_dataframe(df: pd.DataFrame, hits: int, total: int) -> None:
    """Apply status styling to a DataFrame and display it with a summary."""
    if not HAS_IPYTHON:
        print(df.to_string())
        if total > 0:
            print(f"\nCache Summary: {hits}/{total} statements cached ({hits/total*100:.1f}%)")
        return

    def _color_status(val: str) -> str:
        return _STATUS_STYLES.get(val, '')

    styler = df.style.map(_color_status, subset=['Status'])
    display(styler)

    if total > 0:
        print(f"\nCache Summary: {hits}/{total} statements cached ({hits/total*100:.1f}%)")


# ── Public API ────────────────────────────────────────────────────────────────

def _collect_cells_from_notebook(
    code_cells: list,
    cell_numbers: list[int] | None,
    notebook_path: str | None,
) -> list[tuple[int, str]]:
    """Collect (cell_index, code) pairs from a notebook file's code cells."""
    if not notebook_path:
        notebook_path = get_notebook_path()

    print(f"Reading from notebook: {notebook_path}")
    print("Note: Visualizing saved file content. Unsaved changes may not be visible.")

    cells: list[tuple[int, str]] = []
    for idx, cell_code in enumerate(code_cells, start=1):
        if cell_code and cell_code.strip() and (cell_numbers is None or idx in cell_numbers):
            cells.append((idx, cell_code))

    print(f"Analyzed {len(cells)} cells from {notebook_path}")
    return cells


def _collect_cells_from_shell(
    debugger: CacheDebugger,
    cell_numbers: list[int] | None,
) -> list[tuple[int, str]] | None:
    """Collect (cell_index, code) pairs from IPython shell execution history.

    Returns None if the shell history cannot be accessed.
    """
    print("No notebook file found or provided. Falling back to execution history.")
    shell = debugger.shell

    if not (hasattr(shell, 'user_ns') and 'In' in shell.user_ns):
        print("Could not access notebook cells")
        return None

    all_cells = shell.user_ns['In']
    cells: list[tuple[int, str]] = []
    for idx, cell_code in enumerate(all_cells):
        if idx == 0:
            continue
        if cell_code and cell_code.strip() and (cell_numbers is None or idx in cell_numbers):
            cells.append((idx, cell_code))
    return cells


def visualize_notebook(
    debugger: CacheDebugger,
    cell_numbers: list[int] | None = None,
    notebook_path: str | None = None,
) -> pd.DataFrame | None:
    """Analyze cells from the current notebook and display a summary table.

    Args:
        debugger: CacheDebugger instance.
        cell_numbers: Cell numbers to analyze (1-indexed), or None for all.
        notebook_path: Path to the .ipynb file. Auto-detected if omitted.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for visualize_notebook(). Install with: pip install cash-lib[pandas]")

    code_cells = get_notebook_cells(notebook_path)

    if code_cells:
        cells_to_analyze = _collect_cells_from_notebook(code_cells, cell_numbers, notebook_path)
    else:
        result = _collect_cells_from_shell(debugger, cell_numbers)
        if result is None:
            return None
        cells_to_analyze = result

    if not cells_to_analyze:
        print("No cells to analyze")
        return None

    # Analyze each cell
    all_results: list[dict] = []
    for cell_num, cell_code in cells_to_analyze:
        results = debugger.debug(cell_code)
        for res in results:
            res['cell_number'] = cell_num
            all_results.append(res)

    df, hits, total = _build_results_dataframe(all_results, include_cell_column=True)
    _display_styled_dataframe(df, hits, total)
    return df


def visualize_cache(debugger: CacheDebugger, code: str) -> pd.DataFrame | None:
    """Analyze code using the debugger and display a summary table."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for visualize_cache(). Install with: pip install cash-lib[pandas]")
    results = debugger.debug(code)
    df, hits, total = _build_results_dataframe(results, include_cell_column=False)
    _display_styled_dataframe(df, hits, total)
    return df
