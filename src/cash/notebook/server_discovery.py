"""Jupyter Server integration: notebook path discovery and cell reading.

This module handles all I/O and HTTP calls required to locate the current
notebook file and read its cell contents.  It is intentionally separate from
``cash.utils`` (pure path helpers) to make the network/filesystem dependency
explicit and to co-locate the code with its primary consumers in
``cash.notebook``.

Cross-module coupling: the ``_cached_notebook_path`` module-level variable is
shared state for the lifetime of a kernel session.  All callers go through
:func:`get_notebook_path`, :func:`invalidate_notebook_path_cache`, and
:func:`set_notebook_path` — never read the variable directly.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
import urllib.error
import urllib.request
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Session-level cache for notebook path discovery.
#
# Discovery is expensive: it queries the Jupyter Server REST API (an HTTP
# round-trip) and may fall back to scanning the filesystem.  The path is
# stable for the lifetime of a kernel session, so we cache it with a long TTL
# to avoid repeated round-trips on every cell execution.
#
# Lifecycle:
#   Created: lazily, on the first call to get_notebook_path().
#   Invalidated: explicitly via invalidate_notebook_path_cache() — called by
#     %cash_on to handle notebook switches and by _check_notebook_based() when
#     the current cell cannot be found (possible stale session from Jupyter
#     Server returning the wrong active notebook path).
#   Overridden: via set_notebook_path() when a reliable path is available from
#     another source (e.g., extracted from a VS Code cell-ID URI).
#
# Cross-module coupling: all callers (magics.py, upstream.py, utils.py)
# share this module-level variable via the accessor functions below.
# Treat it as a single global instance — do not duplicate or shadow it.
_cached_notebook_path: str | None = None
_cached_notebook_path_time: float = 0.0
_NOTEBOOK_PATH_CACHE_TTL: float = 300.0  # seconds (5 minutes)


def invalidate_notebook_path_cache() -> None:
    """
    Invalidate the cached notebook path so the next call to
    get_notebook_path() re-discovers the current notebook.
    This should be called when %cash_on is invoked to handle
    notebook switches within the same kernel session (Issue 23).
    """
    global _cached_notebook_path, _cached_notebook_path_time
    _cached_notebook_path = None
    _cached_notebook_path_time = 0.0


def set_notebook_path(path: str) -> None:
    """
    Explicitly set the notebook path (bypassing auto-detection).

    This is called when the notebook path is reliably known from another
    source, e.g. extracted from a VS Code cell ID URI.  The value is
    stored in the same session-level cache used by ``get_notebook_path()``.
    """
    global _cached_notebook_path, _cached_notebook_path_time
    if path and os.path.exists(path):
        _cached_notebook_path = path
        _cached_notebook_path_time = _time.monotonic()
        logger.debug("[UTILS] Notebook path set explicitly: %s", path)


def extract_notebook_path_from_vscode_cell_id(cell_id: str) -> str | None:
    """
    Extract the notebook file path from a VS Code cell ID URI.

    VS Code cell IDs look like::

        vscode-notebook-cell:/c%3A/Users/.../notebook.ipynb#W2sZmlsZQ%3D%3D

    Returns the decoded filesystem path if the URI matches, else ``None``.
    """
    if not cell_id or not cell_id.startswith('vscode-notebook-cell:'):
        return None
    try:
        # Strip the fragment (#W2sZmlsZQ==)
        uri_part = cell_id.split('#')[0]
        # Remove scheme
        path_part = uri_part.replace('vscode-notebook-cell:', '', 1)
        # URL-decode  (e.g. %3A → :, %20 → space)
        decoded = unquote(path_part)
        # On Windows the path looks like /c:/Users/...  → strip leading /
        if len(decoded) > 2 and decoded[0] == '/' and decoded[2] == ':':
            decoded = decoded[1:]
        # Normalise separators
        decoded = os.path.normpath(decoded)
        if os.path.exists(decoded) and decoded.endswith('.ipynb'):
            return decoded
    except (ValueError, IndexError, OSError, UnicodeDecodeError):
        logger.debug("[UTILS] Failed to extract notebook path from VS Code cell ID: %s", cell_id)
    return None


def _try_vscode_path() -> str | None:
    """Return notebook path from VS Code's injected variable, or None."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip and hasattr(ip, 'user_ns') and '__vsc_ipynb_file__' in ip.user_ns:
            return ip.user_ns['__vsc_ipynb_file__']
    except (ImportError, AttributeError, KeyError):
        logger.debug("[UTILS] Failed to get notebook path from IPython user_ns")
    return None


def _try_ipynbname_path() -> str | None:
    """Return notebook path via the ipynbname package, or None."""
    try:
        import ipynbname
        return str(ipynbname.path())
    except (ImportError, OSError, AttributeError):
        logger.debug("[UTILS] ipynbname not available or failed")
    return None


def _collect_running_servers() -> list:
    """Return all running Jupyter server descriptors from all server packages."""
    servers: list = []
    try:
        from jupyter_server import serverapp
        servers.extend(list(serverapp.list_running_servers()))
    except (ImportError, AttributeError):
        logger.debug("[UTILS] jupyter_server not available")
    try:
        from notebook import notebookapp
        servers.extend(list(notebookapp.list_running_servers()))
    except (ImportError, AttributeError):
        logger.debug("[UTILS] notebook.notebookapp not available")
    return servers


def _search_servers_for_notebook(kernel_id: str) -> str | None:
    """Query running Jupyter servers to find the notebook matching kernel_id."""
    for server in _collect_running_servers():
        try:
            url = server['url'].rstrip('/') + '/api/sessions'
            token = server.get('token', '')
            req = urllib.request.Request(url)
            if token:
                req.add_header('Authorization', f'token {token}')
            with urllib.request.urlopen(req, timeout=2) as response:
                sessions = json.loads(response.read().decode())
                for session in sessions:
                    if session['kernel']['id'] == kernel_id:
                        notebook_path = session['notebook']['path']
                        return os.path.join(server['notebook_dir'], notebook_path)
        except (OSError, KeyError, ValueError, urllib.error.URLError, json.JSONDecodeError):
            logger.debug("[UTILS] Failed to query sessions from server: %s", server.get('url', '?'))
    return None


def get_notebook_path() -> str | None:
    """Try to find the path of the current notebook.

    Priority:

    1. VS Code injected variable ``__vsc_ipynb_file__``
    2. ``ipynbname`` package (if installed)
    3. Jupyter Server REST API
    4. ``None`` (upstream checking disabled gracefully)
    """
    global _cached_notebook_path, _cached_notebook_path_time
    now = _time.monotonic()
    if _cached_notebook_path and (now - _cached_notebook_path_time) < _NOTEBOOK_PATH_CACHE_TTL:
        return _cached_notebook_path

    def _cache_and_return(path: str) -> str:
        global _cached_notebook_path, _cached_notebook_path_time
        _cached_notebook_path = path
        _cached_notebook_path_time = now
        return path

    if result := _try_vscode_path():
        return _cache_and_return(result)

    if result := _try_ipynbname_path():
        return _cache_and_return(result)

    try:
        import ipykernel
        connection_file = ipykernel.get_connection_file()
        kernel_id = os.path.basename(connection_file).split('-', 1)[1].split('.')[0]
    except (ImportError, AttributeError, OSError, RuntimeError):
        logger.debug("[UTILS] Failed to get kernel connection file")
        return None

    if result := _search_servers_for_notebook(kernel_id):
        return _cache_and_return(result)

    return None


def _wait_for_notebook_save(notebook_path: str) -> None:
    """Wait briefly if the notebook file was just saved (< 1 s ago).

    VS Code saves the notebook before executing a cell.  On cloud-synced
    drives or slow I/O the write may still be in progress, so we sleep
    a short time to let it complete.
    """
    try:
        age = _time.time() - os.path.getmtime(notebook_path)
        if age < 1.0:
            _time.sleep(0.3)
    except OSError:
        logger.debug("Cannot stat notebook file for freshness check: %s", notebook_path)


def _extract_cell_entry(cell: dict, include_ids: bool) -> str | tuple[str | None, str]:
    """Extract a single notebook code cell as a string or (id, string) tuple."""
    source = cell.get('source', [])
    if isinstance(source, list):
        source = "".join(source)
    if include_ids:
        cell_id = cell.get('id', cell.get('metadata', {}).get('id', None))
        return (cell_id, source)
    return source


def _read_notebook_code_cells(notebook_path: str | None = None, include_ids: bool = False) -> list[str] | list[tuple[str | None, str]]:
    """
    Read code cells from the notebook file.

    Args:
        notebook_path: Path to notebook file. Auto-detected if None.
        include_ids: If True, return list of (cell_id, code) tuples.
                     If False, return list of code strings.
    """
    if not notebook_path:
        notebook_path = get_notebook_path()

    if not notebook_path:
        # Do NOT use glob fallback - picking the most recently modified .ipynb
        # file is unreliable and can return the wrong notebook (Issue 23).
        # Return empty list so upstream checking is skipped gracefully.
        return []

    if not os.path.exists(notebook_path):
        return []

    try:
        _wait_for_notebook_save(notebook_path)

        with open(notebook_path, encoding='utf-8') as f:
            nb = json.load(f)

        return [
            _extract_cell_entry(cell, include_ids)
            for cell in nb.get('cells', [])
            if cell.get('cell_type') == 'code'
        ]
    except Exception as e:
        logger.error("Error reading notebook file: %s", e)
        return []


def get_notebook_cells(notebook_path: str | None = None) -> list[str]:
    """Read code cells from the notebook file. Returns a list of code strings."""
    return _read_notebook_code_cells(notebook_path, include_ids=False)


def get_notebook_cells_with_ids(notebook_path: str | None = None) -> list[tuple[str | None, str]]:
    """
    Read code cells with their IDs from the notebook file.
    Returns a list of (cell_id, code_string) tuples.
    """
    return _read_notebook_code_cells(notebook_path, include_ids=True)
