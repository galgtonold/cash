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
import sys
import time as _time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from urllib.parse import unquote

from ..exceptions import CashWarning

logger = logging.getLogger(__name__)


class CashNotebookDiscoveryWarning(CashWarning):
    """Notebook path discovery failed, so upstream dependency tracking is off.

    Emitted once per session when :func:`get_notebook_path` cannot locate the
    running notebook — typically under papermill / nbconvert / CI / scheduled
    jobs (no live Jupyter Server), or when a stale Jupyter runtime entry makes
    discovery time out. Statement-level caching still works; only cross-cell
    upstream staleness detection is disabled.

    Part of the :class:`~cash.exceptions.CashWarning` family, so the documented
    blanket filter silences it::

        warnings.filterwarnings("ignore", category=cash.CashWarning)
    """

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

# Negative (not-found) cache for notebook path discovery.
#
# Discovery that FAILS is even more expensive than one that succeeds: a stale or
# dead Jupyter runtime entry makes ``ipynbname`` / ``list_running_servers`` block
# on a network round-trip until it times out, then returns nothing.  The success
# cache above never covered this — every failed lookup re-probed from scratch —
# and the upstream checker resolves the path many times per cell, so a single
# ``run_all`` under slow-failing discovery paid that timeout dozens of times
# (measured ~78s to cache ``z = 1 + 1``).
#
# We memoize the failure too, but with a MUCH shorter TTL than the success case:
# long enough to dedupe the many resolves within one ``run_all`` down to a single
# probe, short enough that a Jupyter server started *after* cash loaded is still
# picked up on the next cell.  ``%cash_on`` (via invalidate_notebook_path_cache)
# and an explicit set_notebook_path() clear it immediately, so a real notebook is
# never shadowed by a stale negative for longer than the TTL.
#
# The timestamp is recorded when the probe COMPLETES (not when it started), so a
# slow probe whose own latency exceeds the TTL still dedupes correctly.
_negative_cache_time: float = 0.0  # monotonic time of last failed probe (0 = none)
_NOTEBOOK_PATH_NEGATIVE_TTL: float = 2.0  # seconds

# One-shot session flag for the "notebook not found" advisory.  Set the
# first time discovery fails during an upstream check; keeps the warning to once
# per session instead of once per cell.
_warned_notebook_not_found: bool = False


def warn_notebook_not_found_once() -> None:
    """Emit the "upstream tracking disabled" advisory at most once per session.

    Called by the upstream checker when :func:`get_notebook_path` returns
    ``None`` for a cell that would otherwise get dependency tracking.  A user
    must know cash's headline feature is off (papermill / nbconvert / CI, or a
    stale Jupyter runtime) rather than silently receiving no upstream checks.
    """
    global _warned_notebook_not_found
    if _warned_notebook_not_found:
        return
    _warned_notebook_not_found = True
    msg = (
        "Cash: notebook not found -- upstream dependency tracking is disabled "
        "for this session. Statement-level caching still works, but changes in "
        "one cell will not auto-invalidate dependent cells. This is expected "
        "under papermill / nbconvert / CI (no live Jupyter Server); in "
        "JupyterLab or VS Code it can mean a stale Jupyter runtime."
    )
    logger.warning(msg)
    warnings.warn(msg, category=CashNotebookDiscoveryWarning, stacklevel=2)


def reset_notebook_discovery_warning() -> None:
    """Re-arm the once-per-session "notebook not found" advisory (tests)."""
    global _warned_notebook_not_found
    _warned_notebook_not_found = False


def invalidate_notebook_path_cache() -> None:
    """
    Invalidate the cached notebook path so the next call to
    get_notebook_path() re-discovers the current notebook.
    This should be called when %cash_on is invoked to handle
    notebook switches within the same kernel session (Issue 23).
    """
    global _cached_notebook_path, _cached_notebook_path_time, _negative_cache_time
    _cached_notebook_path = None
    _cached_notebook_path_time = 0.0
    # Drop the negative (not-found) cache too, so a server that came up after
    # cash loaded is re-probed immediately on the next lookup instead of being
    # shadowed by a stale negative for the rest of its TTL.
    _negative_cache_time = 0.0
    # A notebook switch invalidates any cached cell parse for the old path too.
    invalidate_notebook_cells_cache()


def set_notebook_path(path: str) -> None:
    """
    Explicitly set the notebook path (bypassing auto-detection).

    This is called when the notebook path is reliably known from another
    source, e.g. extracted from a VS Code cell ID URI.  The value is
    stored in the same session-level cache used by ``get_notebook_path()``.
    """
    global _cached_notebook_path, _cached_notebook_path_time, _negative_cache_time
    if path and os.path.exists(path):
        _cached_notebook_path = path
        _cached_notebook_path_time = _time.monotonic()
        # A known-good path supersedes any prior not-found result.
        _negative_cache_time = 0.0
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
    """Return notebook path via the ipynbname package, or None.

    ``ipynbname.path()`` raises whatever its internal probing hits when no
    discoverable Jupyter server backs the kernel — notably ``IndexError`` from
    its ``_get_kernel_id`` (indexing an empty running-servers list) under Google
    Colab and other server-less runtimes. Discovery is best-effort, so ANY
    failure here must degrade to "no path found" rather than propagate: an
    uncaught error aborts the whole notebook-path resolution and trips the
    upstream pipeline's broad failure handler, which disables caching for the
    cell ("Cash auto-caching failed: list index out of range").
    """
    try:
        # ipynbname imports the notebook/jupyter_server packages internally, so
        # suppress the same third-party import-time warnings here too.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import ipynbname
            return str(ipynbname.path())
    except Exception:  # noqa: BLE001 - a discovery library must never crash cash
        logger.debug("[UTILS] ipynbname not available or failed")
    return None


def _collect_running_servers() -> list:
    """Return all running Jupyter server descriptors from all server packages."""
    servers: list = []
    # Importing these third-party server packages can emit warnings we neither
    # control nor want in the user's cell output — notably a ``SyntaxWarning:
    # invalid escape sequence '\/'`` from the (deprecated) ``notebook`` package's
    # banner on Python 3.12+, seen on Colab. cash triggers the import only for
    # discovery, so swallow those import-time warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
                        # ``notebook_dir`` is the CLASSIC notebook server's key.
                        # ``jupyter_server`` (i.e. every current JupyterLab) calls
                        # it ``root_dir``, so indexing ``notebook_dir`` raised a
                        # KeyError that this except tuple swallowed -- AFTER a
                        # perfectly successful 200 response. Discovery then
                        # returned None, upstream dependency tracking went off for
                        # the whole session, and an edited-but-not-re-run cell fed
                        # a stale value downstream with no error. Prefer the modern
                        # key and keep the legacy one as a fallback. Same for the
                        # session path: ``session['path']`` is current, the nested
                        # ``notebook`` dict is deprecated and may disappear.
                        notebook_path = (
                            session.get('path')
                            or session.get('notebook', {}).get('path')
                        )
                        root_dir = (
                            server.get('root_dir')
                            or server.get('notebook_dir')
                            or ''
                        )
                        if not notebook_path:
                            logger.debug(
                                "[UTILS] Session for kernel %s carries no path", kernel_id,
                            )
                            continue
                        return os.path.join(root_dir, notebook_path)
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
            # Deliberately no longer catches KeyError blind. A missing key is a
            # SCHEMA mismatch, not a transport failure, and reporting it as
            # "failed to query" sent two independent testers hunting the network
            # while the request had actually returned 200.
            logger.debug(
                "[UTILS] Failed to query sessions from server %s: %s",
                server.get('url', '?'), exc,
            )
        except KeyError as exc:
            logger.debug(
                "[UTILS] Session payload from %s is missing key %s -- "
                "unrecognised Jupyter server schema",
                server.get('url', '?'), exc,
            )
    return None


def _kernel_id_from_connection_file(connection_file: str | None) -> str | None:
    """Return the kernel id from a ``kernel-<id>.json`` connection filename.

    Returns ``None`` — never raises — for any filename that carries no id.
    Under bare nbclient / papermill / nbconvert the connection file can be a
    plain ``kernel.json`` with no ``-``, and indexing the split blindly
    (``.split('-', 1)[1]``) raised an ``IndexError`` that was NOT in the caller's
    except tuple. It escaped ``get_notebook_path``, disabled caching for the whole
    run, and printed "Cash auto-caching failed: list index out of range" on EVERY
    cell instead of cash's intended one-time "notebook not found" disclosure.

    Semantics are otherwise identical to the original expression: split on the
    FIRST ``-`` (so a UUID's own dashes are preserved) and drop at the first ``.``.
    """
    base = os.path.basename(connection_file or "")
    _, sep, rest = base.partition("-")
    if not sep:
        return None
    return rest.split(".")[0] or None


def get_notebook_path() -> str | None:
    """Try to find the path of the current notebook.

    Priority:

    1. VS Code injected variable ``__vsc_ipynb_file__``
    2. ``ipynbname`` package (if installed)
    3. Jupyter Server REST API
    4. ``None`` (upstream checking disabled gracefully)
    """
    global _cached_notebook_path, _cached_notebook_path_time, _negative_cache_time
    now = _time.monotonic()
    if _cached_notebook_path and (now - _cached_notebook_path_time) < _NOTEBOOK_PATH_CACHE_TTL:
        return _cached_notebook_path
    # Negative cache: a recent failed probe short-circuits to None without
    # re-paying the (possibly multi-second) discovery timeout.  Checked AFTER the
    # success cache so a resolved path always wins.
    if _negative_cache_time and (now - _negative_cache_time) < _NOTEBOOK_PATH_NEGATIVE_TTL:
        return None

    def _cache_and_return(path: str) -> str:
        global _cached_notebook_path, _cached_notebook_path_time, _negative_cache_time
        _cached_notebook_path = path
        _cached_notebook_path_time = now
        _negative_cache_time = 0.0  # a success supersedes any prior not-found
        return path

    def _cache_not_found() -> None:
        # Record completion time (a fresh monotonic read, NOT the stale ``now``
        # captured before the probe): a probe slower than the TTL must still
        # dedupe, and it can only do so if the window starts when it finishes.
        global _negative_cache_time
        _negative_cache_time = _time.monotonic()
        return None

    if result := _try_vscode_path():
        return _cache_and_return(result)

    if result := _try_ipynbname_path():
        return _cache_and_return(result)

    try:
        import ipykernel
        connection_file = ipykernel.get_connection_file()
        kernel_id = _kernel_id_from_connection_file(connection_file)
    except (ImportError, AttributeError, OSError, RuntimeError, IndexError, ValueError):
        logger.debug("[UTILS] Failed to get kernel connection file")
        return _cache_not_found()

    if not kernel_id:
        # No id in the filename -> nothing to match a server session against.
        logger.debug("[UTILS] No kernel id in connection file %r", connection_file)
        return _cache_not_found()

    if result := _search_servers_for_notebook(kernel_id):
        return _cache_and_return(result)

    return _cache_not_found()


# Tunables for the save-settle wait (see _wait_for_notebook_save).
_SAVE_FRESH_WINDOW_S: float = 1.0      # only wait when the file changed this recently
_SAVE_POLL_INTERVAL_S: float = 0.05    # re-stat cadence while a write is in flight
_SAVE_MAX_WAIT_S: float = 1.0          # hard cap so we never block a run for long


def _wait_for_notebook_save(notebook_path: str) -> None:
    """Block briefly while an in-flight notebook save settles, then return.

    The editor executes from its in-memory model and may write the ``.ipynb`` to
    disk slightly *after* the run begins; on a cloud-synced drive or slow I/O the
    write can still be in progress when we read.  Reading mid-write yields stale
    (pre-edit) cell sources, which silently breaks upstream change detection — the
    bug this guards against.

    There is no guaranteed save-before-execute to wait on (VS Code's Jupyter
    extension has no "autosave on run"; classic Jupyter autosaves on a long
    timer), so a fixed blind sleep is the wrong tool: when nothing is being
    written it only adds latency, and when the write is slow a fixed sleep may be
    too short and still read stale.  Instead, *only* when the file looks freshly
    touched, poll until its ``(mtime, size)`` stops changing — i.e. the writer is
    done — bounded by ``_SAVE_MAX_WAIT_S``.  A long-settled file returns at once.
    """
    try:
        st = os.stat(notebook_path)
    except OSError:
        logger.debug("Cannot stat notebook file for freshness check: %s", notebook_path)
        return

    if _time.time() - st.st_mtime >= _SAVE_FRESH_WINDOW_S:
        return  # settled long ago — nothing in flight to wait for

    deadline = _time.monotonic() + _SAVE_MAX_WAIT_S
    last = (st.st_mtime_ns, st.st_size)
    while _time.monotonic() < deadline:
        _time.sleep(_SAVE_POLL_INTERVAL_S)
        try:
            st = os.stat(notebook_path)
        except OSError:
            return
        current = (st.st_mtime_ns, st.st_size)
        if current == last:
            return  # writer is done — file is stable
        last = current


def _extract_cell_entry(cell: dict, include_ids: bool) -> str | tuple[str | None, str]:
    """Extract a single notebook code cell as a string or (id, string) tuple."""
    source = cell.get('source', [])
    if isinstance(source, list):
        source = "".join(source)
    if include_ids:
        cell_id = cell.get('id', cell.get('metadata', {}).get('id', None))
        return (cell_id, source)
    return source


# Session-level cache of parsed notebook cells, keyed by (path, include_ids).
# Value is (mtime_ns, size, cells).  The (mtime_ns, size) signature is the file's
# own identity, so the cache can never serve content for a different file state —
# any save bumps the signature, forcing a fresh read.  This collapses the many
# reads per cell run (the magic reads once; the upstream checker reads twice) into
# a single parse per file state, and means the save-settle wait runs at most once
# per change instead of once per read.  Cleared on notebook switch.
_notebook_cells_cache: dict[tuple[str, bool], tuple[int, int, list]] = {}


def invalidate_notebook_cells_cache() -> None:
    """Drop the parsed-notebook-cells cache (e.g. on notebook switch / %cash_on)."""
    _notebook_cells_cache.clear()
    _colab_cells_cache.clear()
    _vscode_cells_cache.clear()


# --- JupyterLab live-cell push cell source ------------------------------------
# JupyterLab gives a kernel no route to its live document at all -- unlike
# Colab's frontend API or VS Code's hot-exit backup, there is nothing on disk
# or reachable over an existing channel to read. cash's own extension closes
# that gap by pushing the notebook's current cell sources over a comm; see
# ``cash.notebook.live_cells`` for the receiving half and why it is a push
# rather than a request/response.
def _try_extension_cells(include_ids: bool) -> list | None:
    """Cells pushed by cash's JupyterLab extension, or ``None``.

    Same contract as the Colab and VS Code readers: ``None`` means "not
    available" and the caller falls through to the saved file. Needs no
    environment gate -- a push can only exist if the extension is running.
    """
    try:
        from cash.notebook.live_cells import latest_cells
        cells = latest_cells()
    except Exception as e:  # noqa: BLE001
        logger.debug("[UTILS] extension cell read failed: %s", e)
        return None
    if not cells:
        return None
    try:
        extracted = [
            _extract_cell_entry(cell, include_ids)
            for cell in cells
            if cell.get("cell_type") == "code"
        ]
    except Exception as e:  # noqa: BLE001
        logger.debug("[UTILS] extension cell shape unusable: %s", e)
        return None
    # An unrecognised shape must fall through to the file, never disable the
    # upstream check by returning an empty list -- exactly that made cash
    # strictly worse than not having the feature in CAS-274 Tier 1c.
    return extracted or None


#: Where a prebuilt labextension lives, relative to an environment's data root.
#: Fixed by the wheel's ``shared-data`` mapping in pyproject.toml, which
#: tests/test_notebook/test_labextension_packaging.py pins to this same name.
_LABEXT_RELPATH = ("share", "jupyter", "labextensions", "cash-live-cells")


def _labextension_installed() -> bool:
    """True when cash's prebuilt JupyterLab extension is installed in this env.

    A FILESYSTEM check, not a runtime one, and deliberately so. The only caller
    is ``%cash_on``, which necessarily runs before the extension's comm can
    exist: on a fresh kernel the first ``comm_open`` is refused because the
    target is not registered until ``import cash`` has run (see
    ``live_cells.register_target``), so ``latest_cells()`` is ``None`` at that
    moment for EVERY user and cannot tell "absent" from "not yet". The wheel
    drops the extension at a fixed path under the environment's data root, and
    that is true the instant cash is importable.

    Wrong in exactly one topology: a SPLIT INSTALL, where the JupyterLab server
    and the kernel live in different environments -- this answers for the
    kernel's. Accepted, because the only caller uses it to SUPPRESS a proactive
    hint: the cost is a missing sentence, never a false claim, and the reactive
    once-per-session "cash cannot see unsaved edits here" notice still fires
    correctly there, since it keys on the read that actually happened rather
    than on an inference.

    A probe that cannot answer returns ``False`` -- "assume no extension".
    That is the direction that degrades well at scale: if this check ever
    breaks wholesale (a packaging change moving the directory), everyone keeps
    today's advice, which is right for the majority who have no extension,
    instead of it being silently withdrawn from all of them.
    """
    try:
        import site
        roots = [sys.prefix]
        user_base = site.getuserbase()
        if user_base:
            roots.append(user_base)
        return any(os.path.isdir(os.path.join(root, *_LABEXT_RELPATH))
                   for root in roots)
    except Exception as e:  # noqa: BLE001 - a hint gate must never break %cash_on
        logger.debug("[UTILS] labextension probe failed: %s", e)
        return False


# --- Google Colab cell source ------------------------------------------------
# Colab notebooks live in Google Drive, not a local file, so the file reader
# below finds nothing: get_notebook_path() returns a "/fileId=..." reference
# that is not openable, and reading it yields []. The Colab frontend instead
# exposes the LIVE notebook JSON via google.colab._message, so we read the
# current cells from there. Cached for a short TTL because upstream resolution
# reads cells several times per cell run and each call is a frontend round-trip;
# a failure is cached longer so a frontend-less runtime (scheduled / headless
# execution, where the request would block until timeout) does not pay that on
# every cell.
_COLAB_GET_IPYNB_TIMEOUT = 5.0   # seconds to wait for the frontend to answer
_COLAB_CELLS_TTL = 2.0           # reuse a successful read across one resolution
_COLAB_FAIL_TTL = 30.0           # back off after a failure (no frontend / timeout)
_colab_cells_cache: dict[bool, tuple[float, list | None]] = {}


def _in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def _try_colab_notebook_cells(include_ids: bool) -> list | None:
    """Return code cells via Colab's ``get_ipynb`` frontend API, or ``None``.

    ``None`` means "not applicable / unavailable" — either not running in Colab,
    or the frontend request failed — so the caller falls through to the
    file-based reader. A success returns the current code cells (markdown
    filtered) in the same shape as the file reader, so the rest of the pipeline
    is unchanged.
    """
    if not _in_colab():
        return None
    now = _time.monotonic()
    cached = _colab_cells_cache.get(include_ids)
    if cached is not None:
        ts, val = cached
        ttl = _COLAB_CELLS_TTL if val is not None else _COLAB_FAIL_TTL
        if now - ts < ttl:
            return val
    try:
        from google.colab import _message  # type: ignore[import-not-found]
        resp = _message.blocking_request("get_ipynb", timeout_sec=_COLAB_GET_IPYNB_TIMEOUT)
        nb = resp.get("ipynb") if isinstance(resp, dict) else None
        if not isinstance(nb, dict):
            raise ValueError("unexpected get_ipynb response shape")
        cells = [
            _extract_cell_entry(cell, include_ids)
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
    except Exception as e:  # noqa: BLE001 - the Colab frontend API is best-effort
        logger.debug("[UTILS] Colab get_ipynb failed: %s", e)
        _colab_cells_cache[include_ids] = (now, None)  # back off; don't block every cell
        return None
    _colab_cells_cache[include_ids] = (now, cells)
    return cells


# --- VS Code hot-exit backup cell source -------------------------------------
# Session-level cache, same shape and purpose as _notebook_cells_cache above:
# keyed by (path, include_ids), a hit is validated by two cheap os.stat calls
# instead of re-scanning the backup directory. Two things make this cache
# unlike _notebook_cells_cache, though. First, it must be validated against
# BOTH signatures, not just the file's: live_cells() only returns cells when
# the backup's header pairs with the file on disk, so caching on the file's
# signature alone would freeze the first read of an unsaved edit forever --
# the file's own signature does not change again until the user actually
# saves, even though the backup keeps changing underneath it as they keep
# typing. Second, a "no backup found" result is never cached (see the miss
# path below): unlike a missing file, which os.path.exists rules out in one
# cheap stat, "no backup" can only be established by the directory scan
# itself, so there is no cheap signature to invalidate on and caching it
# would risk never noticing a backup that appears later in the session.
_vscode_cells_cache: dict[tuple[str, bool], tuple[Path, tuple[int, int], tuple[int, int], list]] = {}


def _stat_sig(path: str | Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` for *path*, or ``None`` if it cannot be stat'd."""
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _in_vscode() -> bool:
    """True when this kernel currently carries VS Code's injected signal.

    Reuses ``_try_vscode_path()``'s own detection (the ``__vsc_ipynb_file__``
    variable VS Code injects into ``user_ns``) rather than a second,
    independent check -- it is a plain in-process dict lookup with no I/O, so
    it is free to call again here even though ``get_notebook_path()`` may
    already have consulted it earlier in the same resolution (and, on a cache
    hit, may not call it again at all).
    """
    return _try_vscode_path() is not None


def _try_vscode_backup_cells(notebook_path: str | None, include_ids: bool) -> list | None:
    """Return code cells from VS Code's hot-exit backup, or ``None``.

    Same contract as the Colab reader above: ``None`` means "not applicable /
    unavailable" and the caller falls through to the file. VS Code gives a
    kernel no route to its live document -- the widget webview holds a
    `standaloneModel` stub with an empty cell list -- but it does persist dirty
    editors to disk, and that backup carries the unsaved edits the file does not.

    Gated on ``_in_vscode()`` first: a backup on disk is evidence some VS Code
    window once had this notebook open, not that THIS kernel is the one being
    looked at right now. Without the gate, a plain JupyterLab kernel pointed at
    a file some VS Code window still holds dirty would silently read edits that
    are not on the screen the user is looking at -- and a hot-exit backup can
    outlive the VS Code session that wrote it by weeks, so its mere existence
    is never on its own evidence of anything current.

    Memoized against both the backup's and the file's signature (see
    ``_vscode_cells_cache`` above) so that the ~3 calls a single cell run makes
    (the magic once, the upstream checker twice) pay for the backup-directory
    scan and the settle wait at most once per actual change, not once per call
    -- an unmemoized version was measured to turn the settle wait's 1.5s cap
    into up to 3x that across one run, since each call waited independently.
    """
    if not _in_vscode():
        return None
    if not notebook_path:
        return None

    cache_key = (notebook_path, include_ids)
    file_sig = _stat_sig(notebook_path)
    cached = _vscode_cells_cache.get(cache_key)
    if cached is not None and file_sig is not None:
        cached_backup, cached_backup_sig, cached_file_sig, cached_cells = cached
        if cached_file_sig == file_sig and _stat_sig(cached_backup) == cached_backup_sig:
            return cached_cells

    try:
        from cash.notebook.vscode_backup import find_backup, live_cells

        cells = live_cells(notebook_path)
        if cells is None:
            _vscode_cells_cache.pop(cache_key, None)
            return None

        # The comprehension below is inside this guard deliberately: a cell
        # entry is another product's undocumented format, and a malformed
        # (non-dict) entry must degrade to None like everything else here,
        # not raise out of a caller that has no reason to expect it.
        extracted = [
            _extract_cell_entry(cell, include_ids)
            for cell in cells
            if cell.get("cell_type") == "code"
        ]

        if not extracted:
            # [] is not None: returned as-is it reads as "these ARE the live
            # cells" and the caller (_read_notebook_code_cells) would never
            # fall through to the file. That is indistinguishable from a
            # genuinely all-markdown notebook UNLESS every entry failed to
            # match cell_type == "code" -- exactly what happens when a backup
            # uses VS Code's OTHER notebook serialization (observed:
            # {"cells": [{"kind": 2, "language": "python", "value": ...}]},
            # keyed "kind"/"value" rather than "cell_type"/"source"). Treat an
            # empty extraction as unusable so the file -- which reports the
            # same [] for an actually-empty notebook -- is the one source of
            # truth for "no code cells", never this comprehension by omission.
            _vscode_cells_cache.pop(cache_key, None)
            return None

        backup = find_backup(notebook_path)
        backup_sig = _stat_sig(backup) if backup is not None else None
        if backup is not None and backup_sig is not None and file_sig is not None:
            _vscode_cells_cache[cache_key] = (backup, backup_sig, file_sig, extracted)
        return extracted
    except Exception as e:  # noqa: BLE001 - another product's private format
        logger.debug("[UTILS] VS Code backup read failed: %s", e)
        return None


#: Which reader supplied the most recent cell read, set by
#: ``_read_notebook_code_cells`` and read by ``last_cell_source``. Session-level
#: (module) state, same lifetime as the caches above.
_last_cell_source: str | None = None


def last_cell_source() -> str | None:
    """Which reader supplied the most recent cell read.

    ``"extension"`` / ``"colab"`` / ``"vscode-backup"`` see unsaved edits;
    ``"file"`` does not, which is what the once-per-session "cannot verify"
    notice keys on. ``None`` means no read has happened yet this session.
    """
    return _last_cell_source


def _read_notebook_code_cells(notebook_path: str | None = None, include_ids: bool = False) -> list[str] | list[tuple[str | None, str]]:
    """
    Read code cells from the first reader in the chain that can answer.

    Tried in order, each falling through to the next on ``None``: cash's own
    JupyterLab extension (``_try_extension_cells``), Google Colab
    (``_try_colab_notebook_cells``), VS Code's hot-exit backup
    (``_try_vscode_backup_cells``), and finally the saved ``.ipynb`` on disk.
    The first three see UNSAVED edits; the file does not. ``last_cell_source()``
    reports which one actually answered a given call.

    The extension and Colab readers ignore ``notebook_path`` entirely -- the
    extension keys off the running kernel, not a path, and a Colab notebook is
    a Drive fileId, not a local file. Only the VS Code and file readers use it:
    VS Code to match a backup against the right notebook, the file reader to
    open it.

    The file read (the last resort) is memoized by the file's ``(mtime_ns,
    size)`` signature: repeated reads of an unchanged file return the cached
    parse without re-opening it, while any edit (which bumps mtime/size)
    forces a fresh read — so a quick edit-then-run never serves stale cell
    sources from the file. That caching is local to the file path; none of the
    three live readers ahead of it in the chain go through it.

    Args:
        notebook_path: Path to notebook file, used only by the VS Code and
                        file readers (see above). Auto-detected if None.
        include_ids: If True, return list of (cell_id, code) tuples.
                     If False, return list of code strings.
    """
    global _last_cell_source
    # Default to the pessimistic answer. Every exit below except the three live
    # readers ends up reading (or failing to read) the saved file, and a failed
    # read is emphatically not a verified-fresh one -- so only an actual live
    # hit may upgrade this.
    _last_cell_source = "file"

    # cash's own JupyterLab extension first, when present: it is the most
    # authoritative source (the frontend pushed it moments ago) and checking
    # costs one dict lookup when there is no extension to push anything.
    extension_cells = _try_extension_cells(include_ids)
    if extension_cells is not None:
        _last_cell_source = "extension"
        return extension_cells

    # Colab next: its notebook is a Drive fileId, not a local file, so the
    # file reader below returns []. Reading the live cells from the Colab
    # frontend is what makes upstream resolution work there. A fast no-op when
    # not running in Colab.
    colab_cells = _try_colab_notebook_cells(include_ids)
    if colab_cells is not None:
        _last_cell_source = "colab"
        return colab_cells

    if not notebook_path:
        notebook_path = get_notebook_path()

    # VS Code next. The position here is load-bearing on both sides: after
    # notebook_path is resolved, since the reader needs a path to match a
    # backup against; before the "no notebook_path" early return just below,
    # since moving it after that return would mean this path never runs for
    # the common case of a caller that passes no explicit path.
    vscode_cells = _try_vscode_backup_cells(notebook_path, include_ids)
    if vscode_cells is not None:
        _last_cell_source = "vscode-backup"
        return vscode_cells

    if not notebook_path:
        # Do NOT use glob fallback - picking the most recently modified .ipynb
        # file is unreliable and can return the wrong notebook (Issue 23).
        # Return empty list so upstream checking is skipped gracefully.
        return []

    if not os.path.exists(notebook_path):
        return []

    cache_key = (notebook_path, include_ids)
    try:
        st = os.stat(notebook_path)
        signature = (st.st_mtime_ns, st.st_size)
    except OSError:
        signature = None

    if signature is not None:
        cached = _notebook_cells_cache.get(cache_key)
        if cached is not None and (cached[0], cached[1]) == signature:
            return cached[2]

    try:
        # Reached only on a new/changed file state: let an in-flight save settle
        # before reading so we never parse a half-written (stale) notebook.
        _wait_for_notebook_save(notebook_path)

        with open(notebook_path, encoding='utf-8') as f:
            nb = json.load(f)

        cells = [
            _extract_cell_entry(cell, include_ids)
            for cell in nb.get('cells', [])
            if cell.get('cell_type') == 'code'
        ]
    except Exception as e:
        logger.error("Error reading notebook file: %s", e)
        return []

    # Key by the POST-wait file state — the wait may have let a newer save land —
    # so subsequent reads of the now-settled file hit the cache.
    try:
        st2 = os.stat(notebook_path)
        _notebook_cells_cache[cache_key] = (st2.st_mtime_ns, st2.st_size, cells)
    except OSError:
        logger.debug("Cannot stat notebook file to cache cells: %s", notebook_path)

    return cells


def get_notebook_cells(notebook_path: str | None = None) -> list[str]:
    """Read code cells from the first reader that can answer -- cash's
    JupyterLab extension, Colab, a VS Code hot-exit backup, or (last resort)
    the saved notebook file. Returns a list of code strings. See
    ``_read_notebook_code_cells`` for the reader chain and ``last_cell_source``
    for which one answered."""
    return _read_notebook_code_cells(notebook_path, include_ids=False)


def get_notebook_cells_with_ids(notebook_path: str | None = None) -> list[tuple[str | None, str]]:
    """
    Read code cells with their IDs from the notebook file.
    Returns a list of (cell_id, code_string) tuples.
    """
    return _read_notebook_code_cells(notebook_path, include_ids=True)
