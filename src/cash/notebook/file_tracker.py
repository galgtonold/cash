from __future__ import annotations

"""File-read interception for tracking data file dependencies.

Monkey-patches common I/O functions (``open``, ``pandas.read_csv``,
``numpy.load``, etc.) to record which files each statement reads.
File hashes are incorporated into cache keys so that changed data
automatically invalidates dependent cached results.
"""

import builtins
import contextvars
import importlib
import importlib.abc
import importlib.util
import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any, Optional

from cash.utils import normalize_path

__all__ = ["FileDependencyRegistry", "PostImportHook", "FileAccessTracker", "FileDependencies"]

# Type alias for file dependency tracking: maps normalized file path -> mtime at read time
FileDependencies = dict[str, float]

logger = logging.getLogger(__name__)

# Active tracker for the current asyncio task / thread.
# Read by the patched I/O dispatchers to decide whether to record the
# access. Isolated per task/thread by contextvars semantics.
_active_tracker: contextvars.ContextVar[Optional["FileAccessTracker"]] = (
    contextvars.ContextVar("_active_tracker", default=None)
)

# Per-target install lock: the dispatcher wrappers are installed once
# per (module/dict, attr-name) pair for the lifetime of the process.
# Subsequent ``__enter__`` calls skip already-installed targets (cheap
# attribute-sentinel check). New targets — i.e. modules imported after
# the first tracker entered, or new handlers registered via
# :func:`cash.register_file_handler` — are still picked up because
# ``_apply_patches`` runs on every ``__enter__`` and only the
# already-installed wrappers are no-op skipped.
_install_lock = threading.Lock()


def _dispatch_track(path: Any) -> None:
    """Module-level tracker-dispatching shim. Custom handler factories
    registered via :func:`cash.register_file_handler` receive this as
    their ``tracker_callback`` argument. The shim consults
    ``_active_tracker`` at *call* time, so old-signature factories
    (whose wrappers do ``tracker_callback(path)``) transparently route
    to whichever tracker is active on the current asyncio task or
    thread — same isolation guarantees as the built-in handlers.
    """
    _tracker = _active_tracker.get()
    if _tracker is not None:
        _tracker._track_path(path)

def _find_patch_targets(func_pattern: str, module_obj: Any) -> list:
    """Return the list of attribute names to patch on *module_obj*."""
    if func_pattern.endswith('*'):
        prefix = func_pattern[:-1]
        return [name for name in dir(module_obj) if name.startswith(prefix)]
    if hasattr(module_obj, func_pattern):
        return [func_pattern]
    return []


def _install_module_patches(module_name: str, module_obj: Any) -> None:
    """Install dispatcher patches on a module. Idempotent — skips any
    target whose current attribute is already a dispatcher wrapper.

    Called from :meth:`FileAccessTracker._apply_patches` and from
    :class:`_PatchingLoader.exec_module` (post-import). The dispatcher
    wrappers route via ``_active_tracker`` so they're tracker-agnostic
    — one install serves all trackers.
    """
    registry = FileDependencyRegistry()
    handlers = registry.get_handlers_for_module(module_name)

    for func_pattern, factory in handlers:
        targets = _find_patch_targets(func_pattern, module_obj)
        for name in targets:
            if not hasattr(module_obj, name):
                continue
            original_func = getattr(module_obj, name)

            # Install-once skip.
            if getattr(original_func, '_is_file_tracker_patch', False):
                continue

            real_original = _unwrap_to_real(original_func)
            if not callable(real_original):
                continue

            wrapper = factory(real_original, _dispatch_track)
            wrapper._is_file_tracker_patch = True
            wrapper._original_func = real_original
            try:
                setattr(module_obj, name, wrapper)
            except (AttributeError, TypeError) as e:
                logger.debug("[FILE_TRACKER] Failed to patch %s.%s: %s", module_name, name, e)


def _unwrap_to_real(func: Any) -> Any:
    """Walk a chain of FileAccessTracker wrappers down to the original
    callable. Returns ``func`` unchanged if it isn't a wrapper.

    Used by :func:`_install_module_patches` and
    :meth:`FileAccessTracker._patch_user_ns` so a fresh tracker can
    self-heal past wrappers left behind by a
    prior tracker that failed to unpatch (e.g. an exception during
    ``__exit__``, an orphaned tracker, etc.). Without this, the sentinel
    check ``_is_file_tracker_patch`` would cause the new tracker to skip
    the function entirely, leaving the leaked wrapper installed
    indefinitely and pinning a dead tracker instance in memory via its
    closure on ``_track_path``.
    """
    seen: set[int] = set()
    while getattr(func, '_is_file_tracker_patch', False):
        if id(func) in seen:  # broken/circular chain, bail
            break
        seen.add(id(func))
        next_func = getattr(func, '_original_func', None)
        if next_func is None or not callable(next_func):
            break
        func = next_func
    return func


class FileDependencyRegistry:
    """
    Registry for file dependency handlers.
    Allows easy extension of file tracking to new libraries and functions.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.handlers = {} # Map module -> list of (func_name, handler_factory)
            cls._instance._initialize_defaults()
        return cls._instance

    def _initialize_defaults(self):
        """Initialize default handlers for common libraries."""
        # Builtins
        self.register('builtins', 'open', self._create_open_handler)

        # io (used by pathlib)
        self.register('io', 'open', self._create_open_handler)

        # Pandas
        self.register('pandas', 'read_*', self._create_path_arg_handler)

        # Polars
        self.register('polars', 'read_csv', self._create_path_arg_handler)
        self.register('polars', 'read_parquet', self._create_path_arg_handler)
        self.register('polars', 'read_json', self._create_path_arg_handler)
        self.register('polars', 'read_ndjson', self._create_path_arg_handler)
        self.register('polars', 'read_ipc', self._create_path_arg_handler)
        self.register('polars', 'read_avro', self._create_path_arg_handler)
        self.register('polars', 'read_excel', self._create_path_arg_handler)
        self.register('polars', 'scan_csv', self._create_path_arg_handler)
        self.register('polars', 'scan_parquet', self._create_path_arg_handler)
        self.register('polars', 'scan_ipc', self._create_path_arg_handler)
        self.register('polars', 'scan_ndjson', self._create_path_arg_handler)

        # Numpy
        self.register('numpy', 'load', self._create_path_arg_handler)
        self.register('numpy', 'loadtxt', self._create_path_arg_handler)
        self.register('numpy', 'genfromtxt', self._create_path_arg_handler)
        self.register('numpy', 'fromfile', self._create_path_arg_handler)

        # Joblib
        self.register('joblib', 'load', self._create_path_arg_handler)

        # Pickle
        self.register('pickle', 'load', self._create_path_arg_handler)

        # Json
        self.register('json', 'load', self._create_path_arg_handler)

        # Directory listing (CAS-85): a cell that enumerates a directory and
        # reads the matches gets file-deps only for the files READ on the first
        # run, so a NEW matching file is invisible. Track the enumerated
        # directory itself as a dependency - adding/removing an entry bumps the
        # directory's own mtime on local filesystems, so the existing mtime
        # freshness check invalidates the reader.
        self.register('glob', 'glob', self._create_glob_dir_handler)
        self.register('glob', 'iglob', self._create_glob_dir_handler)
        self.register('os', 'listdir', self._create_listdir_handler)
        self.register('os', 'scandir', self._create_listdir_handler)

    def register(self, module_name: str, func_name: str, handler_factory: Callable[..., Any]):
        """
        Register a file tracking handler for a specific function.

        Args:
            module_name: Name of the module (e.g., 'pandas', 'builtins').
            func_name: Name of the function to track. Supports wildcards like 'read_*'.
            handler_factory: A function that takes (original_function, tracker_callback)
                             and returns a wrapper function.
        """
        if module_name not in self.handlers:
            self.handlers[module_name] = []
        self.handlers[module_name].append((func_name, handler_factory))

    def get_handlers_for_module(self, module_name: str) -> list[tuple[str, Callable[..., Any]]]:
        return self.handlers.get(module_name, [])

    # --- Standard Handler Factories ---

    @staticmethod
    def _create_open_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Handler for open()-like functions.

        ``track_callback`` is part of the user-facing handler factory
        signature (see :meth:`FileDependencyRegistry.register`) so
        custom factories can record the access. The built-in handlers
        ignore the argument and consult ``_active_tracker`` directly —
        that way one patch serves any number of concurrent trackers.
        """
        def tracked_open(file, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            if 'r' in mode or '+' in mode:
                _tracker = _active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(file)
            return original_func(file, *args, **kwargs)
        return tracked_open

    @staticmethod
    def _create_path_arg_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Generic handler for functions where the first argument is a path.

        ``track_callback`` is part of the user-facing factory signature
        — see :meth:`_create_open_handler`. The built-in wrapper
        consults ``_active_tracker`` directly.
        """
        def tracked_func(path_or_buf, *args, **kwargs):
            if isinstance(path_or_buf, (str, bytes, os.PathLike)):
                _tracker = _active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(path_or_buf)
            return original_func(path_or_buf, *args, **kwargs)
        return tracked_func

    @staticmethod
    def _glob_base_dir(pattern: Any) -> str | None:
        """Return the longest leading, magic-free directory of a glob *pattern*.

        ``gdir/*.num`` → ``gdir``; ``a/b*/c`` → ``a`` (deepest stable ancestor).
        The directory's mtime is what we track for membership changes.
        """
        import glob as _glob
        try:
            parts = str(pattern).replace('\\', '/').split('/')
        except (TypeError, ValueError):
            return None
        base: list[str] = []
        for p in parts[:-1]:  # exclude the filename component
            if _glob.has_magic(p):
                break
            base.append(p)
        return '/'.join(base) or '.'

    @staticmethod
    def _create_glob_dir_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Track the directory a ``glob`` pattern enumerates (CAS-85)."""
        def tracked_glob(pathname, *args, **kwargs):
            _tracker = _active_tracker.get()
            if _tracker is not None:
                base = FileDependencyRegistry._glob_base_dir(pathname)
                if base is not None:
                    _tracker._track_path(base)
            return original_func(pathname, *args, **kwargs)
        return tracked_glob

    @staticmethod
    def _create_listdir_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Track the directory passed to ``os.listdir`` / ``os.scandir`` (CAS-85)."""
        def tracked_listdir(path='.', *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                _tracker = _active_tracker.get()
                if _tracker is not None:
                    _tracker._track_path(path)
            return original_func(path, *args, **kwargs)
        return tracked_listdir

class PostImportHook(importlib.abc.MetaPathFinder):
    """Intercepts imports of registered modules to patch them after loading.

    A single shared hook is installed once on ``sys.meta_path`` (see
    ``_shared_import_hook`` below). Module patching is tracker-agnostic
    — :func:`_install_module_patches` routes file reads via
    ``_active_tracker`` so the same patches serve every tracker.
    """
    def __init__(self) -> None:
        self._skip: set[str] = set()  # Avoid recursion

    def find_spec(self, fullname, path, target=None):
        if fullname in self._skip:
            return None

        # Only interest in registered modules
        # Note: We match top-level packages mainly.
        # e.g. 'pandas.io' -> we patch 'pandas' too?
        # The handlers are registered by module name.
        top_level = fullname.split('.')[0]

        targets = FileDependencyRegistry().handlers.keys()
        if fullname not in targets and top_level not in targets:
             return None

        # It's a target. We need to let the real import happen, then patch.
        self._skip.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname, path)
        finally:
            self._skip.remove(fullname)

        if spec is None or spec.loader is None:
            return None

        # Wrap the loader
        spec.loader = _PatchingLoader(spec.loader, fullname)
        return spec

class _PatchingLoader:
    def __init__(self, original_loader, fullname):
        self.original_loader = original_loader
        self.fullname = fullname

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        # execute module
        self.original_loader.exec_module(module)

        # Now patch it via the module-level dispatcher installer —
        # tracker-agnostic, install-once-per-target.
        _install_module_patches(self.fullname, module)


# Shared meta_path hook — installed at most once for the lifetime of
# the process by :meth:`FileAccessTracker.__enter__`. Tracker-agnostic.
_shared_import_hook: Optional[PostImportHook] = None
_shared_import_hook_lock = threading.Lock()


def _ensure_import_hook_installed() -> None:
    global _shared_import_hook
    with _shared_import_hook_lock:
        if _shared_import_hook is None:
            _shared_import_hook = PostImportHook()
            sys.meta_path.insert(0, _shared_import_hook)

class FileAccessTracker:
    """Context manager that intercepts file I/O to record which files a
    statement reads.

    **ContextVar dispatch (current design)**: On the first
    ``__enter__`` ever, install once-permanently dispatcher wrappers on
    ``builtins.open``, registered pandas/polars/numpy/joblib/pickle/json
    I/O functions, the user namespace ``open``, and a meta-path import
    hook for libraries loaded later. The wrappers consult a
    ``ContextVar`` (``_active_tracker``) at *call* time to decide
    whether to record the access. ``__enter__`` sets that ContextVar
    to ``self`` and stores the token; ``__exit__`` ``reset()``s it.

    Because ``ContextVar`` values are isolated per ``asyncio.Task`` and
    per ``threading.Thread`` by default, concurrent trackers — e.g. two
    coroutines under ``asyncio.gather``, or two worker threads — each
    see only their own block's reads.

    **No per-tracker unpatching**: the wrappers stay installed for the
    lifetime of the process. With no active tracker the dispatcher is a
    single ``ContextVar.get()`` + ``is None`` check before falling
    through to the original — sub-microsecond.

    **Limitation**: The heuristic does not track writes (e.g., ``to_csv``,
    ``np.save``).  Write-side dependencies are not needed for cache invalidation
    because the *output* of a statement is hashed directly, not the files it
    writes.
    """
    def __init__(self, user_ns=None, propagate_to_parent: bool = False):
        self.accessed_files = set()
        self.user_ns = user_ns or {}
        self.registry = FileDependencyRegistry()
        # Stack of ContextVar tokens, one per active __enter__. Supports
        # re-entry of the same instance (an async function that reuses
        # a tracker across awaits, or a sync caller using `with` twice).
        self._token_stack: list[contextvars.Token] = []
        # When True, a file read in this block also registers with the
        # enclosing tracker(s) - so an OUTER cached function records the files
        # its (cold) inner cached calls read, otherwise the outer entry would
        # never invalidate when that file changes (file deps don't propagate
        # through depends_on). The decorator sets this; manual `with
        # FileAccessTracker()` nesting stays isolated by default.
        self._propagate_to_parent = propagate_to_parent
        self._parent_stack: list[Optional["FileAccessTracker"]] = []

    def __enter__(self):
        # Install permanent dispatcher patches. Each (module/dict, name)
        # target is patched only once for the lifetime of the process —
        # ``_apply_patches`` self-skips targets whose current attribute
        # already carries the ``_is_file_tracker_patch`` sentinel. This
        # is cheap on the hot path (a single ``getattr`` per registered
        # target) and lets newly-imported modules and newly-registered
        # handlers be picked up on the next ``__enter__``.
        with _install_lock:
            self._apply_patches()
        _ensure_import_hook_installed()
        # Capture the enclosing tracker (if any) BEFORE we become active, so a
        # read inside this block also registers with the outer tracker(s).
        self._parent_stack.append(_active_tracker.get())
        self._token_stack.append(_active_tracker.set(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token_stack:
            _active_tracker.reset(self._token_stack.pop())
        if self._parent_stack:
            self._parent_stack.pop()

    def get_accessed_files(self) -> set[str]:
        return self.accessed_files

    def _track_path(self, path):
        try:
            # Normalize path using realpath to get canonical path
            # This resolves symlinks and normalizes the path, making it
            # stable across os.chdir() calls
            abs_path = normalize_path(os.path.realpath(str(path)))
        except (TypeError, ValueError, OSError) as e:
            logger.debug("[TRACKER] Could not track file path %r: %s", path, e)
            return
        self._add_tracked(abs_path)

    def _add_tracked(self, abs_path: str) -> None:
        """Record *abs_path* on this tracker and, when propagation is enabled,
        on the enclosing tracker(s) too - so nested cached reads count as the
        outer cached function's deps. Manual tracker nesting stays isolated."""
        self.accessed_files.add(abs_path)
        if not self._propagate_to_parent:
            return
        parent = self._parent_stack[-1] if self._parent_stack else None
        if parent is not None and parent is not self:
            parent._add_tracked(abs_path)

    def _apply_patches(self):
        # 1. Patch Builtins
        _install_module_patches('builtins', builtins)

        # 2. Patch User Namespace (for interactive sessions showing 'open')
        self._patch_user_ns()

        # 3. Patch Loaded Modules
        # Iterate over registered modules
        for mod_name in self.registry.handlers:
            if mod_name == 'builtins':
                continue

            if mod_name in sys.modules:
                module = sys.modules[mod_name]
                _install_module_patches(mod_name, module)

    def _patch_user_ns(self):
        """Patch open in user namespace (IPython specific). Self-heals
        by walking past any leaked wrappers to the real callable."""
        # Handle user_ns['open'] — skip if dispatcher already installed.
        if 'open' in self.user_ns and not getattr(
            self.user_ns['open'], '_is_file_tracker_patch', False
        ):
            real_open = _unwrap_to_real(self.user_ns['open'])
            factory = self.registry._create_open_handler
            wrapper = factory(real_open, _dispatch_track)
            wrapper._is_file_tracker_patch = True
            wrapper._original_func = real_open

            self.user_ns['open'] = wrapper

        # Handle user_ns['__builtins__']['open'] (if dict)
        if '__builtins__' in self.user_ns:
            bs = self.user_ns['__builtins__']
            if isinstance(bs, dict) and 'open' in bs and not getattr(
                bs['open'], '_is_file_tracker_patch', False
            ):
                real_open = _unwrap_to_real(bs['open'])
                factory = self.registry._create_open_handler
                wrapper = factory(real_open, _dispatch_track)
                wrapper._is_file_tracker_patch = True
                wrapper._original_func = real_open

                bs['open'] = wrapper

