from __future__ import annotations

"""File-read interception for tracking data file dependencies.

Monkey-patches common I/O functions (``open``, ``pandas.read_csv``,
``numpy.load``, etc.) to record which files each statement reads.
File hashes are incorporated into cache keys so that changed data
automatically invalidates dependent cached results.
"""

import builtins
import importlib
import importlib.abc
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from cash.utils import normalize_path

__all__ = ["FileDependencyRegistry", "PostImportHook", "FileAccessTracker", "FileDependencies"]

# Type alias for file dependency tracking: maps normalized file path -> mtime at read time
FileDependencies = dict[str, float]

logger = logging.getLogger(__name__)

def _unwrap_to_real(func: Any) -> Any:
    """Walk a chain of FileAccessTracker wrappers down to the original
    callable. Returns ``func`` unchanged if it isn't a wrapper.

    Used by :meth:`FileAccessTracker._patch_module` and ``_patch_user_ns``
    so a fresh tracker can self-heal past wrappers left behind by a
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
        """Handler for open()-like functions."""
        def tracked_open(file, *args, **kwargs):
            # Only track read modes or default mode
            mode = args[0] if args else kwargs.get('mode', 'r')
            if 'r' in mode or '+' in mode:
                track_callback(file)
            return original_func(file, *args, **kwargs)
        return tracked_open

    @staticmethod
    def _create_path_arg_handler(original_func: Callable[..., Any], track_callback: Callable[..., Any]):
        """Generic handler for functions where the first argument is a path."""
        def tracked_func(path_or_buf, *args, **kwargs):
            # Check if it looks like a file path
            if isinstance(path_or_buf, (str, bytes, os.PathLike)):
                track_callback(path_or_buf)
            return original_func(path_or_buf, *args, **kwargs)
        return tracked_func

class PostImportHook(importlib.abc.MetaPathFinder):
    """
    Intercepts imports of specific modules to patch them after loading.
    """
    def __init__(self, tracker: FileAccessTracker):
        self.tracker = tracker
        self._skip: set[str] = set()  # Avoid recursion

    def find_spec(self, fullname, path, target=None):
        if fullname in self._skip:
            return None

        # Only interest in registered modules
        # Note: We match top-level packages mainly.
        # e.g. 'pandas.io' -> we patch 'pandas' too?
        # The handlers are registered by module name.
        top_level = fullname.split('.')[0]

        targets = self.tracker.registry.handlers.keys()
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
        spec.loader = _PatchingLoader(spec.loader, self.tracker, fullname)
        return spec

class _PatchingLoader:
    def __init__(self, original_loader, tracker, fullname):
        self.original_loader = original_loader
        self.tracker = tracker
        self.fullname = fullname

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        # execute module
        self.original_loader.exec_module(module)

        # Now patch it!
        self.tracker._patch_module(self.fullname, module)

class FileAccessTracker:
    """Context manager that intercepts file I/O to record which files a
    statement reads.

    **Monkey-patching approach**: On ``__enter__``, patches ``builtins.open``,
    loaded pandas/polars/numpy/joblib/pickle/json I/O functions, and the user
    namespace binding of ``open``.  A ``PostImportHook`` is also installed on
    ``sys.meta_path`` to patch libraries imported *inside* the tracked block.

    **Cleanup guarantee**: ``__exit__`` always restores every patched function
    and removes the import hook — even if an exception is raised.  Each patch
    record in ``self.patches`` is a ``(obj, name, original_fn)`` tuple; all
    are reverted in :meth:`_unpatch`.  The ``sys.meta_path`` hook is removed
    first so that new imports during unpatch are not double-intercepted.

    **Limitation**: The heuristic does not track writes (e.g., ``to_csv``,
    ``np.save``).  Write-side dependencies are not needed for cache invalidation
    because the *output* of a statement is hashed directly, not the files it
    writes.
    """
    def __init__(self, user_ns=None):
        self.accessed_files = set()
        self.patches = []
        self.user_ns = user_ns or {}
        self.registry = FileDependencyRegistry()
        self.import_hook = PostImportHook(self)

    def __enter__(self):
        self._apply_patches()
        sys.meta_path.insert(0, self.import_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove import hook first so new imports during unpatch are not intercepted.
        if self.import_hook in sys.meta_path:
            sys.meta_path.remove(self.import_hook)
        self._unpatch()

    def get_accessed_files(self) -> set[str]:
        return self.accessed_files

    def _track_path(self, path):
        try:
            # Normalize path using realpath to get canonical path
            # This resolves symlinks and normalizes the path, making it
            # stable across os.chdir() calls
            abs_path = normalize_path(os.path.realpath(str(path)))
            self.accessed_files.add(abs_path)
            # logger.debug(f"[TRACKER] Tracked: {abs_path}")
        except (TypeError, ValueError, OSError) as e:
            logger.debug("[TRACKER] Could not track file path %r: %s", path, e)

    def _apply_patches(self):
        # 1. Patch Builtins
        self._patch_module('builtins', builtins)

        # 2. Patch User Namespace (for interactive sessions showing 'open')
        self._patch_user_ns()

        # 3. Patch Loaded Modules
        # Iterate over registered modules
        for mod_name in self.registry.handlers:
            if mod_name == 'builtins':
                continue

            if mod_name in sys.modules:
                module = sys.modules[mod_name]
                self._patch_module(mod_name, module)

    def _find_patch_targets(self, func_pattern: str, module_obj: Any) -> list:
        """Return the list of attribute names to patch on *module_obj*."""
        if func_pattern.endswith('*'):
            prefix = func_pattern[:-1]
            return [name for name in dir(module_obj) if name.startswith(prefix)]
        if hasattr(module_obj, func_pattern):
            return [func_pattern]
        return []

    def _patch_module(self, module_name: str, module_obj: Any):
        """Patch functions in a module based on registry."""
        handlers = self.registry.get_handlers_for_module(module_name)

        for func_pattern, factory in handlers:
            targets = self._find_patch_targets(func_pattern, module_obj)

            # Apply patches
            for name in targets:
                if not hasattr(module_obj, name):
                    continue

                original_func = getattr(module_obj, name)

                # Self-heal: if we see a wrapper from a prior tracker,
                # walk down its ``_original_func`` chain to the real,
                # non-wrapper callable. We wrap THAT instead and remember
                # it as the original for unpatching. Without this step a
                # silently-failed unpatch from a prior tracker (or just
                # a tracker that got orphaned by an exception) leaves
                # its wrapper installed forever, with subsequent
                # trackers refusing to re-patch and the wrapper pinning
                # the dead tracker in memory via its closure.
                real_original = _unwrap_to_real(original_func)

                if not callable(real_original):
                    continue

                # Create wrapper around the *real* original
                wrapper = factory(real_original, self._track_path)
                wrapper._is_file_tracker_patch = True
                wrapper._original_func = real_original  # For debugging/unwrapping

                # Apply patch
                try:
                    setattr(module_obj, name, wrapper)
                    self.patches.append((module_obj, name, real_original))
                except (AttributeError, TypeError) as e:
                    logger.debug("[FILE_TRACKER] Failed to patch %s.%s: %s", module_name, name, e)

    def _patch_user_ns(self):
        """Patch open in user namespace (IPython specific). Like
        ``_patch_module``, this self-heals by walking past any leaked
        wrappers to the real callable."""
        # Handle user_ns['open']
        if 'open' in self.user_ns:
            real_open = _unwrap_to_real(self.user_ns['open'])
            factory = self.registry._create_open_handler
            wrapper = factory(real_open, self._track_path)
            wrapper._is_file_tracker_patch = True
            wrapper._original_func = real_open

            self.user_ns['open'] = wrapper
            self.patches.append((self.user_ns, 'open', real_open))

        # Handle user_ns['__builtins__']['open'] (if dict)
        if '__builtins__' in self.user_ns:
            bs = self.user_ns['__builtins__']
            if isinstance(bs, dict) and 'open' in bs:
                real_open = _unwrap_to_real(bs['open'])
                factory = self.registry._create_open_handler
                wrapper = factory(real_open, self._track_path)
                wrapper._is_file_tracker_patch = True
                wrapper._original_func = real_open

                bs['open'] = wrapper
                self.patches.append((bs, 'open', real_open))

    def _unpatch(self):
        """Revert all patches."""
        for obj, name, original in reversed(self.patches):
            try:
                if isinstance(obj, dict):
                    obj[name] = original
                else:
                    setattr(obj, name, original)
            except (AttributeError, TypeError, KeyError) as e:
                logger.debug("[FILE_TRACKER] Failed to unpatch: %s", e)
        self.patches = []

