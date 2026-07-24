from __future__ import annotations

"""Track function source code changes for cache invalidation."""

import ast
import contextlib
import hashlib
import inspect
import logging
import os
import site
import sys
import sysconfig
import textwrap
import types
from typing import Any

from ..exceptions import SOURCE_RETRIEVAL_ERRORS

__all__ = ["FunctionTracker", "is_local_module"]

logger = logging.getLogger(__name__)

# Lazily computed set of stdlib/site-packages directory prefixes
_STDLIB_SITE_PREFIXES: set[str] | None = None

def _get_stdlib_site_prefixes() -> set[str]:
    """Get the set of directory prefixes for stdlib and site-packages.

    These prefixes are used to distinguish local modules from installed packages.
    Computed once and cached for the process lifetime.
    """
    global _STDLIB_SITE_PREFIXES
    if _STDLIB_SITE_PREFIXES is not None:
        return _STDLIB_SITE_PREFIXES

    prefixes: set[str] = set()

    # stdlib paths
    stdlib_path = sysconfig.get_path('stdlib')
    if stdlib_path:
        prefixes.add(os.path.normcase(os.path.realpath(stdlib_path)))
    platstdlib = sysconfig.get_path('platstdlib')
    if platstdlib:
        prefixes.add(os.path.normcase(os.path.realpath(platstdlib)))

    # site-packages paths
    for sp in site.getsitepackages() if hasattr(site, 'getsitepackages') else []:
        prefixes.add(os.path.normcase(os.path.realpath(sp)))
    user_site = site.getusersitepackages() if hasattr(site, 'getusersitepackages') else None
    if user_site:
        prefixes.add(os.path.normcase(os.path.realpath(user_site)))

    # Also add the Python prefix itself (covers DLLs, Lib, etc.)
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if p:
            norm = os.path.normcase(os.path.realpath(p))
            prefixes.add(norm)

    _STDLIB_SITE_PREFIXES = prefixes
    return prefixes


def _reset_stdlib_site_prefixes() -> None:
    """Reset the cached stdlib/site-packages prefixes (useful for testing)."""
    global _STDLIB_SITE_PREFIXES
    _STDLIB_SITE_PREFIXES = None


def is_local_module(module: types.ModuleType) -> bool:
    """Check if a module is local (not stdlib, not site-packages).

    A module is considered "local" if its source file is not under any of
    the stdlib, site-packages, or Python installation directories.

    Args:
        module: The module to check

    Returns:
        True if the module's source file is outside stdlib/site-packages
    """
    file_path = getattr(module, '__file__', None)
    if not file_path:
        return False

    # Must be a .py file (not .pyd, .so, .pyc without source)
    if not file_path.endswith('.py'):
        # Check if there's a corresponding .py for .pyc
        if file_path.endswith(('.pyc', '.pyo')):
            py_path = file_path[:-1]  # .pyc -> .py
            if not os.path.isfile(py_path):
                return False
        else:
            return False

    real_path = os.path.normcase(os.path.realpath(file_path))
    prefixes = _get_stdlib_site_prefixes()

    return all(not real_path.startswith(prefix) for prefix in prefixes)

def _collect_imported_names(tree: ast.AST) -> set[str]:
    """Return all module names (top-level and dotted) referenced by import statements in *tree*."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
            names.add(node.module)
    return names


def _evict_source_cache(cache: dict, max_size: int) -> None:
    """Evict oldest 25% of entries from the source cache if it is at capacity."""
    if len(cache) >= max_size:
        evict_count = max_size // 4
        keys_to_evict = list(cache.keys())[:evict_count]
        for k in keys_to_evict:
            del cache[k]


def _update_code_object_hash(h: Any, code_obj: Any) -> None:
    """Feed *code_obj* into hash *h*, recursing into nested code objects.

    ``str(co_consts)`` is NOT usable for consts holding nested code objects
    (a factory's inner ``def``): their repr embeds the compile-time memory
    address, so two compiles of IDENTICAL source hashed differently and the
    cache key of every consumer drifted run-to-run (fallout). Recurse
    into nested code objects structurally instead.
    """
    h.update(code_obj.co_code)
    h.update(str(code_obj.co_names).encode('utf-8'))
    h.update(str(code_obj.co_varnames).encode('utf-8'))
    for const in code_obj.co_consts:
        if isinstance(const, types.CodeType):
            _update_code_object_hash(h, const)
        else:
            h.update(repr(const).encode('utf-8'))


def _compute_bytecode_hash(func: Any) -> str | None:
    """Compute a hash from a function's bytecode when source is unavailable."""
    code_obj = getattr(func, '__code__', None)
    if code_obj is None and hasattr(func, '__wrapped__'):
        code_obj = getattr(func.__wrapped__, '__code__', None)
    if code_obj is None:
        return None
    try:
        h = hashlib.sha256()
        _update_code_object_hash(h, code_obj)
        return h.hexdigest()
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("[FUNC_TRACKER] Failed to compute bytecode hash for function: %s", exc)
        return None


class FunctionTracker:
    """Tracks function source code for cache key computation.

    Maintains a cache of function source hashes to avoid repeated
    inspect.getsource() calls (which can be expensive).
    """

    # Max entries in the source hash cache
    MAX_CACHE_SIZE = 500

    def __init__(self):
        # Maps (func_id, func_qualname) -> (source_hash, source_code)
        self._source_cache: dict[tuple[int, str], tuple[str, str]] = {}
        # Maps function_name -> source_hash (for tracking changes)
        self._function_hashes: dict[str, str] = {}
        # Module file tracking: module_name -> last known mtime
        self._module_mtimes: dict[str, float] = {}
        # Set of module names user explicitly asked to track
        self._tracked_modules: set[str] = set()
        # Transitive dependency tracking:
        # Maps sub-dependency file path -> set of tracked (parent) module names that depend on it
        self._dep_file_to_parents: dict[str, set[str]] = {}
        # Maps sub-dependency file path -> last known mtime
        self._dep_file_mtimes: dict[str, float] = {}
        # Per-symbol hash tracking for granular invalidation:
        # Maps module_name -> {symbol_name: hash_of_symbol_source}
        self._module_symbol_hashes: dict[str, dict[str, str]] = {}

    def _tracked_module_file_changed(self, func_module: str | None) -> bool:
        """Return True if *func_module* is tracked and its file mtime has changed."""
        if not func_module or func_module not in self._tracked_modules:
            return False
        module_obj = sys.modules.get(func_module)
        if not module_obj:
            return False
        file_path = getattr(module_obj, '__file__', None)
        if not file_path or not os.path.isfile(file_path):
            return False
        try:
            current_mtime = os.path.getmtime(file_path)
            stored_mtime = self._module_mtimes.get(func_module)
            return stored_mtime is not None and current_mtime != stored_mtime
        except OSError:
            return False

    def get_function_source_hash(self, func: Any) -> str | None:
        """Get the source hash for a callable.

        Returns None if:
        - The object is not a callable
        - The callable is a built-in or C extension
        - The source cannot be retrieved

        For functions from tracked modules, bypasses the id-based cache to
        always read fresh source from disk — this ensures that module file
        changes are reflected even if the function object hasn't been replaced.

        Args:
            func: The callable to hash

        Returns:
            SHA256 hash of the function's source code, or None
        """
        if not callable(func):
            return None

        # Skip built-in functions, C extensions, and special objects
        if isinstance(func, types.BuiltinFunctionType):
            return None
        if isinstance(func, type) and func.__module__ == 'builtins':
            return None

        # Skip lambda (anonymous, can't get reliable source)
        if hasattr(func, '__name__') and func.__name__ == '<lambda>':
            return None

        # Check if this function is from a tracked module — if so, bypass id-cache
        # to ensure we always read fresh source from disk
        func_module = getattr(func, '__module__', None)
        use_cache = not self._tracked_module_file_changed(func_module)

        # Check cache using id + qualname (id alone isn't enough since objects can be recycled)
        cache_key = (id(func), getattr(func, '__qualname__', ''))
        if use_cache and cache_key in self._source_cache:
            return self._source_cache[cache_key][0]

        # Try to get source
        try:
            source = inspect.getsource(func)
            # Dedent to normalize indentation
            source = textwrap.dedent(source)
            source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()

            _evict_source_cache(self._source_cache, self.MAX_CACHE_SIZE)
            self._source_cache[cache_key] = (source_hash, source)
            return source_hash

        except SOURCE_RETRIEVAL_ERRORS:
            # Can't get source - try bytecode fallback (works for IPython-defined
            # functions when %cash_on intercepts execution)
            source_hash = _compute_bytecode_hash(func)
            if source_hash is not None:
                _evict_source_cache(self._source_cache, self.MAX_CACHE_SIZE)
                self._source_cache[cache_key] = (source_hash, '<bytecode>')
                return source_hash

    def get_callable_source_hashes(
        self,
        input_names: set[str],
        user_ns: dict[str, Any]
    ) -> dict[str, str]:
        """Get source hashes for all callable inputs.

        Given a set of input variable names, finds which ones are user-defined
        callables and returns their source hashes.

        Args:
            input_names: Set of variable names used as inputs
            user_ns: The user namespace (IPython shell.user_ns)

        Returns:
            Dict mapping variable name -> source hash (only for trackable callables)
        """
        result = {}
        for name in input_names:
            if name not in user_ns:
                continue
            val = user_ns[name]
            if not callable(val):
                continue

            source_hash = self.get_function_source_hash(val)
            if source_hash is not None:
                result[name] = source_hash

        return result

    def detect_changed_functions(
        self,
        user_ns: dict[str, Any],
        tracked_names: set[str] | None = None
    ) -> set[str]:
        """Detect functions whose source code has changed since last check.

        Args:
            user_ns: The user namespace
            tracked_names: Optional set of function names to check.
                          If None, checks all known functions.

        Returns:
            Set of function names that have changed
        """
        changed = set()
        names_to_check = tracked_names or set(self._function_hashes.keys())

        for name in names_to_check:
            if name not in user_ns:
                # Function was deleted
                if name in self._function_hashes:
                    changed.add(name)
                continue

            val = user_ns[name]
            if not callable(val):
                if name in self._function_hashes:
                    # Was a function, now isn't
                    changed.add(name)
                continue

            current_hash = self.get_function_source_hash(val)
            if current_hash is None:
                continue

            old_hash = self._function_hashes.get(name)
            if old_hash is not None and old_hash != current_hash:
                changed.add(name)

            # Update stored hash
            self._function_hashes[name] = current_hash

        return changed

    def update_function_hash(self, name: str, func: Any) -> str | None:
        """Update the stored hash for a function.

        Args:
            name: The function name in user namespace
            func: The function object

        Returns:
            The new source hash, or None if not trackable
        """
        source_hash = self.get_function_source_hash(func)
        if source_hash is not None:
            self._function_hashes[name] = source_hash
        return source_hash

    def get_called_function_names(self, code: str) -> set[str]:
        """Extract names of functions called in a code block.

        Uses AST parsing to find all function calls.

        Args:
            code: Source code to analyze

        Returns:
            Set of function names called in the code
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()

        called = set()

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # For method calls like obj.method(), we track the base object
                    base = node.func
                    while isinstance(base, ast.Attribute):
                        base = base.value
                    if isinstance(base, ast.Name):
                        called.add(base.id)
                self.generic_visit(node)

        CallVisitor().visit(tree)
        return called

    def clear(self):
        self._source_cache.clear()
        self._function_hashes.clear()
        self._module_mtimes.clear()
        self._tracked_modules.clear()
        self._dep_file_to_parents.clear()
        self._dep_file_mtimes.clear()
        self._module_symbol_hashes.clear()

    # ================================================================
    # Module file tracking for imported functions
    # ================================================================

    def track_module(self, module_name: str) -> str | None:
        """Start tracking a module's file for changes.

        Also discovers and tracks transitive local dependencies — if the
        module imports other local modules, those files are monitored too.
        When a sub-dependency changes, the parent module is reported as changed.

        Args:
            module_name: The module name (e.g., 'my_helpers')

        Returns:
            The file path of the module, or None if not found
        """
        import sys
        self._tracked_modules.add(module_name)

        module = sys.modules.get(module_name)
        if module is None:
            return None

        file_path = getattr(module, '__file__', None)
        if file_path and os.path.isfile(file_path):
            try:
                mtime = os.path.getmtime(file_path)
                self._module_mtimes[module_name] = mtime
                logger.debug("Tracking module '%s' at %s, mtime=%s", module_name, file_path, mtime)
                # Snapshot per-symbol hashes for granular invalidation
                self.snapshot_module_symbols(module_name)
                # Discover and track transitive local dependencies
                self._discover_transitive_dependencies(module_name)
                return file_path
            except OSError:
                logger.debug("Failed to read module file for tracking '%s'", module_name)
        return None

    def _process_sub_modules(self, sub_module_names: set[str], module_name: str, visited: set[str], stack: list[str]) -> None:
        """Walk *sub_module_names*, recording local ones as deps of *module_name*."""
        for sub_name in sub_module_names:
            sub_mod = sys.modules.get(sub_name)
            if sub_mod is None:
                continue
            if not is_local_module(sub_mod):
                continue
            sub_file = getattr(sub_mod, '__file__', None)
            if not sub_file or not os.path.isfile(sub_file):
                continue
            self._register_sub_dependency(sub_name, sub_file, module_name, visited, stack)

    def _register_sub_dependency(self, sub_name: str, sub_file: str, module_name: str, visited: set[str], stack: list[str]) -> None:
        """Record a sub-dependency file and schedule it for further walking."""
        norm_path = os.path.normcase(os.path.realpath(sub_file))

        if norm_path not in self._dep_file_to_parents:
            self._dep_file_to_parents[norm_path] = set()
        self._dep_file_to_parents[norm_path].add(module_name)

        if norm_path not in self._dep_file_mtimes:
            with contextlib.suppress(OSError):
                self._dep_file_mtimes[norm_path] = os.path.getmtime(sub_file)

        if sub_name not in self._tracked_modules:
            self._tracked_modules.add(sub_name)
            with contextlib.suppress(OSError):
                self._module_mtimes[sub_name] = os.path.getmtime(sub_file)

        if sub_name not in visited:
            stack.append(sub_name)

    def _discover_transitive_dependencies(self, module_name: str) -> None:
        """Walk the import graph of a tracked module to find local sub-dependencies.

        For each local module that ``module_name`` (transitively) imports, we
        record:
        * ``_dep_file_to_parents[dep_file_path]`` → set of parent module names
        * ``_dep_file_mtimes[dep_file_path]`` → current mtime

        This way, ``check_tracked_modules`` can detect changes in sub-dependency
        files and map them back to the parent modules that the notebook imported.
        """
        module = sys.modules.get(module_name)
        if module is None:
            return

        visited: set[str] = set()
        stack = [module_name]

        while stack:
            mod_name = stack.pop()
            if mod_name in visited:
                continue
            visited.add(mod_name)

            mod = sys.modules.get(mod_name)
            if mod is None:
                continue

            mod_file = getattr(mod, '__file__', None)
            if not mod_file or not os.path.isfile(mod_file):
                continue

            try:
                with open(mod_file, encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source, filename=mod_file)
            except (SyntaxError, OSError, UnicodeDecodeError):
                logger.debug("Could not parse transitive dependency '%s'", mod_name)
                continue

            sub_module_names = _collect_imported_names(tree)
            self._process_sub_modules(sub_module_names, module_name, visited, stack)

    def refresh_transitive_dependencies(self) -> None:
        """Re-scan transitive dependencies for all currently tracked modules.

        Call this after reloading a module so that newly-added imports
        (e.g. a module that now imports an additional helper) are tracked.
        """
        # Clear existing dependency data and rebuild
        self._dep_file_to_parents.clear()
        self._dep_file_mtimes.clear()
        for mod_name in list(self._tracked_modules):
            # Only rebuild for top-level tracked modules (those the notebook imported)
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            mod_file = getattr(mod, '__file__', None)
            if mod_file and os.path.isfile(mod_file):
                self._discover_transitive_dependencies(mod_name)

    # ================================================================
    # Per-symbol hashing for granular invalidation
    # ================================================================

    @staticmethod
    def _names_from_assign_targets(targets: list[ast.expr]) -> list[str]:
        """Collect simple name strings from assignment LHS targets."""
        names: list[str] = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        return names

    @staticmethod
    def _symbol_names_and_hash_source(node: ast.AST) -> tuple[list[str], str | None]:
        """Return (names, hash_source) for a top-level AST node, or ([], None) if not a symbol."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [node.name], ast.dump(node)
        if isinstance(node, ast.ClassDef):
            return [node.name], ast.dump(node)
        if isinstance(node, ast.Assign):
            return FunctionTracker._names_from_assign_targets(node.targets), ast.dump(node.value)
        if isinstance(node, ast.AnnAssign):
            name = [node.target.id] if isinstance(node.target, ast.Name) else []
            hash_src = ast.dump(node.value) if node.value is not None else "annotation_only"
            return name, hash_src
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_names = [
                f"__import__{alias.asname or alias.name}" for alias in node.names
            ]
            return import_names, ast.dump(node)
        return [], None

    @staticmethod
    def compute_symbol_hashes(file_path: str) -> dict[str, str]:
        """Parse a Python file and compute a hash for each top-level symbol.

        Top-level symbols include:
        - Function definitions (def / async def)
        - Class definitions
        - Simple assignments (constants like ``VERSION = "1.0"``)
        - ``__all__`` exports

        The hash is computed from the AST-unparsed source of each symbol's
        AST node, so whitespace/comment-only changes that don't affect the
        AST are ignored.  For assignments where the target is a simple name,
        the hash is based on the ``ast.dump`` of the value node.

        Returns:
            dict mapping symbol name → SHA256 hex digest
        """
        if not file_path or not os.path.isfile(file_path):
            return {}

        try:
            with open(file_path, encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
        except (SyntaxError, OSError, UnicodeDecodeError):
            return {}

        hashes: dict[str, str] = {}

        for node in ast.iter_child_nodes(tree):
            names, hash_source = FunctionTracker._symbol_names_and_hash_source(node)
            if names and hash_source:
                h = hashlib.sha256(hash_source.encode('utf-8')).hexdigest()
                for name in names:
                    hashes[name] = h

        return hashes

    def snapshot_module_symbols(self, module_name: str) -> dict[str, str]:
        """Compute and store per-symbol hashes for a tracked module.

        Should be called after a module is first tracked or after it is
        reloaded so that the next change can be compared granularly.

        Args:
            module_name: Name of the tracked module

        Returns:
            The symbol hash map (also stored internally)
        """
        module = sys.modules.get(module_name)
        if module is None:
            return {}
        file_path = getattr(module, '__file__', None)
        hashes = self.compute_symbol_hashes(file_path)
        self._module_symbol_hashes[module_name] = hashes
        return hashes

    def get_changed_symbols(self, module_name: str) -> set[str] | None:
        """Compare stored symbol hashes with current file to find changed symbols.

        Returns:
            Set of symbol names whose hash changed, were added, or were removed.
            Returns ``None`` if no previous snapshot exists (caller should fall
            back to full invalidation).
        """
        old_hashes = self._module_symbol_hashes.get(module_name)
        if old_hashes is None:
            return None  # No baseline — can't do granular

        module = sys.modules.get(module_name)
        if module is None:
            return None
        file_path = getattr(module, '__file__', None)
        new_hashes = self.compute_symbol_hashes(file_path)

        changed: set[str] = set()

        # Symbols that changed or were added
        for sym, h in new_hashes.items():
            if old_hashes.get(sym) != h:
                changed.add(sym)

        # Symbols that were removed
        for sym in old_hashes:
            if sym not in new_hashes:
                changed.add(sym)

        return changed

    @staticmethod
    def _collect_top_level_names(tree: ast.AST) -> set[str]:
        """Return all top-level symbol names defined in *tree*."""
        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        return names

    @staticmethod
    def _collect_intra_refs(tree: ast.AST, top_level_names: set[str]) -> dict[str, set[str]]:
        """Map each top-level function/class to the top-level names it references."""
        deps: dict[str, set[str]] = {}
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            sym_name = node.name
            referenced: set[str] = set()
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in top_level_names
                    and child.id != sym_name
                ):
                    referenced.add(child.id)
            if referenced:
                deps[sym_name] = referenced
        return deps

    @staticmethod
    def get_intra_module_call_deps(file_path: str) -> dict[str, set[str]]:
        """Analyze a module file to find which top-level functions call which other
        top-level functions/symbols within the same module.

        This enables **transitive granular invalidation**: if ``dep()`` changed
        and ``fun()`` calls ``dep()``, then ``fun`` must also be considered
        changed even though its own source code didn't change.

        Args:
            file_path: Path to the module's Python source file.

        Returns:
            Dict mapping symbol_name → set of other top-level symbol names
            that it references (calls, reads, etc.).  Only includes references
            to names that are themselves top-level definitions in the module.
        """
        if not file_path or not os.path.isfile(file_path):
            return {}

        try:
            with open(file_path, encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
        except (SyntaxError, OSError, UnicodeDecodeError):
            return {}

        top_level_names = FunctionTracker._collect_top_level_names(tree)
        return FunctionTracker._collect_intra_refs(tree, top_level_names)

    @staticmethod
    def expand_changed_symbols_transitively(
        changed_syms: set[str],
        call_deps: dict[str, set[str]],
    ) -> set[str]:
        """Expand a set of directly changed symbols to include all symbols that
        transitively depend on them (reverse transitive closure).

        If ``dep`` changed and ``fun`` calls ``dep``, then ``fun`` is also
        effectively changed because its behavior depends on ``dep``.

        Args:
            changed_syms: Set of directly changed symbol names.
            call_deps: Dict from ``get_intra_module_call_deps`` mapping
                       symbol → set of symbols it references.

        Returns:
            Expanded set including original changed_syms plus all symbols
            that transitively depend on any changed symbol.
        """
        # Build reverse dependency map: for each symbol, who depends on it?
        reverse_deps: dict[str, set[str]] = {}
        for caller, callees in call_deps.items():
            for callee in callees:
                if callee not in reverse_deps:
                    reverse_deps[callee] = set()
                reverse_deps[callee].add(caller)

        # BFS/DFS from changed_syms through reverse_deps
        expanded = set(changed_syms)
        queue = list(changed_syms)
        while queue:
            sym = queue.pop()
            for dependent in reverse_deps.get(sym, ()):
                if dependent not in expanded:
                    expanded.add(dependent)
                    queue.append(dependent)

        return expanded

    @staticmethod
    def _handle_getattr_call(node: ast.Call, accesses: dict[str, set[str]], bare_uses: set[str]) -> None:
        """Handle getattr(mod, 'attr') patterns, updating accesses/bare_uses in-place."""
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == 'getattr'
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            return
        base = node.args[0].id
        if (
            len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            accesses.setdefault(base, set()).add(node.args[1].value)
        else:
            accesses[base] = set()
            bare_uses.add(base)

    @staticmethod
    def extract_module_attribute_accesses(code: str) -> dict[str, set[str]]:
        """Statically determine which attributes of each name are accessed in code.

        Parses the code AST and looks for patterns like ``mod.func()``,
        ``mod.CONST``, ``mod.Class(...)``, ``from mod import func``.

        Returns:
            dict mapping base name → set of attribute names accessed.
            An empty set for a name means the module itself is used but we
            cannot determine which specific attributes (e.g. passed as argument,
            used in getattr, etc.) — caller should treat as "all attributes".
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        accesses: dict[str, set[str]] = {}
        bare_uses: set[str] = set()

        class AttrVisitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                # mod.attr  or  mod.attr.subattr  — we only care about the
                # first level: base_name.attr
                if isinstance(node.value, ast.Name):
                    base = node.value.id
                    if base not in accesses:
                        accesses[base] = set()
                    accesses[base].add(node.attr)
                self.generic_visit(node)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    # Bare name usage — might be module passed to a function,
                    # used in getattr, etc.
                    bare_uses.add(node.id)
                self.generic_visit(node)

            def visit_Call(self, node):
                FunctionTracker._handle_getattr_call(node, accesses, bare_uses)
                self.generic_visit(node)

        AttrVisitor().visit(tree)

        # For names that appear as bare references (not just as mod.attr base)
        # but aren't in accesses from attribute access, add them with empty set
        # to signal "unknown attributes used".
        # However, if a name IS in accesses and also used bare, the bare usage
        # means we can't be sure only the listed attrs are used.
        for name in bare_uses:
            if name in accesses:
                # Name is used both as mod.attr AND bare — check if it's
                # only used bare as the base of attribute accesses (which is fine)
                # vs. passed to a function or used standalone.
                pass  # We keep the tracked attrs — they're the specific ones we found

        return accesses

    def compute_module_symbol_hash(self, module_name: str, accessed_attrs: set[str] | None) -> str:
        """Compute a hash for a module based on only the accessed symbols.

        If ``accessed_attrs`` is None or empty (meaning we can't determine
        which attributes are used), falls back to hashing the entire module.

        Args:
            module_name: The tracked module name
            accessed_attrs: Set of attribute names accessed, or None/empty for full hash

        Returns:
            Hash string
        """
        symbol_hashes = self._module_symbol_hashes.get(module_name)

        if not accessed_attrs or not symbol_hashes:
            # Fall back to full file hash
            module = sys.modules.get(module_name)
            if module is None:
                return hashlib.sha256(b'unknown').hexdigest()
            file_path = getattr(module, '__file__', None)
            if not file_path or not os.path.isfile(file_path):
                return hashlib.sha256(b'unknown').hexdigest()
            try:
                with open(file_path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
            except OSError:
                return hashlib.sha256(b'unknown').hexdigest()

        # Compute hash from only the accessed symbols (sorted for determinism)
        hasher = hashlib.sha256()
        for attr in sorted(accessed_attrs):
            sym_hash = symbol_hashes.get(attr, '')
            hasher.update(f"{attr}:{sym_hash}".encode())
            # Also include any __import__ symbols that might define this attr
            # (e.g., `from helper import func` makes `func` available)
            import_key = f"__import__{attr}"
            if import_key in symbol_hashes:
                hasher.update(f"{import_key}:{symbol_hashes[import_key]}".encode())
        return hasher.hexdigest()

    def _check_direct_modules(self) -> set[str]:
        """Check each directly tracked module file; return names of changed modules."""
        changed: set[str] = set()
        for module_name in self._tracked_modules:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            file_path = getattr(module, '__file__', None)
            if not file_path or not os.path.isfile(file_path):
                continue
            try:
                current_mtime = os.path.getmtime(file_path)
            except OSError:
                continue
            old_mtime = self._module_mtimes.get(module_name)
            logger.debug(
                "[MODULE_CHECK] %s: current_mtime=%s, old_mtime=%s, changed=%s",
                module_name, current_mtime, old_mtime,
                old_mtime is not None and current_mtime != old_mtime,
            )
            if old_mtime is not None and current_mtime != old_mtime:
                changed.add(module_name)
                self._module_mtimes[module_name] = current_mtime
                logger.info("Module '%s' changed on disk", module_name)
            elif old_mtime is None:
                self._module_mtimes[module_name] = current_mtime
        return changed

    def _check_dep_files(self) -> set[str]:
        """Check transitive dep files; return parent module names that are affected."""
        changed: set[str] = set()
        for dep_path, parent_modules in list(self._dep_file_to_parents.items()):
            if not os.path.isfile(dep_path):
                continue
            try:
                current_mtime = os.path.getmtime(dep_path)
            except OSError:
                continue
            old_mtime = self._dep_file_mtimes.get(dep_path)
            if old_mtime is not None and current_mtime != old_mtime:
                for parent_mod in parent_modules:
                    changed.add(parent_mod)
                    logger.info(
                        "Sub-dependency '%s' changed → invalidating parent module '%s'",
                        dep_path, parent_mod,
                    )
                self._dep_file_mtimes[dep_path] = current_mtime
            elif old_mtime is None:
                self._dep_file_mtimes[dep_path] = current_mtime
        return changed

    def check_tracked_modules(self) -> set[str]:
        """Check if any tracked modules or their transitive dependencies have changed.

        Checks both direct module files and sub-dependency files discovered via
        ``_discover_transitive_dependencies``.  When a sub-dependency file changes,
        every parent module that depends on it is reported as changed.

        Returns:
            Set of module names whose files (or sub-dependency files) have been modified
        """
        changed = self._check_direct_modules()
        changed |= self._check_dep_files()
        return changed

    def reload_module(self, module_name: str) -> bool:
        """Reload a tracked module and update its mtime.

        Args:
            module_name: The module to reload

        Returns:
            True if reload succeeded, False otherwise
        """
        import importlib
        import sys

        module = sys.modules.get(module_name)
        if module is None:
            return False

        try:
            # Invalidate import caches to ensure fresh source is read
            importlib.invalidate_caches()

            # Remove compiled .pyc file if it exists
            file_path = getattr(module, '__file__', None)
            if file_path:
                cache_file = importlib.util.cache_from_source(file_path)
                if os.path.isfile(cache_file):
                    with contextlib.suppress(OSError):  # Stale .pyc removal is best-effort
                        os.remove(cache_file)

            importlib.reload(module)
            if file_path and os.path.isfile(file_path):
                self._module_mtimes[module_name] = os.path.getmtime(file_path)
            # Clear source cache for functions from this module
            self._invalidate_module_functions(module_name)
            logger.info("Reloaded module '%s'", module_name)
            return True
        except (ImportError, ModuleNotFoundError, OSError, AttributeError) as e:
            logger.warning("Failed to reload module '%s': %s", module_name, e)
            return False

    def _invalidate_module_functions(self, module_name: str):
        """Clear cached source hashes for functions from a specific module."""
        to_remove = []
        for (func_id, qualname), _ in self._source_cache.items():
            if module_name in qualname:
                to_remove.append((func_id, qualname))
        for key in to_remove:
            del self._source_cache[key]

        # Also clear function_hashes for functions from this module
        module = sys.modules.get(module_name)
        if module:
            for name in list(self._function_hashes.keys()):
                obj = getattr(module, name, None)
                if obj is not None and callable(obj):
                    del self._function_hashes[name]

    # ================================================================
    # Auto-tracking of local module imports
    # ================================================================

    def auto_track_local_imports(self, code: str) -> set[str]:
        """Scan code for import statements and auto-track local modules.

        Parses the code for `import X` and `from X import Y` statements,
        checks if X is a local (non-stdlib, non-site-packages) module,
        and starts tracking it automatically.

        Args:
            code: Source code of a notebook cell

        Returns:
            Set of module names that were newly tracked
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()

        module_names = _collect_imported_names(tree)

        newly_tracked = set()
        for mod_name in module_names:
            if mod_name in self._tracked_modules:
                continue  # Already tracked

            module = sys.modules.get(mod_name)
            if module is None:
                continue  # Not yet imported

            if is_local_module(module):
                file_path = self.track_module(mod_name)
                if file_path:
                    newly_tracked.add(mod_name)
                    logger.debug("[AUTO_TRACK] Auto-tracking local module '%s' (%s)", mod_name, file_path)

        return newly_tracked

    def _find_directly_changed(self, changed_modules: set[str], pre_check_mtimes: dict[str, float]) -> set[str]:
        """Return the subset of changed_modules whose own file mtime actually changed."""
        directly_changed: set[str] = set()
        for mod_name in changed_modules:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            mod_file = getattr(mod, '__file__', None)
            if not mod_file or not os.path.isfile(mod_file):
                continue
            old_mtime = pre_check_mtimes.get(mod_name)
            if old_mtime is None:
                continue
            try:
                if os.path.getmtime(mod_file) != old_mtime:
                    directly_changed.add(mod_name)
            except OSError:
                pass  # Expected: file may be inaccessible or deleted
        return directly_changed

    def _build_changed_symbols_map(
        self, changed_modules: set[str], directly_changed: set[str]
    ) -> dict[str, set[str] | None]:
        """Build a map of module → changed symbol names (None = full invalidation)."""
        per_module: dict[str, set[str] | None] = {}
        for mod_name in changed_modules:
            if mod_name in directly_changed:
                changed_syms = self.get_changed_symbols(mod_name)
                per_module[mod_name] = changed_syms
                if changed_syms is not None:
                    logger.debug("[GRANULAR] Module '%s' changed symbols: %s", mod_name, changed_syms)
                else:
                    logger.debug("[GRANULAR] Module '%s': no symbol baseline, full invalidation", mod_name)
            else:
                per_module[mod_name] = None
                logger.debug("[GRANULAR] Module '%s': transitive dep change, full invalidation", mod_name)
        return per_module

    def check_and_reload_changed_modules(self, user_ns: dict[str, Any]) -> tuple[dict[str, str], dict[str, set[str] | None]]:
        """Check tracked modules for changes, reload if needed, and update user_ns.

        This is the main entry point for automatic import invalidation.
        Called before each cell execution to detect and handle module changes.

        Handles transitive dependencies: if a sub-dependency file changed, **all**
        modules in the dependency chain are reloaded in bottom-up order so that
        ``from X import Y`` bindings pick up the fresh code.

        Args:
            user_ns: The user namespace (IPython shell.user_ns) — updated in-place
                with fresh function objects from reloaded modules

        Returns:
            Tuple of:
            - Dict mapping changed module names to their file paths
            - Dict mapping changed module names to the set of changed symbol names
              (None means full invalidation — no granular info available)
        """
        # Save module mtimes BEFORE checking for changes, because
        # check_tracked_modules() updates _module_mtimes in-place.
        pre_check_mtimes: dict[str, float] = dict(self._module_mtimes)

        changed_modules = self.check_tracked_modules()
        if not changed_modules:
            return {}, {}

        # Determine which modules changed directly (their own file changed)
        # vs. transitively (a sub-dependency changed but the module's own file didn't).
        directly_changed = self._find_directly_changed(changed_modules, pre_check_mtimes)

        # Before reloading, detect which specific symbols changed (granular invalidation).
        # We must do this BEFORE reload because reload overwrites the module source.
        per_module_changed_symbols = self._build_changed_symbols_map(changed_modules, directly_changed)

        # Compute the full set of modules that need reloading (including
        # intermediate modules in the dependency chain) and sort bottom-up.
        modules_to_reload = self._compute_reload_set(changed_modules)
        reload_order = self._topological_reload_order(modules_to_reload)

        result = {}
        for mod_name in reload_order:
            module = sys.modules.get(mod_name)
            file_path = getattr(module, '__file__', 'unknown') if module else 'unknown'

            if self.reload_module(mod_name):
                result[mod_name] = file_path
                reloaded_module = sys.modules.get(mod_name)
                if reloaded_module:
                    self._update_user_ns_from_module(mod_name, reloaded_module, user_ns)
                # Snapshot new symbol hashes after reload for next comparison
                self.snapshot_module_symbols(mod_name)
                # For modules that were reloaded as transitive deps but not
                # directly changed, record them with None (full invalidation)
                if mod_name not in per_module_changed_symbols:
                    per_module_changed_symbols[mod_name] = None

        # After reloading, refresh the transitive dependency graph
        # (the reloaded module may now import different sub-modules)
        self.refresh_transitive_dependencies()

        return result, per_module_changed_symbols

    def _build_tracked_imports_map(self) -> dict[str, set[str]]:
        """Return a map of tracked_module → set of other tracked modules it imports."""
        imports_map: dict[str, set[str]] = {}
        for mod_name in self._tracked_modules:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            mod_file = getattr(mod, '__file__', None)
            if not mod_file or not os.path.isfile(mod_file):
                continue
            try:
                with open(mod_file, encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            imported_names = _collect_imported_names(tree)
            imports_map[mod_name] = imported_names & self._tracked_modules
        return imports_map

    def _compute_reload_set(self, changed_modules: set[str]) -> set[str]:
        """Compute the full set of tracked modules that need reloading.

        Starting from the directly-changed modules, propagate upward: if module B
        is in the reload set and module A imports B (as determined by AST parsing),
        then A also needs reloading.
        """
        # Build importer graph among tracked modules:
        # imports_map[A] = {B, C} means module A imports modules B and C
        imports_map = self._build_tracked_imports_map()

        # Propagate: if any import of A is in the reload set, A needs reloading too
        reload_set = set(changed_modules)
        changed_expanded = True
        while changed_expanded:
            changed_expanded = False
            for mod_name, deps in imports_map.items():
                if mod_name not in reload_set and (deps & reload_set):
                    reload_set.add(mod_name)
                    changed_expanded = True

        return reload_set

    @staticmethod
    def _build_imports_map_for_set(modules: set[str]) -> dict[str, set[str]]:
        """Build import graph restricted to *modules* (bottom-up dependency graph)."""
        imports_map: dict[str, set[str]] = {}
        for mod_name in modules:
            mod = sys.modules.get(mod_name)
            if mod is None:
                imports_map[mod_name] = set()
                continue
            mod_file = getattr(mod, '__file__', None)
            if not mod_file or not os.path.isfile(mod_file):
                imports_map[mod_name] = set()
                continue
            try:
                with open(mod_file, encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
            except (SyntaxError, OSError, UnicodeDecodeError):
                imports_map[mod_name] = set()
                continue
            imports_map[mod_name] = _collect_imported_names(tree) & modules
        return imports_map

    @staticmethod
    def _kahn_sort_bottom_up(modules: set[str], imports_map: dict[str, set[str]]) -> list[str]:
        """Topological sort: modules with no dependencies in *modules* come first."""
        reverse_in_degree: dict[str, int] = dict.fromkeys(modules, 0)
        for mod_name, deps in imports_map.items():
            for dep in deps:
                if dep != mod_name and dep in modules:
                    reverse_in_degree[mod_name] = reverse_in_degree.get(mod_name, 0) + 1

        order: list[str] = []
        queue = [m for m in modules if reverse_in_degree.get(m, 0) == 0]
        while queue:
            mod_name = queue.pop(0)
            order.append(mod_name)
            for other_mod, deps in imports_map.items():
                if mod_name in deps and other_mod in modules and other_mod not in order:
                    reverse_in_degree[other_mod] -= 1
                    if reverse_in_degree[other_mod] == 0:
                        queue.append(other_mod)
        for m in modules:
            if m not in order:
                order.append(m)
        return order

    def _topological_reload_order(self, modules: set[str]) -> list[str]:
        """Return modules in bottom-up order (leaf dependencies first).

        Modules with no dependencies on other modules in the set come first.
        """
        imports_map = FunctionTracker._build_imports_map_for_set(modules)
        return FunctionTracker._kahn_sort_bottom_up(modules, imports_map)

    def _update_user_ns_from_module(
        self,
        module_name: str,
        module: types.ModuleType,
        user_ns: dict[str, Any]
    ) -> set[str]:
        """Update user_ns with fresh objects from a reloaded module.

        After a module is reloaded, the old function objects in user_ns still
        point to the stale code. This method finds and replaces them with the
        fresh versions from the reloaded module.

        Args:
            module_name: The module name
            module: The reloaded module object
            user_ns: User namespace to update in-place

        Returns:
            Set of variable names that were updated
        """
        updated = set()

        for var_name, var_value in list(user_ns.items()):
            if var_name.startswith('_'):
                continue

            # Check if this value came from the reloaded module
            value_module = getattr(var_value, '__module__', None)
            if value_module and (value_module == module_name or value_module.startswith(module_name + '.')):
                obj_name = getattr(var_value, '__name__', None) or getattr(var_value, '__qualname__', None)
                if obj_name:
                    # Try to get the fresh version from the reloaded module
                    fresh = getattr(module, obj_name, None)
                    if fresh is not None:
                        user_ns[var_name] = fresh
                        updated.add(var_name)
                        logger.debug(
                            "[AUTO_TRACK] Updated '%s' from reloaded module '%s'",
                            var_name, module_name
                        )

        return updated

    # ================================================================
    # Opaque call pattern detection
    # ================================================================

    def detect_opaque_call_patterns(self, code: str, user_ns: dict[str, Any]) -> list[str]:
        """Detect function call patterns that bypass source tracking.

        These patterns mean Cash cannot guarantee correct invalidation because
        the called function can't be identified statically.

        Detected patterns:
        1. Higher-order calls: `apply(func, args)`, `map(func, data)` with user vars
        2. getattr calls: `getattr(obj, 'method_name')()`
        3. Indirect calls via subscript: `funcs[0](x)`, `registry['handler'](x)`
        4. eval/exec: `eval('func(x)')`, `exec(code)`
        5. Starred calls: `func(*args)` where args could contain callables

        Args:
            code: Source code to analyze
            user_ns: User namespace for resolving variables

        Returns:
            List of human-readable warning messages
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        warnings: list[str] = []
        _OpaqueCallVisitor(warnings, user_ns).visit(tree)
        return warnings

class _OpaqueCallVisitor(ast.NodeVisitor):
    """AST visitor that detects call patterns Cash cannot statically track."""

    def __init__(self, warnings: list[str], user_ns: dict[str, Any]) -> None:
        self._warnings = warnings
        self._user_ns = user_ns

    def _check_getattr_dispatch(self, node: ast.Call) -> None:
        """Warn on getattr()() dynamic dispatch."""
        if isinstance(node.func, ast.Call):
            inner_func = node.func
            if isinstance(inner_func.func, ast.Name) and inner_func.func.id == 'getattr':
                self._warnings.append(
                    "Dynamic dispatch via getattr()() — Cash cannot track which function is called"
                )

    def _check_subscript_call(self, node: ast.Call) -> None:
        """Warn on registry['key']() / funcs[0]() patterns."""
        if isinstance(node.func, ast.Subscript):
            base = _get_ast_base_name(node.func.value)
            if base:
                self._warnings.append(
                    f"Indexed call {base}[...]() — Cash cannot track which function is called"
                )

    def _check_eval_exec(self, node: ast.Call) -> None:
        """Warn on eval()/exec() with embedded function calls."""
        if not (isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec')):
            return
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            try:
                inner_tree = ast.parse(node.args[0].value)
                has_calls = any(isinstance(n, ast.Call) for n in ast.walk(inner_tree))
                if has_calls:
                    self._warnings.append(
                        f"{node.func.id}() with function calls — Cash cannot track source changes"
                    )
            except SyntaxError:
                self._warnings.append(
                    f"{node.func.id}() with dynamic code — Cash cannot track source changes"
                )
        elif node.args and not isinstance(node.args[0], ast.Constant):
            self._warnings.append(
                f"{node.func.id}() with dynamic expression — Cash cannot track source changes"
            )

    def visit_Call(self, node: ast.Call) -> None:
        self._check_getattr_dispatch(node)
        self._check_subscript_call(node)
        # Pattern 3: Higher-order function calls with user-defined callable args
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ('apply', 'map', 'filter') or func_name not in _SAFE_HOF_NAMES:
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in self._user_ns:
                        val = self._user_ns[arg.id]
                        if (
                            callable(val)
                            and not isinstance(val, type)
                            and hasattr(val, '__module__')
                            and val.__module__ not in ('builtins', None)
                            and func_name in ('apply', 'map', 'filter', 'sorted', 'reduce')
                        ):
                            pass  # Trackable via input variable system
        self._check_eval_exec(node)
        self.generic_visit(node)


# Names of functions that are safe higher-order functions (their callable
# arguments are still trackable via the input variable system)
_SAFE_HOF_NAMES = frozenset({
    'sorted', 'min', 'max', 'map', 'filter', 'reduce',
    'functools.reduce', 'itertools.starmap',
    # pandas HOFs
    'apply', 'transform', 'agg', 'aggregate', 'pipe',
})

def _get_ast_base_name(node: ast.AST) -> str | None:
    """Get a human-readable name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_ast_base_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None

