"""CLI for inspecting and managing notebook caches."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_version() -> str:
    try:
        from cash import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def cmd_version(args: argparse.Namespace) -> None:
    """Show cash version."""
    print(f"cash {get_version()}")


def cmd_info(args: argparse.Namespace) -> None:
    """Show cash configuration."""
    from cash.config import get_config
    config = get_config()

    print(f"Cash v{get_version()}")
    print(f"  Backend:    {config.backend}")
    print(f"  Cache dir:  {config.cache_dir}")
    print(f"  Debug:      {config.debug}")
    print(f"  Compress:   {config.compress}")
    if config.max_cache_size is None:
        print("  Max size:   auto (scaled to disk/RAM per tier)")
    else:
        print(f"  Max size:   {config.max_cache_size / (1024**3):.1f} GB")
    # Report what actually decides persistence — the serialization-aware cost
    # model — rather than a raw threshold number.
    if config.smart_persistence:
        print("  Persist:    cost model (0.1s compute floor, "
              f"{config.min_cache_savings_pct:.0%} savings required)")
    else:
        print("  Persist:    cost model, conservative (1.0s compute floor)")
    if config.tiers:
        print(f"  Tiers:      {', '.join(t.type for t in config.tiers)}")
    print(f"  Source:     {config._source}")


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MB"
    return f"{size_bytes / (1024**3):.2f} GB"


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect cache for a notebook or cache directory."""
    target = args.path

    if target and os.path.isfile(target) and target.endswith('.ipynb'):
        _inspect_notebook(target)
    elif target and os.path.isdir(target):
        _inspect_cache_dir(target)
    else:
        # Default to .cash directory
        default_dir = ".cash"
        if os.path.isdir(default_dir):
            _inspect_cache_dir(default_dir)
        else:
            print("No cache found. Specify a notebook or cache directory.")
            sys.exit(1)


def _inspect_notebook(notebook_path: str) -> None:
    """Inspect a notebook and its associated cache."""
    print(f"Notebook: {notebook_path}")

    # Read notebook
    try:
        import nbformat
        nb = nbformat.read(notebook_path, as_version=4)
    except ImportError:
        print("  nbformat not installed. Install with: pip install nbformat")
        return
    except Exception as e:
        print(f"  Error reading notebook: {e}")
        return

    # Count cells
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    md_cells = [c for c in nb.cells if c.cell_type == 'markdown']
    print(f"  Code cells: {len(code_cells)}")
    print(f"  Markdown cells: {len(md_cells)}")

    # Check if %cash_on is used
    uses_cash = any(
        '%cash_on' in c.source or '%%cash' in c.source
        for c in code_cells
    )
    print(f"  Uses cash: {'Yes' if uses_cash else 'No'}")

    # Check for associated cache directory
    nb_dir = os.path.dirname(os.path.abspath(notebook_path))
    cache_dir = os.path.join(nb_dir, ".cash")
    if os.path.isdir(cache_dir):
        _inspect_cache_dir(cache_dir)
    else:
        print("  Cache: not found (no .cash directory)")


def _inspect_cache_dir(cache_dir: str) -> None:
    """Inspect a cache directory."""
    cache_path = Path(cache_dir)
    print(f"Cache directory: {cache_path}")

    # Count files and total size
    total_size = 0
    file_count = 0
    meta_count = 0
    data_extensions = {}

    for f in cache_path.rglob('*'):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            file_count += 1
            ext = f.suffix
            if ext == '.meta':
                meta_count += 1
            data_extensions[ext] = data_extensions.get(ext, 0) + 1

    print(f"  Total files: {file_count}")
    print(f"  Total size:  {_format_bytes(total_size)}")
    print(f"  Cache entries: ~{meta_count}")

    if data_extensions:
        print(f"  File types: {dict(sorted(data_extensions.items()))}")

    # Try reading some metadata
    meta_files = list(cache_path.glob('*.meta'))
    if meta_files:
        print("\n  Recent entries:")
        import pickle
        shown = 0
        for mf in sorted(meta_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
            try:
                with open(mf, 'rb') as f:
                    metadata = pickle.load(f)
                key = metadata.get('key', mf.stem[:16] + '...')
                created = metadata.get('created_at', 'unknown')
                if isinstance(created, (int, float)):
                    from datetime import datetime
                    created = datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M')
                outputs = metadata.get('outputs', [])
                print(f"    [{created}] {key[:50]}  -> {', '.join(outputs) if outputs else 'no outputs'}")
                shown += 1
            except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
                logger.debug("Failed to read cache metadata from %s: %s", mf, exc)
        if shown == 0:
            print("    (could not read metadata)")


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear cache."""
    if args.all:
        # Clear default .cash directory
        cache_dir = ".cash"
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared: {os.path.abspath(cache_dir)}")
        else:
            print("No .cash directory found in current directory")
        return

    target = args.path
    if not target:
        print("Specify a path or use --all to clear all caches")
        sys.exit(1)

    if os.path.isdir(target):
        shutil.rmtree(target)
        print(f"Cleared: {target}")
    elif os.path.isfile(target) and target.endswith('.ipynb'):
        nb_dir = os.path.dirname(os.path.abspath(target))
        cache_dir = os.path.join(nb_dir, ".cash")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared: {cache_dir}")
        else:
            print(f"No cache found for {target}")
    else:
        print(f"Not found: {target}")
        sys.exit(1)


HOOK_FILENAME = "00-cash.py"
HOOK_MARKER = "# cash-ipython-hook (managed by `cash autoload`)"

HOOK_BODY_AVAILABLE = f'''{HOOK_MARKER}
# Mode: available — `%cash_on` (and `cash.cache`) ready to use in every session.
# Remove with `cash autoload off`.
import cash  # auto-registers cash IPython magics
'''

HOOK_BODY_ACTIVE = f'''{HOOK_MARKER}
# Mode: active — caching is enabled automatically in every IPython/Jupyter
# session.  Run %cash_off in any session you want to opt out of, or remove
# this file with `cash autoload off`.
import cash  # auto-registers cash IPython magics

try:
    _ip = get_ipython()  # noqa: F821  (IPython injects this at startup)
except NameError:
    _ip = None
if _ip is not None:
    _ip.run_line_magic("cash_on", "")
'''


def _ipython_startup_dir(profile: str) -> Path:
    """Return the IPython startup directory for ``profile``.

    Uses :mod:`IPython.paths` when IPython is importable; falls back to the
    documented default layout (``~/.ipython/profile_<name>/startup``) so the
    installer remains usable on machines where IPython has not been imported
    yet (it'll still be imported the moment Jupyter starts).
    """
    try:
        from IPython.paths import get_ipython_dir
        ipython_dir = Path(get_ipython_dir())
    except ImportError:
        ipython_dir = Path.home() / ".ipython"
    return ipython_dir / f"profile_{profile}" / "startup"


def _is_cash_hook(content: str) -> bool:
    """True if the file content was written by a current or prior `cash autoload`."""
    return "cash-ipython-hook" in content


def cmd_autoload_on(args: argparse.Namespace) -> None:
    """Write a startup file so cash is available (or active) in every kernel."""
    startup_dir = _ipython_startup_dir(args.profile)
    hook_path = startup_dir / HOOK_FILENAME

    body = HOOK_BODY_ACTIVE if args.mode == "active" else HOOK_BODY_AVAILABLE

    if hook_path.exists() and not args.force:
        existing = hook_path.read_text(encoding="utf-8")
        if existing == body:
            print(f"Autoload already on (mode={args.mode}): {hook_path}")
            return
        print(f"Refusing to overwrite existing file: {hook_path}")
        print("  Pass --force to replace it, or run `cash autoload off` first.")
        sys.exit(1)

    startup_dir.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(body, encoding="utf-8")

    print(f"Autoload on (mode={args.mode}): {hook_path}")
    if args.mode == "active":
        print("  Every new IPython/Jupyter kernel will auto-import cash and run %cash_on.")
        print("  Run %cash_off in a notebook to opt out for that session.")
    else:
        print("  Every new IPython/Jupyter kernel will auto-import cash so %cash_on works without a prior import.")
    print("  Disable with: cash autoload off")


def cmd_autoload_off(args: argparse.Namespace) -> None:
    """Remove the startup file written by ``cash autoload on``."""
    startup_dir = _ipython_startup_dir(args.profile)
    hook_path = startup_dir / HOOK_FILENAME

    if not hook_path.exists():
        print(f"Autoload not installed at: {hook_path}")
        return

    content = hook_path.read_text(encoding="utf-8")
    if not _is_cash_hook(content) and not args.force:
        print(f"Refusing to remove unrecognized file: {hook_path}")
        print("  Pass --force if you really want to delete it.")
        sys.exit(1)

    hook_path.unlink()
    print(f"Autoload off: {hook_path}")


def cmd_autoload(args: argparse.Namespace) -> None:
    """Dispatch ``cash autoload on|off`` to the appropriate handler."""
    if args.state == "on":
        cmd_autoload_on(args)
    elif args.state == "off":
        cmd_autoload_off(args)
    else:  # argparse choices guards against this
        raise AssertionError(f"unexpected state {args.state!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='cash',
        description='Cash - Smart caching for Jupyter notebooks',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # version
    sub_version = subparsers.add_parser('version', help='Show cash version')
    sub_version.set_defaults(func=cmd_version)

    # info
    sub_info = subparsers.add_parser('info', help='Show cash configuration')
    sub_info.set_defaults(func=cmd_info)

    # inspect
    sub_inspect = subparsers.add_parser('inspect', help='Inspect cache for a notebook or directory')
    sub_inspect.add_argument('path', nargs='?', default=None, help='Notebook (.ipynb) or cache directory path')
    sub_inspect.set_defaults(func=cmd_inspect)

    # clear
    sub_clear = subparsers.add_parser('clear', help='Clear cache')
    sub_clear.add_argument('path', nargs='?', default=None, help='Notebook or cache directory to clear')
    sub_clear.add_argument('--all', action='store_true', help='Clear all caches in current directory')
    sub_clear.set_defaults(func=cmd_clear)

    # autoload on|off
    sub_autoload = subparsers.add_parser(
        'autoload',
        help='Toggle whether cash auto-loads in every new IPython/Jupyter kernel',
        description=(
            'Install or remove an IPython startup hook so cash is loaded (and optionally '
            'enabled) automatically in every new kernel — no `import cash` needed per notebook.'
        ),
    )
    sub_autoload.add_argument(
        'state', choices=['on', 'off'],
        help='on: install the startup hook. off: remove it.',
    )
    sub_autoload.add_argument(
        '--mode',
        choices=['available', 'active'],
        default='active',
        help='(on only) available: just `import cash`. active (default): also run %%cash_on so caching is on by default.',
    )
    sub_autoload.add_argument(
        '--profile', default='default',
        help='IPython profile to target (default: "default")',
    )
    sub_autoload.add_argument(
        '--force', action='store_true',
        help='(on) overwrite a different file at this path. (off) remove a file lacking the cash marker.',
    )
    sub_autoload.set_defaults(func=cmd_autoload)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
