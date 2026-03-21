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
    print(f"  Backend:    {config.backend_type}")
    print(f"  Cache dir:  {config.cache_dir}")
    print(f"  Debug:      {config.debug}")
    print(f"  Compress:   {config.compress}")
    print(f"  Max size:   {config.max_cache_size / (1024**3):.1f} GB")
    print(f"  Threshold:  {config.smart_persistence_threshold}s")

    # Check if config file exists
    config_path = Path.home() / ".cash" / "config.toml"
    if config_path.exists():
        print(f"  Config:     {config_path}")
    else:
        print("  Config:     (defaults, no config file)")


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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
