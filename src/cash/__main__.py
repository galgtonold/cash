"""CLI for inspecting and managing notebook caches."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
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


@dataclass
class _Entry:
    """One cache entry on disk: its ``.meta``, its ``.data``, and who wrote it."""
    stem: str
    function: str
    key: str
    size: int
    mtime: float
    # What the entry is WORTH, which is the number a delete decision turns on:
    # bytes alone say what you get back, not what it costs you to lose.
    saves: float = 0.0
    uses: int = 0
    outputs: tuple[str, ...] = ()


# What a user may type instead of the literal ``(notebook statements)`` group
# heading. The heading has to read as prose in a table; it should not have to
# be typed with its brackets to be addressable.
NOTEBOOK_GROUP = '(notebook statements)'
_NOTEBOOK_ALIASES = frozenset({'notebook', 'notebooks', 'statements',
                               'notebook statements'})


def _function_of(key: str) -> str:
    """The function a cache key belongs to.

    Decorator keys are ``{module.qualname}:{state}:{dynamic}:{args}``, so the
    owner is simply the first segment -- no extra metadata needed to group a
    cache directory by function. Notebook statements are keyed ``stmt:<sha>``
    and have no function to name, so they are collected under one heading
    rather than reported as a function called "stmt".
    """
    if key.startswith('stmt:'):
        return '(notebook statements)'
    return key.split(':', 1)[0] if ':' in key else '(unknown)'


def _scan_entries(cache_path: Path) -> list[_Entry]:
    """Read every ``.meta`` in *cache_path* and pair it with its payload."""
    entries: list[_Entry] = []
    for meta_file in cache_path.glob('*.meta'):
        try:
            with open(meta_file, 'rb') as fh:
                metadata = pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            logger.debug("Failed to read cache metadata from %s: %s", meta_file, exc)
            continue
        key = metadata.get('key') or ''
        size = meta_file.stat().st_size
        data_file = meta_file.with_suffix('.data')
        if data_file.exists():
            size += data_file.stat().st_size
        outputs = metadata.get('outputs') or ()
        entries.append(_Entry(
            stem=meta_file.stem,
            function=_function_of(key),
            key=key,
            size=size,
            mtime=meta_file.stat().st_mtime,
            saves=float(metadata.get('execution_time') or 0.0),
            uses=int(metadata.get('access_count') or 0),
            outputs=tuple(str(o) for o in outputs),
        ))
    return entries


def _resolve_entry(entries: list[_Entry], wanted: str) -> _Entry | None:
    """Find one entry by id, accepting any unambiguous prefix.

    The ids are SHA-256 stems, so nobody is going to type one; a prefix is the
    only usable handle, the same bargain a short commit hash makes.
    """
    matches = [e for e in entries if e.stem.startswith(wanted)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"No cache entry starts with {wanted!r}.")
        print("Run `cash inspect --function NAME` to list entry ids.")
        return None
    print(f"{wanted!r} is ambiguous - it matches {len(matches)} entries:")
    for entry in matches[:10]:
        print(f"  {entry.stem[:16]}  {entry.function}")
    print("Use more characters.")
    return None


def _age(mtime: float) -> str:
    seconds = max(0.0, time.time() - mtime)
    for limit, divisor, unit in ((60, 1, "s"), (3600, 60, "min"),
                                 (86400, 3600, "h")):
        if seconds < limit:
            return f"{int(seconds // divisor)}{unit} ago"
    return f"{int(seconds // 86400)}d ago"


def _resolve_function(entries: list[_Entry], wanted: str) -> str | None:
    """Map what the user typed onto one function name, or explain why not.

    A decorator key carries the full ``module.qualname``, and for a function
    defined in the script you ran that module is ``__main__`` -- which nobody
    wants to type. So an unambiguous trailing segment is accepted too:
    ``--function work`` finds ``__main__.work``. Ambiguity is reported with
    the candidates rather than resolved by guessing, because the two remedies
    (clear one, clear the other) are not interchangeable.
    """
    names = sorted({e.function for e in entries})
    if wanted in names:
        return wanted
    if wanted.strip().lower() in _NOTEBOOK_ALIASES and NOTEBOOK_GROUP in names:
        return NOTEBOOK_GROUP
    matches = [n for n in names if n.rsplit('.', 1)[-1] == wanted
               or n.endswith('.' + wanted)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"No cached function matches {wanted!r}.")
        if names:
            print("Cached functions:")
            for name in names:
                print(f"  {name}")
        return None
    # ASCII only in CLI output: an em-dash renders as "?" on a cp1252 Windows
    # console, which is where a lot of these users are.
    print(f"{wanted!r} is ambiguous - it matches:")
    for name in matches:
        print(f"  {name}")
    print("Pass the full name.")
    return None


def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect cache for a notebook or cache directory."""
    target = args.path
    # getattr, not attribute access: a flag added here must not break a
    # caller that builds its own Namespace without it.
    only_function = getattr(args, "function", None)

    if target and os.path.isfile(target) and target.endswith('.ipynb'):
        if only_function:
            print("--function applies to a cache directory, not a notebook.")
            sys.exit(2)
        _inspect_notebook(target)
        return

    cache_dir = target if (target and os.path.isdir(target)) else ".cash"
    if not os.path.isdir(cache_dir):
        print("No cache found. Specify a notebook or cache directory.")
        sys.exit(1)
    _inspect_cache_dir(cache_dir, only_function=only_function)


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


def _inspect_cache_dir(cache_dir: str, only_function: str | None = None) -> None:
    """Inspect a cache directory.

    The default view is a per-function table sorted by SIZE, because the
    question that sends anyone here is "what is filling my disk, and what can
    I afford to drop?". A user who wanted that used to get a file-extension
    histogram and had to go to the file explorer instead.
    """
    cache_path = Path(cache_dir)
    total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
    entries = _scan_entries(cache_path)

    print(f"Cache directory: {cache_path}")

    if only_function is not None:
        resolved = _resolve_function(entries, only_function)
        if resolved is None:
            sys.exit(1)
        owned = sorted((e for e in entries if e.function == resolved),
                       key=lambda e: e.size, reverse=True)
        owned_size = sum(e.size for e in owned)
        noun = "entry" if len(owned) == 1 else "entries"
        print(f"  {resolved} - {len(owned)} {noun}, {_format_bytes(owned_size)}\n")

        # SAVES is the column a delete decision actually turns on: bytes say
        # what you get back, seconds say what it costs you to lose. The old
        # view showed an opaque id, a size and an age -- enough to see that
        # entries exist, not enough to choose between them.
        shows_outputs = any(e.outputs for e in owned)
        header = (f"  {'ENTRY':<14}{'SAVES':>9}{'SIZE':>11}{'USES':>7}"
                  f"   {'LAST USED':<12}")
        print((header + "PRODUCES") if shows_outputs else header.rstrip())
        for entry in owned:
            saves = f"{entry.saves:.1f}s" if entry.saves else "-"
            produces = ", ".join(entry.outputs) if shows_outputs else ""
            row = (f"  {entry.stem[:12]:<14}{saves:>9}"
                   f"{_format_bytes(entry.size):>11}{str(entry.uses) + 'x':>7}"
                   f"   {_age(entry.mtime):<12}{produces}")
            print(row.rstrip())
        print("\n  cash clear --entry ID   to drop one of these "
              "(any unambiguous prefix)")
        return

    functions: dict[str, list[_Entry]] = {}
    for entry in entries:
        functions.setdefault(entry.function, []).append(entry)

    print(f"  Total size: {_format_bytes(total_size)}    "
          f"Entries: {len(entries)}    Functions: {len(functions)}")

    if not functions:
        print("\n  (no readable entries)")
        return

    ranked = sorted(functions.items(),
                    key=lambda kv: sum(e.size for e in kv[1]), reverse=True)
    print(f"\n  {'FUNCTION':<40}{'ENTRIES':>9}{'SIZE':>12}   LAST USED")
    for name, owned in ranked:
        newest = max(e.mtime for e in owned)
        print(f"  {name[:40]:<40}{len(owned):>9}"
              f"{_format_bytes(sum(e.size for e in owned)):>12}   {_age(newest)}")
    print("\n  cash inspect --function NAME   to list one function's entries")
    print("  cash clear   --function NAME   to drop them")
    if NOTEBOOK_GROUP in functions:
        # The heading reads as prose in the table; say plainly that it is
        # addressable without having to type the brackets.
        print("  --function notebook            for the notebook statements")


def _clear_function(cache_dir: str, wanted: str) -> None:
    """Delete every entry belonging to one cached function.

    The alternative for someone short on disk used to be all-or-nothing: keep
    a cache they cannot afford or delete work they still want. Dropping the
    one function they are finished with is the decision they actually wanted
    to make.
    """
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        print(f"No cache directory at {cache_dir}")
        sys.exit(1)
    entries = _scan_entries(cache_path)
    resolved = _resolve_function(entries, wanted)
    if resolved is None:
        sys.exit(1)

    owned = [e for e in entries if e.function == resolved]
    freed = sum(e.size for e in owned)
    removed = 0
    for entry in owned:
        _remove_entry_files(cache_path, entry.stem)
        removed += 1
    noun = "entry" if removed == 1 else "entries"
    print(f"Cleared {removed} {noun} for {resolved} ({_format_bytes(freed)} freed)")


def _remove_entry_files(cache_path: Path, stem: str) -> None:
    """Delete one entry's ``.meta`` and ``.data``, surviving a locked file."""
    for suffix in ('.meta', '.data'):
        path = cache_path / f"{stem}{suffix}"
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            # Antivirus or another process holding it. Partial progress still
            # frees space, and saying so beats a traceback part way through.
            print(f"  could not remove {path.name}: {exc}")


def _clear_entry(cache_dir: str, wanted: str) -> None:
    """Delete a single cache entry by id."""
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        print(f"No cache directory at {cache_dir}")
        sys.exit(1)
    entry = _resolve_entry(_scan_entries(cache_path), wanted)
    if entry is None:
        sys.exit(1)
    _remove_entry_files(cache_path, entry.stem)
    print(f"Cleared entry {entry.stem[:12]} from {entry.function} "
          f"({_format_bytes(entry.size)} freed)")


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear cache."""
    only_entry = getattr(args, "entry", None)
    if only_entry:
        target = args.path if (args.path and os.path.isdir(args.path)) else ".cash"
        _clear_entry(target, only_entry)
        return

    only_function = getattr(args, "function", None)
    if only_function:
        target = args.path if (args.path and os.path.isdir(args.path)) else ".cash"
        _clear_function(target, only_function)
        return

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
        # The one-liner this replaces named two of the three options and
        # left the user to guess the rest; the help text is the list.
        parser = getattr(args, "clear_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print("Specify a path, --function NAME, or --all.")
        sys.exit(2)

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
        description='A Python cache that re-runs only what changed.',
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
    sub_inspect.add_argument('--function', default=None, metavar='NAME',
                             help="List one function's entries, with what each one saves. An unambiguous "
                                  'trailing segment is enough ("work" finds "__main__.work").')
    sub_inspect.set_defaults(func=cmd_inspect)

    # clear
    sub_clear = subparsers.add_parser('clear', help='Clear cache')
    sub_clear.add_argument('path', nargs='?', default=None, help='Notebook or cache directory to clear')
    sub_clear.add_argument('--all', action='store_true', help='Clear all caches in current directory')
    sub_clear.add_argument('--function', default=None, metavar='NAME',
                           help="Clear only this function's entries, leaving the rest "
                                'of the cache intact. "notebook" selects the '
                                'notebook statements.')
    sub_clear.add_argument('--entry', default=None, metavar='ID',
                           help='Clear one entry by id, as listed by '
                                '`cash inspect --function NAME`. Any unambiguous '
                                'prefix works. Takes precedence over --function.')
    sub_clear.set_defaults(func=cmd_clear, clear_parser=sub_clear)

    # autoload on|off
    sub_autoload = subparsers.add_parser(
        'autoload',
        help='Toggle whether cash auto-loads in every new IPython/Jupyter kernel',
        description=(
            'Install or remove an IPython startup hook so cash is loaded (and optionally '
            'enabled) automatically in every new kernel - no `import cash` needed per notebook.'
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
