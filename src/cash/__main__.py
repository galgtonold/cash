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

from cash.backends.entry_format import ENTRY_SUFFIX, read_entry

logger = logging.getLogger(__name__)


def get_version() -> str:
    try:
        from cash import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def resolved_cache_dir() -> str:
    """The cache directory the LIBRARY would use, for commands that act on it.

    ``inspect`` and ``clear`` used to assume ``./.cash`` while ``info`` read
    the merged config, so the two commands whose whole job is to act on the
    cache acted on a different one from the library that wrote it. Setting
    ``CASH_CACHE_DIR`` and then running ``cash inspect`` reported on nothing,
    and ``cash clear --all`` reported success having deleted a directory the
    user did not mean (CAS-83, reproduced by a round-16 tester).

    Going through ``get_config()`` means one merge order for everyone --
    defaults, user TOML, project TOML, environment -- so the CLI cannot drift
    from the library again. It also follows the project anchor, which is what
    makes the CLI usable at all now that the default is not relative to the
    directory you are standing in.
    """
    try:
        from cash.config import get_config
        return str(get_config().cache_dir)
    except Exception:  # noqa: BLE001 - a broken config must not break `clear`
        return ".cash"


def tool_cache_dir(name: str) -> str:
    """The per-user cache an installed console script named *name* uses.

    An installed tool run from outside any project caches under the
    platform's cache root, one directory per tool. The CLI cannot infer which
    tool you mean -- it is itself a different console script -- so ``--tool``
    names it.
    """
    from cash.config import _per_user_cache_root
    return str(_per_user_cache_root() / name)


def _target_dir(args: argparse.Namespace) -> str:
    """The directory a subcommand acts on when no path was given."""
    tool = getattr(args, "tool", None)
    return tool_cache_dir(tool) if tool else resolved_cache_dir()


def _per_user_tool_caches() -> list[tuple[str, str, int, int]]:
    """``(tool, path, entries, bytes)`` for every per-user tool cache."""
    try:
        from cash.config import _per_user_cache_root
        root = _per_user_cache_root()
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except (OSError, RuntimeError):
        return []
    found = []
    for child in children:
        entries = size = 0
        try:
            for f in child.iterdir():
                if f.name.endswith(ENTRY_SUFFIX):
                    entries += 1
                    size += f.stat().st_size
        except OSError:
            continue
        found.append((child.name, str(child), entries, size))
    return found


def cmd_version(args: argparse.Namespace) -> None:
    """Show cash version."""
    print(f"cash {get_version()}")


def cmd_info(args: argparse.Namespace) -> None:
    """Show cash configuration."""
    from cash.config import get_config
    config = get_config()

    from cash.config import format_size
    origins = getattr(config, "_origins", {})

    print(f"Cash v{get_version()}")
    print(f"  Backend:    {config.backend}")
    print(f"  Cache dir:  {config.cache_dir}")
    if config.disable:
        print(f"  Disabled:   yes -- every cached function runs uncached "
              f"({origins.get('disable', 'disable = true')})")
    # Resolved, not just configured. "auto (scaled per tier)" is true and
    # useless: a user asking what their cache is allowed to hold needs the two
    # numbers it actually resolves to, and the RAM one in particular appears
    # nowhere else -- a tester spent a round reading a growing RSS as a leak
    # when it was a 4 GiB cap doing exactly what it says.
    from cash.backends.adaptive_caps import (
        human_bytes,
        resolve_disk_cap,
        resolve_ram_cap,
    )
    if config.max_cache_size is None:
        disk = human_bytes(resolve_disk_cap(config.cache_dir))
        print(f"  Max size:   auto -- disk {disk}, RAM {human_bytes(resolve_ram_cap())}")
    else:
        print(f"  Max size:   {format_size(config.max_cache_size)} "
              f"({config.max_cache_size:,} bytes) on disk, "
              f"RAM {human_bytes(resolve_ram_cap())}")
    # Report what actually decides persistence — the serialization-aware cost
    # model — rather than a raw threshold number.
    if config.smart_persistence:
        print("  Persist:    cost model (0.1s compute floor, "
              f"{config.min_cache_savings_pct:.0%} savings required)")
    else:
        print("  Persist:    cost model, conservative (1.0s compute floor)")
    if config.tiers:
        print(f"  Tiers:      {', '.join(t.type for t in config.tiers)}")
    # Where this run looked, and what each layer set. Round 18: a nested
    # pyproject.toml, a pytest launched from the directory above its project
    # and a `disable = true` were each invisible here -- a Source line names
    # the layers, not which file, nor which setting came from where.
    from cash.config import TOML_FLAT, TOML_MISSING, TOML_NOT_CASH, TOML_SECTION
    outcome = {TOML_SECTION: "read", TOML_FLAT: "read", TOML_MISSING: "not found",
               TOML_NOT_CASH: "no [tool.cash] section"}
    files = getattr(config, "_files", [])
    if files:
        print("  Config files:")
        for layer, path, found in files:
            print(f"    {layer:<12} {path}  ({outcome.get(found, 'could not be read')})")
    if origins:
        print("  Settings (where each came from):")
        for key in sorted(origins):
            print(f"    {key + ' = ' + _setting_text(config, key):<40} {origins[key]}")
    else:
        print("  Settings:   all defaults")
    print(f"  Source:     {config._source}")
    # Installed tools run from outside a project cache per user, per tool --
    # somewhere this command cannot reach by default, because it is a
    # different console script. Say where they are, so an operator does not
    # have to know the platform's cache root to find them.
    tools = _per_user_tool_caches()
    if tools:
        print("  Tool caches (reach one with --tool NAME):")
        for name, path, entries, size in tools:
            print(f"    {name:<20} {entries:>5} entries  {_format_bytes(size):>10}  {path}")


def _setting_text(config, key: str) -> str:
    """One setting's effective value, as `cash info` prints it."""
    from cash.config import _SIZE_FIELDS, format_size
    value = getattr(config, key, None)
    if key == "tiers":
        return ", ".join(t.type for t in value) or "[]"
    if key == "redis_password" and value:
        return "***"
    if key in _SIZE_FIELDS and isinstance(value, int):
        return format_size(value)
    return repr(value)


def _format_bytes(size_bytes: int) -> str:
    # Powers of 1024, so labelled as such: "GB" here read as a mis-parsed
    # "2GB" setting (round 18).
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MiB"
    return f"{size_bytes / (1024**3):.2f} GiB"


@dataclass
class _Entry:
    """One cache entry on disk: its file, its key, and who wrote it."""
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
    # The files the entry was computed from: the question behind most "why
    # did this recompute?" and "why did this NOT recompute?" reports.
    reads: tuple[str, ...] = ()
    # When the entry stops being served (created_at + ttl), or None for none.
    expires: float | None = None


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
    """Read every entry's metadata in *cache_path*.

    Only the metadata region of each file is read, never the payload, so
    listing a cache full of large frames costs the same as listing one full
    of small ones.
    """
    entries: list[_Entry] = []
    for entry_file in cache_path.glob(f'*{ENTRY_SUFFIX}'):
        try:
            metadata, _ = read_entry(str(entry_file), with_payload=False)
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            logger.debug("Failed to read cache metadata from %s: %s", entry_file, exc)
            continue
        key = metadata.get('key') or ''
        stat = entry_file.stat()
        outputs = metadata.get('outputs') or ()
        entries.append(_Entry(
            stem=entry_file.stem,
            function=_function_of(key),
            key=key,
            size=stat.st_size,
            mtime=stat.st_mtime,
            saves=float(metadata.get('execution_time') or 0.0),
            uses=int(metadata.get('access_count') or 0),
            outputs=tuple(str(o) for o in outputs),
            reads=tuple(str(p) for p in (metadata.get('auto_file_deps') or {})),
            expires=(float(metadata.get('created_at') or stat.st_mtime) + float(metadata['ttl'])
                     if metadata.get('ttl') is not None else None),
        ))
    return entries


def _expires(when: float | None) -> str:
    """``-`` for no ttl, ``expired``, or how long is left (``in 5m``)."""
    if when is None:
        return "-"
    left = when - time.time()
    if left <= 0:
        return "expired"
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if left >= size:
            return f"in {int(left // size)}{unit}"
    return f"in {int(left)}s"


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
    if target and getattr(args, "tool", None):
        print("cash inspect: --tool and a path are mutually exclusive.")
        sys.exit(2)

    if target and os.path.isfile(target) and target.endswith('.ipynb'):
        if only_function:
            print("--function applies to a cache directory, not a notebook.")
            sys.exit(2)
        _inspect_notebook(target)
        return

    cache_dir = target if (target and os.path.isdir(target)) else _target_dir(args)
    if not os.path.isdir(cache_dir):
        print(f"No cache found at {os.path.abspath(cache_dir)}.")
        print("Specify a notebook or cache directory, or set CASH_CACHE_DIR.")
        sys.exit(1)
    # The directory is printed once, by `_inspect_cache_dir`, absolute: this
    # used to print it too, so every default `cash inspect` began with two
    # headers naming the same place (round 18).
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

    print(f"Cache directory: {cache_path.resolve()}")

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
        # EXPIRES: an entry with a ttl stops being served at a time the table
        # could not show, so an expired entry looked like a live one.
        header = (f"  {'ENTRY':<14}{'SAVES':>9}{'SIZE':>11}{'USES':>7}"
                  f"   {'LAST USED':<12}{'EXPIRES':<12}")
        print((header + "PRODUCES") if shows_outputs else header.rstrip())
        for entry in owned:
            saves = f"{entry.saves:.1f}s" if entry.saves else "-"
            produces = ", ".join(entry.outputs) if shows_outputs else ""
            row = (f"  {entry.stem[:12]:<14}{saves:>9}"
                   f"{_format_bytes(entry.size):>11}{str(entry.uses) + 'x':>7}"
                   f"   {_age(entry.mtime):<12}{_expires(entry.expires):<12}{produces}")
            print(row.rstrip())
            if entry.reads:
                shown = ", ".join(entry.reads[:3])
                more = f" and {len(entry.reads) - 3} more" if len(entry.reads) > 3 else ""
                print(f"      reads: {shown}{more}")
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
    _bump_generation(cache_path)
    noun = "entry" if removed == 1 else "entries"
    print(f"Cleared {removed} {noun} for {resolved} ({_format_bytes(freed)} freed)")


def _remove_entry_files(cache_path: Path, stem: str) -> None:
    """Delete one entry's file, surviving a locked one."""
    path = cache_path / f"{stem}{ENTRY_SUFFIX}"
    try:
        path.unlink()
    except FileNotFoundError:
        return
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
    _bump_generation(cache_path)
    print(f"Cleared entry {entry.stem[:12]} from {entry.function} "
          f"({_format_bytes(entry.size)} freed)")


def _bump_generation(cache_path: Path) -> None:
    """Tell running processes that entries were removed under them.

    A process keeps results in RAM, and a clear of some entries left those
    served (round 18). Replacing the format stamp gives it a new identity,
    which a running `TieredBackend` notices within a second and drops its RAM
    tier. (`--all` needs nothing: the stamp goes with the directory.)
    """
    stamp = cache_path / "CACHE_VERSION"
    try:
        if stamp.exists():
            tmp = cache_path / "CACHE_VERSION.tmp"
            tmp.write_text(stamp.read_text(encoding="utf-8"), encoding="utf-8")
            os.replace(tmp, stamp)
    except OSError:
        logger.debug("Could not refresh %s", stamp, exc_info=True)


def _looks_like_a_cache(cache_dir: str) -> bool:
    """Does this directory hold a cash cache, or something else entirely?

    ``clear --all`` now deletes a directory the user did not type -- whatever
    the config resolved to. That is the point of the fix, and it is also a
    reason to look before recursively removing: a mistyped ``CASH_CACHE_DIR``
    used to cost the user nothing because the CLI ignored it.
    """
    if os.path.exists(os.path.join(cache_dir, "CACHE_VERSION")):
        return True
    try:
        entries = os.listdir(cache_dir)
    except OSError:
        return False
    return not entries or any(e.endswith(ENTRY_SUFFIX) for e in entries)


def _contains_cwd(path: str) -> bool:
    """Is *path* the current directory, or one of its ancestors?"""
    try:
        here = os.path.normcase(os.path.realpath(os.getcwd()))
        there = os.path.normcase(os.path.realpath(path))
    except OSError:
        return False
    return here == there or here.startswith(there.rstrip(os.sep) + os.sep)


def _rmtree_cache(cache_dir: str, force: bool = False) -> None:
    """Remove a cache directory, having checked that it is one.

    Every removal goes through here -- an explicit path, ``--all``, ``--tool``,
    a notebook's sibling ``.cash``. The explicit path used to go straight to
    ``shutil.rmtree``: round 17 ran ``cash clear .`` in a project, which
    deleted the project's files and then crashed trying to remove the
    directory it was standing in (CAS-107). "Destructive without confirmation"
    covered deleting a cache you meant to delete; it never covered deleting
    something that was not a cache at all.
    """
    resolved = os.path.abspath(cache_dir)
    if _contains_cwd(resolved):
        # Never, even with --force: nobody means to delete the directory they
        # are standing in or anything above it, and on Windows the removal
        # cannot even complete -- it deletes the contents, then fails.
        print(f"Refusing to clear {resolved}: it is the current directory or "
              f"contains it. Change to another directory first.")
        sys.exit(1)
    if not force and not _looks_like_a_cache(resolved):
        print(f"Refusing to clear {resolved}: it does not look like a cash "
              f"cache (no CACHE_VERSION and no {ENTRY_SUFFIX} files).")
        print("Check the path, CASH_CACHE_DIR and [tool.cash] cache_dir. If it "
              "really is a cache that lost its marker, clear it with --force.")
        sys.exit(1)
    shutil.rmtree(resolved)
    print(f"Cleared: {resolved}")


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear cache."""
    if args.all and args.path:
        # "everything" and "this one directory" are two different requests and
        # the flag reads as neither. It used to accept both and silently drop
        # the path, clearing a cache the user had not named -- the one
        # behaviour that cannot be right (CAS-83).
        print("cash clear: --all and a path are mutually exclusive.")
        print(f"  To clear that directory:   cash clear {args.path}")
        print(f"  To clear the cache in use: cash clear --all "
              f"  ({os.path.abspath(resolved_cache_dir())})")
        sys.exit(2)

    tool = getattr(args, "tool", None)
    if tool and args.path:
        print("cash clear: --tool and a path are mutually exclusive.")
        sys.exit(2)

    only_entry = getattr(args, "entry", None)
    if only_entry:
        target = args.path if (args.path and os.path.isdir(args.path)) else _target_dir(args)
        _clear_entry(target, only_entry)
        return

    only_function = getattr(args, "function", None)
    if only_function:
        target = args.path if (args.path and os.path.isdir(args.path)) else _target_dir(args)
        _clear_function(target, only_function)
        return

    force = bool(getattr(args, "force", False))
    if args.all or tool:
        cache_dir = _target_dir(args)
        if os.path.isdir(cache_dir):
            _rmtree_cache(cache_dir, force=force)
        else:
            # "Nothing here" is true and was not enough: a live service kept
            # its whole cache while this reported success (round 18). Say
            # which directory was looked at, and the two ways a running
            # program's cache is somewhere else.
            print(f"Nothing cleared: no cache at {os.path.abspath(cache_dir)}, "
                  f"the directory `cash info` reports for here.")
            print("  A script outside any project (no pyproject.toml, setup.py, "
                  "setup.cfg or .git above it) caches in a .cash beside the script: "
                  "cash clear <script dir>/.cash")
            print("  A cache_dir that was changed leaves the old directory behind: "
                  "cash clear <old path>")
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
        _rmtree_cache(target, force=force)
    elif os.path.isfile(target) and target.endswith('.ipynb'):
        nb_dir = os.path.dirname(os.path.abspath(target))
        cache_dir = os.path.join(nb_dir, ".cash")
        if os.path.isdir(cache_dir):
            _rmtree_cache(cache_dir, force=force)
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
    sub_inspect.add_argument('path', nargs='?', default=None,
                             help='Notebook (.ipynb) or cache directory path. '
                                  'Defaults to the cache the library is using.')
    sub_inspect.add_argument('--function', default=None, metavar='NAME',
                             help="List one function's entries, with what each one saves. An unambiguous "
                                  'trailing segment is enough ("work" finds "__main__.work").')
    sub_inspect.add_argument('--tool', default=None, metavar='NAME',
                             help='Inspect the per-user cache of the installed console '
                                  'script NAME (listed by `cash info`).')
    sub_inspect.set_defaults(func=cmd_inspect)

    # clear
    sub_clear = subparsers.add_parser('clear', help='Clear cache')
    sub_clear.add_argument('path', nargs='?', default=None, help='Notebook or cache directory to clear')
    sub_clear.add_argument('--all', action='store_true',
                           help='Clear the cache the library is using -- the same '
                                'directory `cash info` reports. Cannot be combined '
                                'with a path.')
    sub_clear.add_argument('--function', default=None, metavar='NAME',
                           help="Clear only this function's entries, leaving the rest "
                                'of the cache intact. "notebook" selects the '
                                'notebook statements.')
    sub_clear.add_argument('--entry', default=None, metavar='ID',
                           help='Clear one entry by id, as listed by '
                                '`cash inspect --function NAME`. Any unambiguous '
                                'prefix works. Takes precedence over --function.')
    sub_clear.add_argument('--tool', default=None, metavar='NAME',
                           help='Act on the per-user cache of the installed console '
                                'script NAME instead of the cache in use. On its own it '
                                'clears that whole cache; with --function or --entry, '
                                'just those entries.')
    sub_clear.add_argument('--force', action='store_true',
                           help='Clear a directory even though it holds no CACHE_VERSION '
                                'and no .entry files. Never clears the current directory '
                                'or one that contains it.')
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
