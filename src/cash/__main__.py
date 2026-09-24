"""CLI for inspecting and managing notebook caches."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cash import __version__
from cash._location import per_user_cache_root
from cash.backends._base import effective_ttl
from cash.backends.adaptive_caps import adaptive_disk_cap_for, resolve_ram_cap
from cash.backends.cache_dir import DB_FILENAME, KEYS_DIRNAME, VERSION_FILENAME, entry_totals, is_cash_file
from cash.backends.entry_format import ENTRY_SUFFIX
from cash.backends.file_backend import FileBackend, StoredEntry
from cash.backends.persistence_policy import PersistencePolicy
from cash.config import (
    SIZE_FIELDS,
    TOML_MISSING,
    TOML_NOT_CASH,
    TOML_SECTION,
    config_provenance,
    format_size,
    get_config,
    human_bytes,
)

logger = logging.getLogger(__name__)


def get_version() -> str:
    return __version__


def resolved_cache_dir() -> str:
    """The cache directory the LIBRARY would use, for commands that act on it.

    Through ``get_config()``: one merge order for everyone -- defaults, user
    TOML, project TOML, environment -- and the same project anchor, so
    ``inspect`` and ``clear`` act on the cache the library wrote, never on a
    ``./.cash`` the user did not mean.
    """
    try:
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

    return str(per_user_cache_root() / name)


def notebook_cache_dir(notebook_path: str) -> str:
    """The cache directory a kernel for *notebook_path* uses.

    Jupyter starts the kernel in the notebook's directory, which is then its
    cwd and its project anchor: ``[tool.cash] cache_dir`` in a
    ``pyproject.toml`` above it applies, and so does ``CASH_CACHE_DIR``, as
    the kernel reads them.
    """
    nb_dir = Path(notebook_path).resolve().parent
    try:
        cache_dir = str(get_config(anchor=nb_dir).cache_dir)
    except Exception:  # noqa: BLE001 - a broken config must not break `clear`
        cache_dir = ".cash"
    return os.path.normpath(os.path.join(nb_dir, cache_dir))


def _target_dir(args: argparse.Namespace) -> str:
    """The directory a subcommand acts on when no path was given."""
    tool = getattr(args, "tool", None)
    return tool_cache_dir(tool) if tool else resolved_cache_dir()


def _sqlite_cache(cache_dir: str) -> tuple[int, int] | None:
    """``(entries, bytes)`` for a sqlite cache here, or ``None`` if there is none.

    A sqlite cache is one database file, so the entry-file count every other
    command uses would report "nothing here" while the cache works fine.
    """
    path = cache_dir if os.path.isfile(cache_dir) else os.path.join(cache_dir, DB_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        return int(rows[0]), os.path.getsize(path)
    except Exception:  # noqa: BLE001 - not a cash database, or unreadable
        return None


def _per_user_tool_caches() -> list[tuple[str, str, int, int]]:
    """``(tool, path, entries, bytes)`` for every per-user tool cache."""
    try:
        root = per_user_cache_root()
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except (OSError, RuntimeError):
        return []
    found = []
    for child in children:
        totals = entry_totals(str(child))
        if totals is not None:
            found.append((child.name, str(child), *totals))
    return found


def cmd_version(args: argparse.Namespace) -> None:
    """Show cash version."""
    print(f"cash {get_version()}")


def cmd_info(args: argparse.Namespace) -> None:
    """Show cash configuration."""

    config = get_config(config_path=getattr(args, "config", None))

    source, origins, files = config_provenance(config)

    print(f"Cash v{get_version()}")
    print(f"  Backend:    {config.backend}")
    print(f"  Cache dir:  {config.cache_dir}")
    # What it holds, next to where it is: the number a user asks for when
    # deciding whether to clear it.
    database = _sqlite_cache(config.cache_dir)
    held = database if database is not None else entry_totals(config.cache_dir)
    if database is not None:
        print(f"  Holds:      {database[0]} entries, {human_bytes(database[1])} (one sqlite database)")
    elif held is None:
        print("  Holds:      nothing yet (no cache written here)")
    else:
        print(f"  Holds:      {held[0]} entries, {human_bytes(held[1])}")
    if held:
        # A total alone is a dead end: the next question is always which
        # entries, and whether they earn their space.
        print("              `cash inspect` lists them, with what each one saves")
    if config.disable:
        print(f"  Disabled:   yes -- every cached function runs uncached ({origins.get('disable', 'disable = true')})")
    # Resolved, not just configured: a user asking what their cache may hold
    # needs the two numbers "auto" resolves to, and the RAM one appears
    # nowhere else (a growing RSS is easily read as a leak).
    if config.max_cache_size is None:
        # Sized the way the BACKEND sizes it: from free space plus what the
        # cache already holds. Free space alone (`resolve_disk_cap`) excludes
        # the cache's own bytes and would show a cap lower than the one
        # enforced, next to a "Holds" that seems to exceed it.
        own = held[1] if held is not None else 0
        disk = human_bytes(adaptive_disk_cap_for(config.cache_dir, own))
        print(f"  Max size:   auto -- disk {disk}, RAM {human_bytes(resolve_ram_cap())}")
    else:
        print(
            f"  Max size:   {format_size(config.max_cache_size)} "
            f"({config.max_cache_size:,} bytes) on disk, "
            f"RAM {human_bytes(resolve_ram_cap())}"
        )
    print(f"  Persist:    {PersistencePolicy.from_config(config).describe()}")
    if config.tiers:
        print(f"  Tiers:      {', '.join(_tier_text(t) for t in config.tiers)}")
    # Where this run looked, and what each layer set: the Source line names
    # the layers, not which file, nor which setting came from where.
    outcome = {
        TOML_SECTION: "read",
        TOML_MISSING: "not found",
        TOML_NOT_CASH: TOML_NOT_CASH,
    }
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
    print(f"  Source:     {source}")
    # Installed tools run from outside a project cache per user, per tool --
    # somewhere this command cannot reach by default, because it is a
    # different console script. Say where they are, so an operator does not
    # have to know the platform's cache root to find them.
    tools = _per_user_tool_caches()
    if tools:
        print("  Tool caches (reach one with --tool NAME):")
        for name, path, entries, size in tools:
            print(f"    {name:<20} {entries:>5} entries  {human_bytes(size):>10}  {path}")


def _tier_text(tier) -> str:
    """``file (default_ttl=5s)``: a tier's type and every option it sets,
    since a ``default_ttl`` decides when entries expire."""

    parts = []
    for f in dataclasses.fields(tier):
        value = getattr(tier, f.name, None)
        if f.name == "type" or value is None:
            continue
        if "password" in f.name:
            value = "***"
        elif f.name == "default_ttl" and isinstance(value, int):
            value = f"{value}s"
        elif f.name == "max_size_bytes" and isinstance(value, int):
            value = format_size(value)
        parts.append(f"{f.name}={value}")
    return f"{tier.type} ({', '.join(parts)})" if parts else tier.type


def _setting_text(config, key: str) -> str:
    """One setting's effective value, as `cash info` prints it."""

    value = getattr(config, key, None)
    if key == "tiers":
        return ", ".join(_tier_text(t) for t in value) or "[]"
    if key == "redis_password" and value:
        return "***"
    if key in SIZE_FIELDS and isinstance(value, int):
        return format_size(value)
    return repr(value)


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
NOTEBOOK_GROUP = "(notebook statements)"
_NOTEBOOK_ALIASES = frozenset({"notebook", "notebooks", "statements", "notebook statements"})


def _function_of(key: str, metadata: dict | None = None) -> str:
    """The function a cache key belongs to.

    Decorator keys are ``{module.qualname}:{state}:{dynamic}:{args}``, so the
    owner is simply the first segment -- no extra metadata needed to group a
    cache directory by function. Notebook statements are keyed ``stmt:<sha>``
    and have no function to name, so they are collected under one heading
    rather than reported as a function called "stmt".
    """
    if key.startswith("stmt:"):
        return "(notebook statements)"
    # An intercepted call's key is `call:<sha>`: its function is recorded in
    # the entry instead. Older entries without it still read "call".
    if key.startswith("call:") and metadata and metadata.get("function"):
        return str(metadata["function"])
    return key.split(":", 1)[0] if ":" in key else "(unknown)"


def _tier_default_ttl() -> int | None:
    """The ``default_ttl`` of the first configured tier that has one, now."""
    try:
        for tier in get_config().tiers or ():
            if getattr(tier, "default_ttl", None) is not None:
                return int(tier.default_ttl)
    except Exception:  # a listing must not fail over config
        logger.debug("Could not read the tiers' default_ttl", exc_info=True)
    return None


def _store(cache_dir: str | os.PathLike) -> FileBackend:
    """The cache directory, opened the way the library opens it.

    Every change the CLI makes goes through it, so the backend's own
    bookkeeping -- the byte total, the stamp -- is kept as a write would keep
    it. No flusher: the command is over before one would run.
    """
    return FileBackend(str(cache_dir), flush_interval=0)


def _scan_entries(cache_dir: str | os.PathLike) -> list[_Entry]:
    """Every readable entry in *cache_dir*, from its metadata alone.

    The payload is never read, so listing a cache full of large frames costs
    the same as listing one full of small ones.
    """
    tier_default = _tier_default_ttl()
    return [_entry_of(stored, tier_default) for stored in _store(cache_dir).entries()]


def _entry_of(stored: StoredEntry, tier_default: int | None) -> _Entry:
    metadata = stored.metadata
    ttl = effective_ttl(metadata, tier_default)
    return _Entry(
        stem=stored.id,
        function=_function_of(stored.key, metadata),
        key=stored.key,
        size=stored.size,
        mtime=stored.mtime,
        saves=float(metadata.get("execution_time") or 0.0),
        uses=int(metadata.get("access_count") or 0),
        outputs=tuple(str(o) for o in metadata.get("outputs") or ()),
        reads=tuple(str(p) for p in (metadata.get("auto_file_deps") or {})),
        expires=None if ttl is None else float(metadata.get("created_at") or stored.mtime) + float(ttl),
    )


def _drop(cache_dir: str | os.PathLike, entries: list[_Entry]) -> None:
    """Delete *entries* through the backend, then tell running processes."""
    if not entries:
        return
    store = _store(cache_dir)
    for entry in entries:
        store.delete(entry.key)
        name = f"{entry.stem}{ENTRY_SUFFIX}"
        if os.path.exists(os.path.join(store.cache_dir, name)):
            # Antivirus or another process holding it. Partial progress still
            # frees space, and saying so beats a traceback part way through.
            print(f"  could not remove {name}")
    store.bump_generation()


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
    for limit, divisor, unit in ((60, 1, "s"), (3600, 60, "min"), (86400, 3600, "h")):
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
    matches = [n for n in names if n.rsplit(".", 1)[-1] == wanted or n.endswith("." + wanted)]
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

    if target and os.path.isfile(target) and target.endswith(".ipynb"):
        if only_function:
            print("--function applies to a cache directory, not a notebook.")
            sys.exit(2)
        _inspect_notebook(target)
        return

    if target and not os.path.isdir(target):
        # Not silently replaced by the configured cache: a mistyped path must
        # not print a plausible report about an unrelated cache, and `cash
        # clear` refuses the same input.
        print(f"Not found: {target}")
        print(
            "Pass a cache directory or a notebook, or leave it out to inspect the cache `cash info` reports for here."
        )
        sys.exit(1)
    cache_dir = target if (target and os.path.isdir(target)) else _target_dir(args)
    if not os.path.isdir(cache_dir):
        print(f"No cache found at {os.path.abspath(cache_dir)}.")
        print("Specify a notebook or cache directory, or set CASH_CACHE_DIR.")
        sys.exit(1)
    # The directory is printed once, absolute, by `_inspect_cache_dir`.
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
    except Exception as e:  # noqa: BLE001 - any unreadable notebook is reported, not raised
        print(f"  Error reading notebook: {e}")
        return

    # Count cells
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    print(f"  Code cells: {len(code_cells)}")
    print(f"  Markdown cells: {len(md_cells)}")

    # Check if %cash_on is used
    uses_cash = any("%cash_on" in c.source for c in code_cells)
    print(f"  Uses cash: {'Yes' if uses_cash else 'No'}")

    cache_dir = notebook_cache_dir(notebook_path)
    if os.path.isdir(cache_dir):
        _inspect_cache_dir(cache_dir)
    else:
        print(f"  Cache: not found (no directory at {cache_dir})")


def _inspect_cache_dir(cache_dir: str, only_function: str | None = None) -> None:
    """Inspect a cache directory.

    The default view is a per-function table sorted by SIZE, because the
    question that sends anyone here is "what is filling my disk, and what can
    I afford to drop?".
    """
    cache_path = Path(cache_dir)
    total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
    entries = _scan_entries(cache_path)

    print(f"Cache directory: {cache_path.resolve()}")

    database = _sqlite_cache(cache_dir)
    if database is not None and not entries:
        # A sqlite cache is one file; the per-function table below reads entry
        # files and would report an empty cache over a working one.
        print(
            f"  Total size: {human_bytes(database[1])}    Entries: {database[0]}"
            f"    (one sqlite database: {DB_FILENAME})"
        )
        print("\n  Per-function detail is not available for the sqlite backend.")
        return

    if only_function is not None:
        resolved = _resolve_function(entries, only_function)
        if resolved is None:
            sys.exit(1)
        owned = sorted((e for e in entries if e.function == resolved), key=lambda e: e.size, reverse=True)
        owned_size = sum(e.size for e in owned)
        noun = "entry" if len(owned) == 1 else "entries"
        print(f"  {resolved} - {len(owned)} {noun}, {human_bytes(owned_size)}\n")

        # SAVES is the column a delete decision actually turns on: bytes say
        # what you get back, seconds say what it costs you to lose. The old
        # view showed an opaque id, a size and an age -- enough to see that
        # entries exist, not enough to choose between them.
        shows_outputs = any(e.outputs for e in owned)
        # EXPIRES: an entry with a ttl stops being served at a time the table
        # could not show, so an expired entry looked like a live one.
        header = f"  {'ENTRY':<14}{'SAVES':>9}{'SIZE':>11}{'USES':>7}   {'LAST USED':<12}{'EXPIRES':<12}"
        print((header + "PRODUCES") if shows_outputs else header.rstrip())
        for entry in owned:
            saves = f"{entry.saves:.1f}s" if entry.saves else "-"
            produces = ", ".join(entry.outputs) if shows_outputs else ""
            row = (
                f"  {entry.stem[:12]:<14}{saves:>9}"
                f"{human_bytes(entry.size):>11}{str(entry.uses) + 'x':>7}"
                f"   {_age(entry.mtime):<12}{_expires(entry.expires):<12}{produces}"
            )
            print(row.rstrip())
            if entry.reads:
                shown = ", ".join(entry.reads[:3])
                more = f" and {len(entry.reads) - 3} more" if len(entry.reads) > 3 else ""
                print(f"      reads: {shown}{more}")
        print("\n  cash clear --entry ID   to drop one of these (any unambiguous prefix)")
        return

    functions: dict[str, list[_Entry]] = {}
    for entry in entries:
        functions.setdefault(entry.function, []).append(entry)

    print(f"  Total size: {human_bytes(total_size)}    Entries: {len(entries)}    Functions: {len(functions)}")

    if not functions:
        print("\n  (no readable entries)")
        return

    ranked = sorted(functions.items(), key=lambda kv: sum(e.size for e in kv[1]), reverse=True)
    print(f"\n  {'FUNCTION':<40}{'ENTRIES':>9}{'SIZE':>12}   LAST USED")
    for name, owned in ranked:
        newest = max(e.mtime for e in owned)
        print(f"  {name[:40]:<40}{len(owned):>9}{human_bytes(sum(e.size for e in owned)):>12}   {_age(newest)}")
    print("\n  cash inspect --function NAME   to list one function's entries")
    print("  cash clear   --function NAME   to drop them")
    if NOTEBOOK_GROUP in functions:
        # The heading reads as prose in the table; say plainly that it is
        # addressable without having to type the brackets.
        print("  --function notebook            for the notebook statements")


def _clear_function(cache_dir: str, wanted: str) -> None:
    """Delete every entry belonging to one cached function.

    For someone short on disk, the alternative to keeping a cache they cannot
    afford or deleting work they still want.
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
    _drop(cache_path, owned)
    noun = "entry" if len(owned) == 1 else "entries"
    print(f"Cleared {len(owned)} {noun} for {resolved} ({human_bytes(sum(e.size for e in owned))} freed)")


def _clear_entry(cache_dir: str, wanted: str) -> None:
    """Delete a single cache entry by id."""
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        print(f"No cache directory at {cache_dir}")
        sys.exit(1)
    entry = _resolve_entry(_scan_entries(cache_path), wanted)
    if entry is None:
        sys.exit(1)
    _drop(cache_path, [entry])
    print(f"Cleared entry {entry.stem[:12]} from {entry.function} ({human_bytes(entry.size)} freed)")


def _clear_expired(cache_dir: str) -> None:
    """Delete the entries whose ttl has run out, and say what that freed.

    An expired entry is never served, but it stays on disk until something
    writes over it, so lowering a tier's ``default_ttl`` alone frees nothing.
    """
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        print(f"No cache directory at {cache_dir}")
        sys.exit(1)
    now = time.time()
    expired = [e for e in _scan_entries(cache_path) if e.expires is not None and e.expires <= now]
    _drop(cache_path, expired)
    print(
        f"Cleared {len(expired)} expired "
        f"entr{'y' if len(expired) == 1 else 'ies'} from {cache_path} "
        f"({human_bytes(sum(e.size for e in expired))} freed)"
    )


def _looks_like_a_cache(cache_dir: str) -> bool:
    """Does this directory hold a cash cache, or something else entirely?

    ``clear --all`` deletes a directory the user did not type -- whatever the
    config resolved to -- so a mistyped ``CASH_CACHE_DIR`` is a reason to look
    before recursively removing.
    """
    if os.path.exists(os.path.join(cache_dir, VERSION_FILENAME)):
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
    a notebook's sibling ``.cash`` -- so ``cash clear .`` in a project cannot
    delete the project. "Destructive without confirmation" covers deleting a
    cache you meant to delete, never something that is not a cache at all.
    """
    resolved = os.path.abspath(cache_dir)
    if _contains_cwd(resolved):
        # Never, even with --force: nobody means to delete the directory they
        # are standing in or anything above it, and on Windows the removal
        # cannot even complete -- it deletes the contents, then fails.
        print(
            f"Refusing to clear {resolved}: it is the current directory or "
            f"contains it. Change to another directory first."
        )
        sys.exit(1)
    if not force and not _looks_like_a_cache(resolved):
        print(
            f"Refusing to clear {resolved}: it does not look like a cash "
            f"cache (no {VERSION_FILENAME} and no {ENTRY_SUFFIX} files)."
        )
        print(
            "Check the path, CASH_CACHE_DIR and [tool.cash] cache_dir. If it "
            "really is a cache that lost its marker, clear it with --force."
        )
        sys.exit(1)
    if not force:
        # Looking like a cache is not enough: cash writes its stamp into
        # whatever directory it is pointed at, so a `cache_dir` beside the
        # user's data made this a recursive delete of that data -- a project
        # with `cache_dir = "../shared_data"` lost `shared_data/precious.csv`
        # to `cash clear --all`, exit 0. Nothing cash did not write is removed.
        foreign = _not_cash_files(resolved)
        if foreign:
            shown = ", ".join(foreign[:3]) + (", ..." if len(foreign) > 3 else "")
            print(f"Refusing to clear {resolved}: it holds files cash did not write ({shown}).")
            print(
                "Point cache_dir at a directory of its own, or clear it "
                "anyway with --force (which removes everything in it)."
            )
            sys.exit(1)
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        # On Windows a file another process holds open cannot be deleted, so
        # clearing the cache of a notebook whose kernel is still running
        # stops at its `cache.db` (WinError 32). Say which file and what to
        # do, not a traceback.
        culprit = exc.filename or resolved
        reason = exc.strerror or exc
        print(
            f"Could not clear {resolved}: cannot delete {culprit} ({reason}). "
            f"Close the notebook or stop the kernel using this cache first, then run cash clear again."
        )
        sys.exit(1)
    print(f"Cleared: {resolved}")


def _not_cash_files(cache_dir: str) -> list[str]:
    """Names in *cache_dir* that cash did not write, shallowest first."""
    found: list[str] = []
    for root, dirs, files in os.walk(cache_dir):
        rel_root = os.path.relpath(root, cache_dir)
        for name in files:
            rel = name if root == cache_dir else os.path.join(rel_root, name)
            if not is_cash_file(rel):
                found.append(rel)
        if not files and not dirs and root != cache_dir and rel_root != KEYS_DIRNAME:
            # An empty directory is the user's too, unless it is one of cash's.
            found.append(rel_root + os.sep)
        if len(found) > 32:
            return found
    return found


def _refuse_entry_flags_on_sqlite(cache_dir: str, flag: str) -> None:
    """Exit with a message when *cache_dir* is a SQLite cache.

    The entry-level flags work on a file cache's entry files. A SQLite cache
    has none, so they reported "Cleared 0" or "No cached function matches"
    over a cache that was full.
    """
    if _sqlite_cache(cache_dir) is None or entry_totals(cache_dir) not in (None, (0, 0)):
        return
    print(
        f"cash clear {flag} works on a file cache's entries, and {os.path.abspath(cache_dir)} "
        f"holds a sqlite database ({DB_FILENAME}) instead."
    )
    print(f"  To clear it whole: cash clear {cache_dir}")
    sys.exit(2)


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear cache."""
    if args.all and args.path:
        # "everything" and "this one directory" are two different requests;
        # dropping either would clear a cache the user had not named.
        print("cash clear: --all and a path are mutually exclusive.")
        print(f"  To clear that directory:   cash clear {args.path}")
        print(f"  To clear the cache in use: cash clear --all   ({os.path.abspath(resolved_cache_dir())})")
        sys.exit(2)

    tool = getattr(args, "tool", None)
    if tool and args.path:
        print("cash clear: --tool and a path are mutually exclusive.")
        sys.exit(2)

    only_entry = getattr(args, "entry", None)
    only_function = getattr(args, "function", None)
    if getattr(args, "expired", False):
        if only_entry or only_function:
            # --function would otherwise win and delete the live entries too.
            print(
                "cash clear: --expired clears across the whole cache; it cannot be combined with --function or --entry."
            )
            sys.exit(2)
        target = args.path if (args.path and os.path.isdir(args.path)) else _target_dir(args)
        _refuse_entry_flags_on_sqlite(target, "--expired")
        _clear_expired(target)
        return

    if only_entry:
        target = args.path if (args.path and os.path.isdir(args.path)) else _target_dir(args)
        _refuse_entry_flags_on_sqlite(target, "--entry")
        _clear_entry(target, only_entry)
        return

    if only_function:
        target = args.path if (args.path and os.path.isdir(args.path)) else _target_dir(args)
        _refuse_entry_flags_on_sqlite(target, "--function")
        _clear_function(target, only_function)
        return

    force = bool(getattr(args, "force", False))
    if args.all or tool:
        cache_dir = _target_dir(args)
        if os.path.isdir(cache_dir):
            _rmtree_cache(cache_dir, force=force)
        else:
            # "Nothing here" is true and not enough: a running program's cache
            # may be elsewhere. Say which directory was looked at, and the two
            # ways that happens.
            print(
                f"Nothing cleared: no cache at {os.path.abspath(cache_dir)}, "
                f"the directory `cash info` reports for here."
            )
            print(
                "  A script outside any project (no pyproject.toml, setup.py, "
                "setup.cfg or .git above it) caches in a .cash beside the script: "
                "cash clear <script dir>/.cash"
            )
            print("  A cache_dir that was changed leaves the old directory behind: cash clear <old path>")
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
    elif os.path.isfile(target) and target.endswith(".ipynb"):
        cache_dir = notebook_cache_dir(target)
        if os.path.isdir(cache_dir):
            _rmtree_cache(cache_dir, force=force)
        else:
            print(f"No cache found for {target} (looked in {cache_dir})")
    else:
        print(f"Not found: {target}")
        sys.exit(1)


HOOK_FILENAME = "00-cash.py"
HOOK_MARKER = "# cash-ipython-hook (managed by `cash autoload`)"

HOOK_BODY_AVAILABLE = f"""{HOOK_MARKER}
# Mode: available — `%cash_on` (and `cash.cache`) ready to use in every session.
# Remove with `cash autoload off`.
import cash  # auto-registers cash IPython magics
"""

HOOK_BODY_ACTIVE = f"""{HOOK_MARKER}
# Mode: active — caching is enabled automatically in every IPython/Jupyter
# session.  Run %cash_off in any session you want to opt out of, or remove
# this file with `cash autoload off`.
import cash  # auto-registers cash IPython magics

try:
    _ip = get_ipython()  # noqa: F821 - IPython injects this at startup
except NameError:
    _ip = None
if _ip is not None:
    _ip.run_line_magic("cash_on", "")
"""


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
        prog="cash",
        description="A Python cache that re-runs only what changed.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    sub_version = subparsers.add_parser("version", help="Show cash version")
    sub_version.set_defaults(func=cmd_version)

    # info
    sub_info = subparsers.add_parser("info", help="Show cash configuration")
    sub_info.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Resolve as a program that passes Cash(config_path=PATH) would -- a packaged tool's own config file.",
    )
    sub_info.set_defaults(func=cmd_info)

    # inspect
    sub_inspect = subparsers.add_parser("inspect", help="Inspect cache for a notebook or directory")
    sub_inspect.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Notebook (.ipynb) or cache directory path. Defaults to the cache the library is using.",
    )
    sub_inspect.add_argument(
        "--function",
        default=None,
        metavar="NAME",
        help="List one function's entries, with what each one saves. An unambiguous "
        'trailing segment is enough ("work" finds "__main__.work").',
    )
    sub_inspect.add_argument(
        "--tool",
        default=None,
        metavar="NAME",
        help="Inspect the per-user cache of the installed console script NAME (listed by `cash info`).",
    )
    sub_inspect.set_defaults(func=cmd_inspect)

    # clear
    sub_clear = subparsers.add_parser("clear", help="Clear cache")
    sub_clear.add_argument("path", nargs="?", default=None, help="Notebook or cache directory to clear")
    sub_clear.add_argument(
        "--all",
        action="store_true",
        help="Clear the cache the library is using -- the same "
        "directory `cash info` reports. Cannot be combined "
        "with a path.",
    )
    sub_clear.add_argument(
        "--function",
        default=None,
        metavar="NAME",
        help="Clear only this function's entries, leaving the rest "
        'of the cache intact. "notebook" selects the '
        "notebook statements.",
    )
    sub_clear.add_argument(
        "--entry",
        default=None,
        metavar="ID",
        help="Clear one entry by id, as listed by "
        "`cash inspect --function NAME`. Any unambiguous "
        "prefix works. Takes precedence over --function.",
    )
    sub_clear.add_argument(
        "--expired",
        action="store_true",
        help="Clear only the entries whose ttl has run out -- by "
        "the rule reads apply, a lowered default_ttl included. "
        "Frees the disk they hold; nothing else is touched.",
    )
    sub_clear.add_argument(
        "--tool",
        default=None,
        metavar="NAME",
        help="Act on the per-user cache of the installed console "
        "script NAME instead of the cache in use. On its own it "
        "clears that whole cache; with --function or --entry, "
        "just those entries.",
    )
    sub_clear.add_argument(
        "--force",
        action="store_true",
        help="Clear a directory even though it holds no CACHE_VERSION "
        "and no .entry files. Never clears the current directory "
        "or one that contains it.",
    )
    sub_clear.set_defaults(func=cmd_clear, clear_parser=sub_clear)

    # autoload on|off
    sub_autoload = subparsers.add_parser(
        "autoload",
        help="Toggle whether cash auto-loads in every new IPython/Jupyter kernel",
        description=(
            "Install or remove an IPython startup hook so cash is loaded (and optionally "
            "enabled) automatically in every new kernel - no `import cash` needed per notebook."
        ),
    )
    sub_autoload.add_argument(
        "state",
        choices=["on", "off"],
        help="on: install the startup hook. off: remove it.",
    )
    sub_autoload.add_argument(
        "--mode",
        choices=["available", "active"],
        default="active",
        help="(on only) available: just `import cash`. active (default): also run %%cash_on so caching is on by default.",
    )
    sub_autoload.add_argument(
        "--profile",
        default="default",
        help='IPython profile to target (default: "default")',
    )
    sub_autoload.add_argument(
        "--force",
        action="store_true",
        help="(on) overwrite a different file at this path. (off) remove a file lacking the cash marker.",
    )
    sub_autoload.set_defaults(func=cmd_autoload)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
