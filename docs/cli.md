# Command-line interface

Cash ships a `cash` CLI for the tasks that don't fit cleanly inside a notebook
cell — installing the global IPython autoload hook, inspecting on-disk cache
directories, clearing caches, and reporting which configuration is in effect.
Behaviour is derived directly from `src/cash/__main__.py`; this page is the
canonical reference.

`cash` and `python -m cash` are equivalent entry points (the script is declared
as `cash = "cash.__main__:main"` in `pyproject.toml`). Running `cash` with no
subcommand prints help and exits 0.

<!-- claim: cash/__main__.py:main @92b1b60e broad="the quick-reference table is a claim about the whole subcommand set" -->
## Quick reference

| Subcommand | Purpose | Destructive? |
|---|---|---|
| [`cash version`](#cash-version) | Print the installed cash version. | No |
| [`cash info`](#cash-info) | Show the effective merged configuration. | No |
| [`cash inspect [path] [--function NAME] [--tool NAME]`](#cash-inspect-path) | Summarise a cache directory or notebook; drill into one function. | No |
| [`cash clear [path] [--all] [--function NAME] [--entry ID] [--tool NAME]`](#cash-clear-path-all) | Delete a cache directory, one function's entries, or a single entry. | **Yes** — no confirmation prompt |
| [`cash autoload on`](#cash-autoload-on) | Install the IPython startup hook. | No (refuses to clobber by default) |
| [`cash autoload off`](#cash-autoload-off) | Remove the startup hook. | Yes (deletes one file) |

There are no global flags beyond `-h/--help`. `--version` is not a flag — it is
its own subcommand (`cash version`).

---

## Installing the autoload hook

This is the command most users reach for. It writes a single Python file into
your IPython startup directory so that every new kernel either has `cash`
pre-imported (`--mode available`) or has caching automatically enabled
(`--mode active`, the default).

The target path is:

```
<IPython dir>/profile_<profile>/startup/00-cash.py
```

`<IPython dir>` is resolved via `IPython.paths.get_ipython_dir()` when IPython
is importable, and falls back to `~/.ipython` otherwise. `<profile>` defaults
to `default`, giving `~/.ipython/profile_default/startup/00-cash.py` on most
machines.

### `cash autoload on`

Install the startup hook so cash is loaded automatically in every new
IPython/Jupyter kernel.

**Usage:** `cash autoload on [--mode {available,active}] [--profile NAME] [--force]`

**Arguments:**

- `--mode {available,active}` — *Optional, default `active`.*
    - `active` writes the **active** hook: `import cash` plus
      `get_ipython().run_line_magic("cash_on", "")`. Caching is on the moment
      the kernel boots; run `%cash_off` per session to opt out.
    - `available` writes the **available** hook: just `import cash`. You still
      run `%cash_on` (or use `@cash.cache`) per notebook, but the import is
      already done.
- `--profile NAME` — *Optional, default `default`.* IPython profile to install
  into. Use this if you keep parallel profiles (e.g. one for work, one for
  data-science experiments).
- `--force` — *Optional.* Overwrite any existing file at the target path, even
  if cash didn't write it. Without `--force`, cash refuses to clobber a
  non-cash file.

**Examples:**

```bash
cash autoload on                       # active mode (default): cash + %cash_on
cash autoload on --mode available      # lighter: just `import cash`
cash autoload on --profile work        # install into ~/.ipython/profile_work
cash autoload on --force               # replace whatever's at 00-cash.py
```

**Behaviour notes:**

- **Idempotent.** Re-running `cash autoload on` when the file already contains
  the exact body for the requested mode prints
  `Autoload already on (mode=<mode>): <path>` and exits 0 — no rewrite.
- **Safe by default.** If a different file already exists at the target path,
  cash refuses and exits 1 with the message
  `Refusing to overwrite existing file: <path>`. Pass `--force` to replace it,
  or run `cash autoload off` first if it was a cash hook from an earlier
  install.
- The written file starts with a `# cash-ipython-hook (managed by ...)`
  marker comment. `cash autoload off` uses this marker to decide whether the
  file is safe to delete.

### `cash autoload off`

Remove the startup hook written by `cash autoload on`.

**Usage:** `cash autoload off [--profile NAME] [--force]`

**Arguments:**

- `--profile NAME` — *Optional, default `default`.* IPython profile to clean
  up. Must match whatever you passed to `cash autoload on`.
- `--force` — *Optional.* Delete the target file **even if it lacks the
  `cash-ipython-hook` marker.** Without `--force`, cash refuses to remove a
  file it doesn't recognise as its own.

**Examples:**

```bash
cash autoload off                      # remove from the default profile
cash autoload off --profile work       # remove from a custom profile
cash autoload off --force              # delete whatever sits at 00-cash.py
```

**Behaviour notes:**

- If no file exists at the target path, cash prints
  `Autoload not installed at: <path>` and exits 0 — it's a no-op, not an
  error.
- The marker check is a substring search for `cash-ipython-hook` inside the
  file. Older cash hooks (or files you wrote by hand) without the marker will
  trip the safety check; use `--force` to override.

---

## Inspecting your environment

### `cash version`

Print the installed cash version.

**Usage:** `cash version`

**Examples:**

```bash
cash version
# cash <!-- docnum:version -->0.10.0<!-- /docnum -->
```

**Behaviour notes:**

- Imports `cash.__version__`. If the import fails (e.g. the package is broken
  or partially installed), prints `cash unknown` and exits 0.
- This is a subcommand, not a global flag. `cash --version` does not work.

### `cash info`

Print the effective merged configuration.

**Usage:** `cash info`

**Output fields:**

- `Backend` — the configured backend type (e.g. `file`, `memory`, `tiered`).
- `Cache dir` — the on-disk cache directory the file backend will use. When
  something other than the default set it, `Settings` says what.
- `Disabled` — present only when `disable` is on, with where it was set:
  every cached function is running uncached.
- `Max size` — the caps the two persistent tiers actually resolve to, not the
  configured value: `auto -- disk 16.4 GiB, RAM 4.0 GiB` when unset (the
  default), or the size in the unit you wrote it in plus the exact bytes
  (`2 GB (2,000,000,000 bytes) on disk, RAM <M>`) when `max_cache_size` is
  set. The RAM figure appears nowhere else, and a growing RSS is usually that
  cap doing its job rather than a leak.
- `Persist` — what actually decides disk persistence: the cost model
  (`0.1s compute floor, N% savings required`), or a conservative fallback when
  smart persistence is off.
- `Tiers` — present when the active config declares an explicit tier
  list; lists each tier's type in order, with the options it sets —
  `memory, file (default_ttl=3600s)`.
- `Config files` — every config file looked for and what happened: `read`,
  `not found`, `no [tool.cash] section`, or `could not be read` (a warning
  says why).
- `Settings` — every setting some layer set, its effective value, and the
  layer that won: a file path, `CASH_<NAME>`, or `Cash(...)`.
- `Source` — which layers contributed to the resolved config (e.g.
  `project:./pyproject.toml,env`, or `defaults` when nothing was set).
- `Tool caches` — present when installed console scripts have cached
  anything per user: one line per tool with its entry count, size and
  directory. An installed tool run from outside any project caches there,
  where the plain commands cannot see it; `--tool NAME` reaches it.

**Examples:**

```bash
cash info
# Cash v<!-- docnum:version -->0.10.0<!-- /docnum -->
#   Backend:    tiered
#   Cache dir:  /home/me/project/.cash
#   Max size:   auto -- disk 16.4 GiB, RAM 4.0 GiB
#   Persist:    cost model (0.1s compute floor, 20% savings required)
#   Config files:
#     user         /home/me/.config/cash/config.toml  (not found)
#     project      /home/me/project/pyproject.toml  (read)
#   Settings (where each came from):
#     compress = True                          /home/me/project/pyproject.toml
#     debug = True                             CASH_DEBUG
#   Source:     project:/home/me/project/pyproject.toml,env
```

**Behaviour notes:**

- A setting that appears in more than one layer is listed once, with the
  layer that won (see the precedence in
  [Configuration](getting-started/configuration.md)). A key
  that is not a setting, or a value cash could not use, is left out of the
  list and reported as a warning.

<!-- claim: cash/__main__.py:cmd_inspect @e3a7eb28, cash/__main__.py:_inspect_cache_dir @24d5dddf, cash/__main__.py:_inspect_notebook @06ba3efe -->
### `cash inspect [path] [--function NAME]` { #cash-inspect-path }

Summarise a cache directory, or report on a notebook and its sibling `.cash`
directory.

**Usage:** `cash inspect [path] [--function NAME]`

**Arguments:**

- `path` — *Optional.* One of:
    - **A `.ipynb` file.** Reads the notebook with `nbformat`, counts code
      and markdown cells, detects whether `%cash_on` or `%%cash` is used, and
      then inspects the sibling `.cash` directory next to the notebook (if
      any).
    - **A directory.** Treated as a cache directory; cash walks it
      recursively.
    - **Omitted.** Defaults to **the cache the library is using** — the same
      directory `cash info` reports, resolved through the same merge
      (defaults, user config, project config, `CASH_CACHE_DIR`), so it follows
      the [project anchor](getting-started/configuration.md#what-paths-are-relative-to)
      rather than your current directory. The path is printed above the table.
      If it does not exist, cash names it and exits 1.

**Examples:**

```bash
cash inspect                           # inspect the cache in use
cash inspect ./.cash                   # a specific directory
cash inspect ./notebooks/analysis.ipynb
cash inspect /tmp/some-cache-dir
```

- `--function NAME` — *Optional.* List one function's individual entries
  instead of the per-function overview: the id `cash clear --entry` takes
  (and the per-call debug line prints), what the entry saves, its size,
  `USES` (reads from disk — hits a running process serves from RAM are not
  counted), when it was last used and, for an entry written with a `ttl`,
  when it `EXPIRES`. An unambiguous trailing segment is
  enough: `--function work` finds `model.work`. `notebook` (or
  `statements`) selects the `(notebook statements)` group without its
  brackets. An ambiguous name prints the candidates and exits 1; an unknown
  one prints the functions that *are* cached.
- `--tool NAME` — *Optional.* Inspect the per-user cache of the installed
  console script `NAME` instead of the cache in use. `cash info` lists the
  names. Cannot be combined with a path.

  Each row says what the entry is **worth**, not just how big it is:

  ```
  ENTRY             SAVES       SIZE   USES   LAST USED   PRODUCES
  aaaaaaaaaaaa      12.5s     4.0 KB     3x   2 min ago   df, model
  bbbbbbbbbbbb       0.4s     1.0 KB     1x   2 min ago   scores
  ```

  `SAVES` is the recorded execution time of the function's own body (cash's
  key work is paid on a hit too, so it is not counted) — what you lose by deleting it —
  which together with `SIZE` and `USES` is the whole trade. `PRODUCES` names
  the variables a notebook statement produced, and is omitted when no entry
  in the group has any. An entry computed from files gets a `reads:` line
  under it naming them — the first three, then a count:

  ```
  ENTRY             SAVES       SIZE   USES   LAST USED
  94babd0e07b7       0.2s      898 B     0x   2s ago
        reads: /srv/etl/data/prices.csv
  ```

  The `ENTRY` id is the one `f.explain(...).entry_id` reports, so an
  explanation leads straight to the entry to drop.

**Output for a cache directory:**

A per-function table, sorted by size — because the question that sends people
here is "what is filling my disk, and what can I afford to drop?":

```
Cache directory: .cash
  Total size: 13.7 MB    Entries: 3    Functions: 2

  FUNCTION                      ENTRIES        SIZE   LAST USED
  model.heavy_field                   2     13.7 MB   2 min ago
  model.small_helper                  1       620 B   2 min ago
```

The grouping needs no extra bookkeeping: a decorator cache key is
`{module.qualname}:{state}:{dynamic}:{args}`, so the owning function is the
first segment of every key on disk. Notebook statements are keyed `stmt:<sha>`
and have no function to name, so they group under `(notebook statements)`.

The module part is the **defining file's name**, not `__main__`: a function in
`model.py` groups under `model.` whether you ran the script or imported it, so
the two agree. `__main__.` is what you see for code with no defining file — a
REPL, `python -c`, or a notebook kernel.

**Output for a notebook:**

- Code-cell and markdown-cell counts.
- `Uses cash: Yes/No` based on a textual scan for `%cash_on` or `%%cash`.
- The cache summary above if a sibling `.cash/` exists; otherwise
  `Cache: not found (no .cash directory)`.

**Behaviour notes:**

- `nbformat` is an optional dependency. If it isn't installed when you point
  `cash inspect` at a `.ipynb`, cash prints
  `nbformat not installed. Install with: pip install nbformat` and continues
  cleanly (exit 0).
- Each entry is a single `.entry` file whose metadata sits in a
  length-prefixed header, so the report reads the header and never touches the
  payload. A corrupted or unreadable entry is skipped (logged at debug level)
  rather than aborting the report — one bad file must not cost you the totals
  for everything else. When nothing is readable the table is replaced by
  `(no readable entries)`.

---

## Clearing caches

<!-- claim: cash/__main__.py:cmd_clear @9e608a46 -->
### `cash clear [path] [--all] [--function NAME]` { #cash-clear-path-all }

Delete a cache directory, or just one function's entries.

!!! warning "Destructive without confirmation"
    `cash clear` calls `shutil.rmtree()` immediately on the resolved
    directory. There is **no confirmation prompt** — running the command
    deletes the cache as soon as you press enter. Be sure of the target
    before you run it, especially in CI.

    What it will **not** delete is something that isn't a cache. Every
    directory it removes — named explicitly, resolved by `--all` or
    `--tool`, or found next to a notebook — must hold a `CACHE_VERSION` stamp
    or `.entry` files, or it is refused (exit 1). `--force` overrides that
    for a cache that lost its stamp. The current directory, and any
    directory containing it, is refused always, `--force` or not: an
    explicit path used to go straight to `rmtree`, and `cash clear .` in a
    project deleted the project's files.

**Usage:** `cash clear [path] [--all] [--function NAME]`

**Arguments:**

- `path` — *Optional.* One of:
    - **A directory.** Removed in full via `shutil.rmtree` — if it looks like
      a cache (see the warning above); otherwise refused unless `--force`.
    - **A `.ipynb` file.** Cash removes the sibling `.cash/` directory next
      to the notebook (if any). If there's no sibling cache, prints
      `No cache found for <path>` and exits 0.
    - **Anything else.** Prints `Not found: <path>` and exits 1.
- `--all` — *Optional.* Clear **the cache the library is using** — the same
  directory `cash info` reports, not `./.cash` unless that is where it
  resolves. Combining it with a `path` is **refused** (exit 2): "all of the
  cache" and "this one directory" are two different requests, and it used to
  accept both and silently clear the one you did not name. If the resolved
  directory doesn't exist, cash names it and exits 0. If it exists but holds
  no `CACHE_VERSION` and no `.entry` files, cash refuses to remove it and
  exits 1 — a mistyped `CASH_CACHE_DIR` cost nothing while the CLI ignored the
  variable, and costs a recursive delete now that it doesn't.
- `--function NAME` — *Optional.* Delete only that function's entries and
  leave the rest of the cache intact — the alternative to keeping a cache you
  cannot afford or deleting work you still want. Resolves names exactly as
  `cash inspect --function` does, including `notebook`. Takes precedence over
  `--all`. It clears that function and nothing that calls it: a cached
  function that calls another is keyed on the inner one's *code*, not on its
  result, so clearing the inner one leaves the outer one's entries serving.
  To recompute a caller, clear the caller (or both).
- `--force` — *Optional.* Clear a directory even though it holds no
  `CACHE_VERSION` and no `.entry` files. Never clears the current directory
  or one that contains it.
- `--entry ID` — *Optional.* Delete a single entry, using an id from
  `cash inspect --function NAME`. Any unambiguous prefix works, like a short
  commit hash; an ambiguous one lists the matches and deletes nothing. Takes
  precedence over `--function`.
- `--tool NAME` — *Optional.* Act on the per-user cache of the installed
  console script `NAME` instead of the cache in use. On its own it clears
  that tool's whole cache; with `--function` or `--entry`, only those
  entries. Cannot be combined with a path.

**Examples:**

```bash
cash clear --entry a1b2c3              # drop one entry
cash clear --function ray.build_grid   # drop one function, keep the rest
cash clear --function notebook         # drop the notebook statements only
cash clear --all                       # nuke the cache in use
cash clear ./.cash                     # same thing, explicit
cash clear ./notebooks/analysis.ipynb  # nuke the sibling .cash next to the notebook
cash clear /tmp/some-cache-dir         # nuke any directory
```

**Behaviour notes:**

- If none of `path`, `--all` or `--function` is supplied, cash prints the
  `cash clear` help and exits 2 without touching anything.
- `--all` together with a `path` also exits 2, naming both commands you might
  have meant.
- The no-op "nothing to clear" message paths (no resolved cache, no sibling
  cache) exit 0; they're treated as success, not failure.
<!-- claim: cash/backends/file_backend.py:FileBackend._check_format_version @8efef298, cash/backends/file_backend.py:FileBackend._entries_are_current_format @a4172379 -->
<!-- claim: cash/backends/tiered_backend.py:TieredBackend._drop_ram_if_cleared @751b655a, cash/__main__.py:_bump_generation @9e2ac2be -->
- **Clearing the cache of a process that is still running** reaches its memory
  too. A running process checks, at most once a second, whether its cache
  directory was cleared (`--all`, `--function`, `--entry`), and if so drops
  what it holds in RAM, so within about a second it stops serving pre-clear
  results. That includes a process that started with no cache at all and
  created the directory itself, which earlier versions took for "the directory
  is still new" and kept serving from memory. Whatever it
  writes after the clear goes into a freshly re-created directory, which it
  stamps with the format version, so the next process keeps those entries
  rather than discarding them.
- **A call that is computing while you clear its function** is not stopped: it
  finishes, and stores what it computed into the cleared cache, because the
  result is still right for the code and inputs it ran with. If you are
  clearing *because* that code or its data changed, stop the job first — or
  change the code or the file, which makes the entry miss anyway.
- A cache directory cash creates contains a `.gitignore` of `*`, so `git add .`
  in the project leaves it out — including when a clear removed it under a
  running process and that process re-creates it. A directory that already
  existed is left alone.
- `Nothing cleared: no cache at …` means the directory `cash info` reports for
  where you are standing has no cache. A script outside any project (no
  `pyproject.toml`, `setup.py`, `setup.cfg` or `.git` above it) caches beside
  itself, and a `cache_dir` that was changed leaves the old directory behind;
  pass either directory to `cash clear` explicitly.

---

<!-- claim: cash/__main__.py:cmd_autoload @528fa896, cash/__main__.py:cmd_version @700ebd0c, cash/__main__.py:cmd_info @c8efcbea -->
## Exit codes

| Code | When |
|---|---|
| `0` | The command succeeded, including no-op outcomes ("nothing to clear", "autoload not installed at ..."). |
| `1` | User-error refusals: `cash inspect` when the resolved cache directory does not exist and no path was given; `cash clear <missing-path>`; `cash clear` asked to remove a directory that is not a cash cache (without `--force`), or the current directory or one containing it; `cash autoload on` refusing to overwrite without `--force`; `cash autoload off` refusing to delete a non-cash file without `--force`. |
| `2` | `cash clear` with none of `path`, `--all` or `--function`, or with `--all` *and* a path — argparse's own "bad invocation" code. Nothing is touched. |
| traceback | Uncaught exceptions bubble up as Python tracebacks — `main()` does not wrap dispatch in `try/except`. If you see one, treat it as a bug and please file an issue. |
