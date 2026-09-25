# Command-line interface

!!! info "Applies to: both paths"
    The `cash` command: see your configuration, inspect a cache and clear it. `cash autoload` is for notebook users only.

`cash` and `python -m cash` are the same command. With no subcommand it prints
help and exits 0.

<!-- claim: cash/__main__.py:main @fc218b1b broad="the quick-reference table is a claim about the whole subcommand set" -->
| Command | What it does | Deletes? |
|---|---|---|
| [`cash version`](#cash-version) | Print the installed version. | No |
| [`cash info`](#cash-info) | Show the configuration in effect and where each setting came from. | No |
| [`cash inspect`](#cash-inspect-path) | Summarise a cache: what each function stores, and what it is worth. | No |
| [`cash clear`](#cash-clear-path-all) | Delete a cache, one function's entries, one entry, or the expired entries. | **Yes**, without asking |
| [`cash autoload`](#cash-autoload) | Load cash in every new IPython kernel (notebook only). | Only its own hook file |

Every subcommand takes `-h` / `--help`. There is no `--version` flag; use
`cash version`.

## `cash version`

**Usage:** `cash version`

```bash
cash version
# cash <!-- docnum:version -->0.11.0<!-- /docnum -->
```

## `cash info`

**Usage:** `cash info [--config PATH]`

Prints the configuration in effect, the cache directory and its size, and
where each setting came from.

- `--config PATH`: resolve the configuration as `Cash(config_path=PATH)`
  would. Use it to check the config file a packaged tool ships, without running
  the tool.

```bash
cash info
# Cash v<!-- docnum:version -->0.11.0<!-- /docnum -->
#   Backend:    tiered
#   Cache dir:  /home/me/project/.cash
#   Holds:      412 entries, 1.3 GiB
#               `cash inspect` lists them, with what each one saves
#   Max size:   auto -- disk 16.4 GiB, RAM 4.0 GiB
#   Persist:    cost model (0.1s compute floor, 20% savings required; notebook statements: 0.01s store floor, 0.05s restore budget)
#   Config files:
#     user         /home/me/.config/cash/config.toml  (not found)
#     project      /home/me/project/pyproject.toml  (read)
#   Settings (where each came from):
#     compress = True                          /home/me/project/pyproject.toml
#     debug = True                             CASH_DEBUG
#   Source:     project:/home/me/project/pyproject.toml,env
```

<!-- claim: cash/__main__.py:local_cache_dir @909369d0 -->
- `Cache dir` is where the entries are: the first file or sqlite tier's own
  `cache_dir` (or `db_path`) when a tier stack sets one, else `cache_dir`.
  `cash inspect` and `cash clear` act on the same directory.
- `Max size` shows the caps the disk and RAM tiers actually use, including
  the automatic ones.
- `Config files` lists every file looked for: `read`, `not found`,
  `no [tool.cash] section`, or `could not be read`.
- `Settings` lists each setting some layer set, with the layer that won
  ([precedence](getting-started/configuration.md)). A key that is not a
  setting, or a value cash cannot use, is reported as a warning instead.
- `Disabled` appears when `disable` is on. `Tiers` appears when you declare a
  tier stack. `Tool caches` appears when installed command-line tools have
  per-user caches; `--tool NAME` reaches one.

<!-- claim: cash/__main__.py:cmd_inspect @0975006f, cash/__main__.py:_inspect_cache_dir @79ce6b6f, cash/__main__.py:_inspect_notebook @d58d3a2b -->
## `cash inspect` { #cash-inspect-path }

**Usage:** `cash inspect [--function NAME] [--tool NAME] [path]`

Summarises a cache directory, one function per row, largest first.

- `path`: a cache directory, or a `.ipynb` file (reports its cells, whether it
  uses `%cash_on`, and its cache). Without a path, inspects the cache the
  library is using, the one `cash info` reports.
- `--function NAME`: list one function's entries instead. A unique trailing
  part of the name is enough (`work` finds `model.work`); `notebook` selects
  the notebook statements. An ambiguous name lists the candidates.
- `--tool NAME`: inspect the per-user cache of the installed command-line tool
  `NAME` (listed by `cash info`). Cannot be combined with a path.

```bash
cash inspect                           # the cache in use
cash inspect ./.cash                   # a specific directory
cash inspect ./notebooks/analysis.ipynb
cash inspect --function model.heavy_field
```

```
Cache directory: .cash
  Total size: 13.7 MiB    Entries: 3    Functions: 2

  FUNCTION                      ENTRIES        SIZE   LAST USED
  model.heavy_field                   2    13.7 MiB   2 min ago
  model.small_helper                  1       620 B   2 min ago
```

A script's functions are named after the file that defines them (`model.`),
whether you ran it or imported it. Notebook statements group under
`(notebook statements)`.

With `--function`, each row shows what the entry is worth:

```
ENTRY             SAVES       SIZE   USES   LAST USED   PRODUCES
aaaaaaaaaaaa      12.5s    4.0 KiB     3x   2 min ago   df, model
      reads: /srv/etl/data/prices.csv
```

- `ENTRY` is the id `cash clear --entry` takes, and the same id
  `f.explain(...).entry_id` reports.
- `SAVES` is how long the body took to compute: what deleting the entry costs.
- `USES` counts reads from disk; hits a running process serves from RAM are
  not counted.
- `PRODUCES` names the variables a notebook statement produced. `EXPIRES`
  shows when an entry with a `ttl` runs out. `reads:` lists the files an entry
  depends on.

A notebook's cache is the one its kernel uses: `.cash` beside the notebook,
unless a `pyproject.toml` above it sets `[tool.cash] cache_dir` or
`CASH_CACHE_DIR` is set. If there is none, the report names where it looked.

Unreadable entries are skipped. Inspecting a notebook needs `nbformat`; without
it, cash says how to install it and exits 0.

<!-- claim: cash/__main__.py:cmd_clear @a08b9044 -->
## `cash clear` { #cash-clear-path-all }

**Usage:** `cash clear [--all] [--function NAME] [--entry ID] [--expired] [--tool NAME] [--force] [path]`

Deletes cache data immediately. There is no confirmation prompt.

- `path`: a cache directory to delete in full, or a `.ipynb` file whose whole
  cache directory is deleted. If it has none, cash prints where it looked and
  exits 0.
- `--all`: delete the cache the library is using, the one `cash info`
  reports. Cannot be combined with a path.
- `--function NAME`: delete one function's entries and keep the rest. Names
  resolve as in `cash inspect --function`, including `notebook`. It does not
  clear callers: a cached function that calls another is keyed on the inner
  function's code, not its result, so clear the caller too if you want it
  recomputed.
- `--entry ID`: delete one entry, by an id or any unique prefix of it from
  `cash inspect --function`. Takes precedence over `--function`.
- `--expired`: delete only entries whose `ttl` has run out, and report the disk
  freed. Expired entries are never served but stay on disk until removed.
  Cannot be combined with `--function` or `--entry`.
- `--tool NAME`: act on the per-user cache of the installed tool `NAME`. Alone
  it clears that whole cache; with `--function` or `--entry`, only those
  entries. Cannot be combined with a path.
- `--force`: clear a directory that does not look like a cash cache.

`--function`, `--entry` and `--expired` work on a file cache's entries. On a
SQLite cache (one `cache.db` file) they are refused with exit code 2; clear it
whole with `cash clear <dir>`.

```bash
cash clear --expired                   # free what will never be served again
cash clear --entry a1b2c3              # drop one entry
cash clear --function ray.build_grid   # drop one function, keep the rest
cash clear --function notebook         # drop the notebook statements only
cash clear --all                       # delete the cache in use
cash clear ./notebooks/analysis.ipynb  # delete that notebook's cache
```

<!-- claim: cash/backends/cache_dir.py:CacheDirStamp.check @c89cf812, cash/backends/cache_dir.py:CacheDirStamp._entries_are_current @853438c9 -->
**Safety rules.**

- Cash deletes a directory only if it holds cash's `CACHE_VERSION` stamp or
  `.entry` files, and nothing cash did not write. Otherwise it refuses and
  names what it found. `--force` overrides this.
- It never deletes the current directory or one that contains it, even with
  `--force`.

<!-- claim: cash/backends/clear_watch.py:ClearWatcher.cleared @04551b75, cash/backends/file_backend.py:FileBackend.bump_generation @9c482978 -->
**Clearing under a running process.** A running process notices within about a
second that its cache was cleared and drops what it holds in RAM. A call that
is computing while you clear its function still finishes and stores its
result; stop the job first if you are clearing because the code or data
changed.

<!-- claim: cash/__main__.py:_rmtree_cache @fec6529c, cash/__main__.py:_remove_markers_last @e10b3bc7 -->
**A cache a notebook has open.** On Windows a file that a running kernel holds
open cannot be deleted. `cash clear` then stops, names the file, and exits 1;
close the notebook or stop its kernel, then run the same command again. What
is left still counts as a cash cache, so it needs no `--force`.

`Nothing cleared: no cache at …` means the directory `cash info` reports has
no cache. A script outside any project caches beside itself, and changing
`cache_dir` leaves the old directory behind; pass either directory
explicitly.

## `cash autoload` { #cash-autoload }

*Notebook only.*

**Usage:** `cash autoload [--mode {available,active}] [--profile PROFILE] [--force] {on,off}`

Installs or removes an IPython startup file,
`<IPython dir>/profile_<profile>/startup/00-cash.py`, so every new kernel loads
cash.

- `on`: install the hook. `--mode active` (the default) imports cash and runs
  `%cash_on`, so caching is on in every kernel; `%cash_off` opts one session
  out. `--mode available` only imports cash; you still run `%cash_on`.
- `off`: remove the hook. If none is installed, it says so and exits 0.
- `--profile PROFILE`: the IPython profile, `default` unless you name another.
- `--force`: with `on`, overwrite a different file at that path; with `off`,
  remove a file that lacks cash's marker comment.

```bash
cash autoload on                       # import cash and run %cash_on in every kernel
cash autoload on --mode available      # only import cash
cash autoload off
```

Running `on` again with the same mode does nothing. To switch modes, run
`cash autoload off` first, or pass `--force`: `on` refuses to overwrite an
existing hook with a different body.

<!-- claim: cash/__main__.py:cmd_autoload @528fa896, cash/__main__.py:cmd_version @700ebd0c, cash/__main__.py:cmd_info @ee74bfe2 -->
## Exit codes

| Code | When |
|---|---|
| `0` | Success, including "nothing to clear" and "autoload not installed". |
| `1` | A refusal or a missing target: `cash inspect` with no cache or an unknown or ambiguous `--function`; `cash clear` with a missing path, an unknown `--function` or `--entry`, a directory that is not a cash cache (without `--force`), the current directory, or a file it cannot delete; `cash autoload` refusing to overwrite or remove a file (without `--force`). |
| `2` | A bad `cash clear` invocation: none of `path`, `--all`, `--function`, `--entry`, `--expired` or `--tool`; `--all` or `--tool` together with a path; `--expired` with `--function` or `--entry`; `--function`, `--entry` or `--expired` on a SQLite cache. Nothing is touched. |
