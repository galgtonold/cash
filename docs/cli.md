# Command-line interface

Cash ships a `cash` CLI for the tasks that don't fit cleanly inside a notebook
cell — installing the global IPython autoload hook, inspecting on-disk cache
directories, clearing caches, and reporting which configuration is in effect.
Behaviour is derived directly from `src/cash/__main__.py`; this page is the
canonical reference.

`cash` and `python -m cash` are equivalent entry points (the script is declared
as `cash = "cash.__main__:main"` in `pyproject.toml`). Running `cash` with no
subcommand prints help and exits 0.

<!-- claim: cash/__main__.py:main @88959c97 broad="the quick-reference table is a claim about the whole subcommand set" -->
## Quick reference

| Subcommand | Purpose | Destructive? |
|---|---|---|
| [`cash version`](#cash-version) | Print the installed cash version. | No |
| [`cash info`](#cash-info) | Show the effective merged configuration. | No |
| [`cash inspect [path] [--function NAME]`](#cash-inspect-path) | Summarise a cache directory or notebook; drill into one function. | No |
| [`cash clear [path] [--all] [--function NAME] [--entry ID]`](#cash-clear-path-all) | Delete a cache directory, one function's entries, or a single entry. | **Yes** — no confirmation prompt |
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
# cash <!-- docnum:version -->0.7.0<!-- /docnum -->
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
- `Cache dir` — the on-disk cache directory the file backend will use.
- `Debug` — whether debug logging is enabled.
- `Compress` — whether cache entries are compressed on disk.
- `Max size` — the maximum cache size, or `auto (scaled to disk/RAM per tier)`
  when unset (the default).
- `Persist` — what actually decides disk persistence: the cost model
  (`0.1s compute floor, N% savings required`), or a conservative fallback when
  smart persistence is off.
- `Tiers` — present when the active config declares an explicit tier
  list; lists each tier's type in order.
- `Source` — which layers contributed to the resolved config (e.g.
  `project:./pyproject.toml,env`, or `defaults` when nothing was set).

**Examples:**

```bash
cash info
# Cash v<!-- docnum:version -->0.7.0<!-- /docnum -->
#   Backend:    tiered
#   Cache dir:  /home/me/project/.cash
#   Debug:      False
#   Compress:   True
#   Max size:   auto (scaled to disk/RAM per tier)
#   Persist:    cost model (0.1s compute floor, 20% savings required)
#   Source:     project:/home/me/project/pyproject.toml,env
```

**Behaviour notes:**

- `cash info` shows a `Source:` line listing which layers contributed
  (`defaults`, `user:<path>`, `project:<path>`, `env`, `kwargs`) but
  does NOT show which layer was authoritative for each individual
  field. To debug a specific override, inspect the relevant `CASH_*`
  env vars and the relevant TOML files (project `pyproject.toml`
  `[tool.cash]` and XDG user config — see
  [Configuration](getting-started/configuration.md#file-locations)).

<!-- claim: cash/__main__.py:cmd_inspect @f70de23e, cash/__main__.py:_inspect_cache_dir @24ec3843, cash/__main__.py:_inspect_notebook @06ba3efe -->
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
    - **Omitted.** Defaults to `./.cash` in the current working directory. If
      that directory does not exist, prints
      `No cache found. Specify a notebook or cache directory.` and exits 1.

**Examples:**

```bash
cash inspect                           # inspect ./.cash
cash inspect ./.cash                   # explicit
cash inspect ./notebooks/analysis.ipynb
cash inspect /tmp/some-cache-dir
```

- `--function NAME` — *Optional.* List one function's individual entries
  instead of the per-function overview. An unambiguous trailing segment is
  enough: `--function work` finds `model.work`. `notebook` (or
  `statements`) selects the `(notebook statements)` group without its
  brackets. An ambiguous name prints the candidates and exits 1; an unknown
  one prints the functions that *are* cached.

  Each row says what the entry is **worth**, not just how big it is:

  ```
  ENTRY             SAVES       SIZE   USES   LAST USED   PRODUCES
  aaaaaaaaaaaa      12.5s     4.0 KB     3x   2 min ago   df, model
  bbbbbbbbbbbb       0.4s     1.0 KB     1x   2 min ago   scores
  ```

  `SAVES` is the recorded execution time — what you lose by deleting it —
  which together with `SIZE` and `USES` is the whole trade. `PRODUCES` names
  the variables a notebook statement produced, and is omitted when no entry
  in the group has any.

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

<!-- claim: cash/__main__.py:cmd_clear @f0272005 -->
### `cash clear [path] [--all] [--function NAME]` { #cash-clear-path-all }

Delete a cache directory, or just one function's entries.

!!! warning "Destructive without confirmation"
    `cash clear` calls `shutil.rmtree()` immediately on the resolved
    directory. There is **no confirmation prompt** and **no `--force` flag**
    — running the command deletes the cache as soon as you press enter. Be
    sure of the target before you run it, especially in CI.

**Usage:** `cash clear [path] [--all] [--function NAME]`

**Arguments:**

- `path` — *Optional.* One of:
    - **A directory.** Removed in full via `shutil.rmtree`.
    - **A `.ipynb` file.** Cash removes the sibling `.cash/` directory next
      to the notebook (if any). If there's no sibling cache, prints
      `No cache found for <path>` and exits 0.
    - **Anything else.** Prints `Not found: <path>` and exits 1.
- `--all` — *Optional.* Clear `./.cash` in the current working directory.
  When set, any `path` argument is ignored. If `./.cash` doesn't exist,
  prints `No .cash directory found in current directory` and exits 0.
- `--function NAME` — *Optional.* Delete only that function's entries and
  leave the rest of the cache intact — the alternative to keeping a cache you
  cannot afford or deleting work you still want. Resolves names exactly as
  `cash inspect --function` does, including `notebook`. Takes precedence over
  `--all`.
- `--entry ID` — *Optional.* Delete a single entry, using an id from
  `cash inspect --function NAME`. Any unambiguous prefix works, like a short
  commit hash; an ambiguous one lists the matches and deletes nothing. Takes
  precedence over `--function`.

**Examples:**

```bash
cash clear --entry a1b2c3              # drop one entry
cash clear --function ray.build_grid   # drop one function, keep the rest
cash clear --function notebook         # drop the notebook statements only
cash clear --all                       # nuke ./.cash
cash clear ./.cash                     # same thing, explicit
cash clear ./notebooks/analysis.ipynb  # nuke the sibling .cash next to the notebook
cash clear /tmp/some-cache-dir         # nuke any directory
```

**Behaviour notes:**

- If none of `path`, `--all` or `--function` is supplied, cash prints the
  `cash clear` help and exits 2 without touching anything.
- The no-op "nothing to clear" message paths (no `./.cash`, no sibling cache)
  exit 0; they're treated as success, not failure.

---

<!-- claim: cash/__main__.py:cmd_autoload @528fa896, cash/__main__.py:cmd_version @700ebd0c, cash/__main__.py:cmd_info @c6a1b14a -->
## Exit codes

| Code | When |
|---|---|
| `0` | The command succeeded, including no-op outcomes ("nothing to clear", "autoload not installed at ..."). |
| `1` | User-error refusals: `cash inspect` with no `./.cash` and no path; `cash clear <missing-path>`; `cash autoload on` refusing to overwrite without `--force`; `cash autoload off` refusing to delete a non-cash file without `--force`. |
| `2` | `cash clear` with none of `path`, `--all` or `--function` — argparse's own "bad invocation" code. The subcommand's help is printed and nothing is touched. |
| traceback | Uncaught exceptions bubble up as Python tracebacks — `main()` does not wrap dispatch in `try/except`. If you see one, treat it as a bug and please file an issue. |
