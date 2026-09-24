# Glossary

!!! info "Applies to: both paths"
    Each term is tagged with the path it belongs to: decorator, notebook, or
    both.

<div class="cash-glossary" markdown="1">

## Annotation

*Notebook.* A `# @cash:...` comment above a statement that overrides cash's
decision for it, such as `# @cash:no-cache` or `# @cash:persist`. See
[Annotations](annotations.md).

## `assume_safe`

*Decorator.* `@cash.cache(assume_safe=True)` tells cash you have checked the
function's side effects and it should cache it anyway. `# @cash:assume-safe`
on one line does the same for that line only. See the
[`@cash.cache` guide](decorator.md).

## Badge

*Notebook.* The summary cash shows above each cell's output: one row per
statement, reading **CACHED**, **EXECUTED**, **NOT CACHED** or **SKIPPED**,
with timings and warnings. The JSON from `%cash_status` uses the enum names
`COMPUTED` and `RESTORED` for the first two. See
[Reading the badge](badges.md).

## Cache key

*Both.* The hash that decides a hit or a miss. For a decorated function it
covers the arguments, the function's source and the source of the helpers it
calls. For a notebook statement it covers the code, the [lineage](#lineage) of
the variables it reads and its [file dependencies](#file-dependency). See
[Cache keys and lineage](how-it-works/cache-keys-and-lineage.md).

## Call-level caching

*Notebook.* Caching of a slow call inside a statement, keyed on the values
it receives, so a statement that cannot be cached (an `.append()` in a loop)
still skips the slow part. The badge shows it as a `sub-call` line.
`# @cash:no-cache-calls` turns it off. See
[Annotations](annotations.md).

## Cost model

*Notebook.* How cash decides whether a statement's result is worth storing
in RAM and on disk. It does not apply to decorated functions. See
[Cost model](cost-model.md).

## `depends_on` / `file_depends_on`

*Decorator.* Parameters that add dependencies cash cannot find itself.
`file_depends_on=` names files, checked by content like the files a function
reads. `depends_on=` takes other functions or data sources. See the
[`@cash.cache` guide](decorator.md).

## `explain()`

*Decorator.* `f.explain(*args)` says whether that call would hit, and if
not, why. See the [`@cash.cache` guide](decorator.md).

## File dependency

*Both.* A file a function or statement read through a tracked call
(`pd.read_csv`, `np.load`, `open`, ...). Change the file and the result
recomputes. See [Knowing when to recompute](how-it-works/invalidation.md).

## Hit / miss

*Both.* A **hit** means the [cache key](#cache-key) matched a stored entry, so
the result was restored without running the code. A **miss** means it ran.

## `KEY-*` warnings

*Decorator.* Warnings about what goes into a decorated call's key. For
example, `KEY-UNHASHABLE-ARG` means an argument cannot be hashed, so the call
runs uncached, and `KEY-NETWORK-READ` means the function reads from the
network, so it should have a `ttl`. See [Warnings](warnings.md).

## Lineage

*Notebook.* A hash attached to each variable that records how it was made:
the statement's code, the lineage of its inputs and the files it read. It
flows to every statement that reads the variable, so a change upstream
reaches everything below it. See
[Cache keys and lineage](how-it-works/cache-keys-and-lineage.md).

## Live reader

*Notebook.* A way for cash to read your cells as they are in the editor, not
as last saved: Colab, JupyterLab with cash's extension, and VS Code with hot
exit. Without one, cash reads the saved `.ipynb`. See
[Writing cache-safe cells](known-limitations.md#editing-without-saving).

## Mutation detection

*Notebook.* Detection of in-place changes such as `df['x'] = 0`,
`lst.append(...)` or `+=`, so that statements below see the changed object.
See [Knowing when to recompute](how-it-works/invalidation.md).

## Promotion

*Notebook.* Writing a statement's result from RAM to disk so it survives a
kernel restart. It happens when the statement took more than 0.1 s and
reloading is cheaper than recomputing. `# @cash:persist` forces it. Decorated
results are always written to disk. See [Cost model](cost-model.md).

## Provenance

*Notebook.* The statements and inputs that produced a variable, shown by
`%cash_provenance`. See [Seeing what cash did](how-it-works/inspecting.md).

## Purity and side effects

*Both.* Code is pure when it only computes a result: no file writes, network
or database calls. A notebook statement with a side effect is not cached; a
decorated function gets a warning, and its side effects run on the first call
only. `@cash.pure` and `@cash.stateful` state your intent. See
[Knowing when not to cache](how-it-works/safety.md).

## Skip / not cached

*Notebook.* **NOT CACHED** means the statement ran but its result was not
stored: it was under 10 ms, had a side effect, or carried
`# @cash:no-cache`. **SKIPPED** means there was nothing to do, such as an
`import` whose names are already bound. See [Reading the badge](badges.md).

## Statement-level caching

*Notebook.* Caching each statement in a cell on its own, so editing one line
re-runs only that line and what depends on it. See
[The notebook path](how-it-works/notebook-path.md).

## TTL

*Both.* A time to live: after `ttl` seconds an entry expires and the next call
recomputes. `@cash.cache(ttl=3600)`, `%cash_on ttl=3600`, or
`# @cash:ttl=3600` on one statement. See the
[`@cash.cache` guide](decorator.md) and [Annotations](annotations.md).

## Unseeded randomness

*Both.* A random draw with no fixed seed. Cash caches it like any other
result, so a hit replays the same draw, and it warns. Seed the generator, or
skip caching for a fresh draw. `allow_random=True` (decorator) and
`# @cash:allow-random` (notebook) only hide the warning.
See [Writing cache-safe cells](known-limitations.md).

## Upstream simulation

*Notebook.* Cash's dry run over the cells above the one you run. It works out
what to restore and what to re-run so the cell sees what a top-to-bottom run
would give it. See [The notebook path](how-it-works/notebook-path.md).

</div>
