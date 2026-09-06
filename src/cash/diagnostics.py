"""Stable identifiers for every warning Cash emits.

A warning without a handle is a warning nobody can look up: there is nothing to
search for, nothing to google, nothing to ask a colleague about. Every Cash
warning carries a code from this module and a link to its section in
``docs/warnings.md``.

Codes are mnemonic on purpose. ``CACHE-THRASH`` tells the reader something
before they click; ``CASH-W012`` requires exactly the lookup we are trying to
make optional.

**A released code is permanent.** It appears in users' terminal output and in
issue reports, so renaming one orphans those. To retire a code, leave its
anchor in the docs as a stub pointing at the replacement.

A code names a *documentation section*, not a call site. Sites that tell one
story share one code (three "the backend refused the write" sites are all
``STORE-FAILED``), and one site whose message varies over a family of causes
still gets one code (``CACHE-IDENTITY-COUPLED`` names every kind of
identity-coupled object). The test for "same code?" is whether one section of
prose, with one piece of advice, serves every site that emits it.
"""
from __future__ import annotations

_DOCS_BASE = "https://cash-lib.readthedocs.io/en/stable/warnings/"

#: Every diagnostic code Cash can emit. Adding a warning means adding its code
#: here AND a section in ``docs/warnings.md``; the bijection test in
#: ``tests/docs/test_warning_codes_documented.py`` fails if either is missing.
#:
#: The gloss on each line is the one-sentence claim its doc section expands.
DIAGNOSTIC_CODES: frozenset[str] = frozenset({
    # -- ANNOT: a ``# @cash:`` annotation Cash could not honour -------------
    "ANNOT-TTL-INVALID",       # `# @cash:ttl=` is not whole seconds; ignored

    # -- CACHE: caching happened, or refused to, and it is worth saying -----
    "CACHE-ASYNC-GENERATOR",   # async generators are returned unwrapped
    "CACHE-IDENTITY-COUPLED",  # result is a live Figure/Axes; storing it would
                               # detach the library's copy from yours
    "CACHE-IF-BYPASSED",       # result outgrew a chunk, so cache_if never ran
    "CACHE-IF-RAISED",         # the cache_if predicate raised
    "CACHE-LOOP-GROWTH",       # a loop is persisting every state of a growing
                               # object, costing the sum of every snapshot
    "CACHE-NET-LOSS",          # key hashing has cost more than it has saved
    "CACHE-THRASH",            # at the cap, evicting within writes of storing
    "CACHE-VALUE-TOO-BIG",     # too large for any persistent tier; RAM only

    # -- KEY: something the result depends on is not in the cache key -------
    "KEY-BOOL-STATE-TOKEN",    # DataSource.has_changed() returned a bool,
                               # which cannot track changes
    "KEY-BUILD-FAILED",        # key construction raised
    "KEY-DEPENDS-ON-OPAQUE",   # a declared depends_on= target has no readable
                               # source, so editing it invalidates nothing
    "KEY-DYNAMIC-DEP-FAILED",  # a dynamic_depends_on resolver raised
    "KEY-INSTANCE-STATE",      # a bound method's instance could not be hashed;
                               # falling back to its process-local identity
    "KEY-OPAQUE-CALLABLE",     # a callable reached the call but its code could
                               # not be hashed, so editing it changes nothing
    "KEY-UNHASHABLE-ARG",      # an argument could not be hashed; not cached
    "KEY-UNHASHABLE-DEFAULT",  # a parameter default could not be hashed
    "KEY-UNHASHABLE-GLOBAL",   # a global the function reads could not be
                               # hashed, so changing it invalidates nothing

    # -- IMPURE: the function does something a cache hit will not repeat ----
    "IMPURE-OBSERVED-EFFECTS", # watching the first call caught effects static
                               # analysis could not see
    "IMPURE-SCOPE-MUTATION",   # calling it rewrites a global or captured
                               # variable, which can then no longer be tracked
    "IMPURE-SIDE-EFFECTS",     # static analysis found likely side effects

    # -- NOTEBOOK: notebook-wide machinery, not one statement ---------------
    "NOTEBOOK-CELL-SYNTAX",    # an upstream cell does not parse, so cells that
                               # depend on it stop being tracked
    "NOTEBOOK-NOT-FOUND",      # no notebook path; upstream tracking is off
    "NOTEBOOK-SAVEFIG-SKIP",   # refused to re-run plt.savefig() during
                               # reconstruction; it would overwrite your chart

    # -- RANDOM: a cached value that randomness makes non-reproducible ------
    "RANDOM-REPLAYED",         # what you are seeing is a replay of an earlier
                               # draw, not a fresh one
    "RANDOM-SEED-NONE",        # seed(None) cannot refresh cached values below
    "RANDOM-UNSEEDED",         # an unseeded draw is being cached and frozen

    # -- REMOTE: tracking a remote object's freshness -----------------------
    "REMOTE-FRESHNESS-COST",   # freshness checks cost more than they protect
    "REMOTE-SIZE-ONLY",        # tracked by size alone; a same-size edit is
                               # invisible
    "REMOTE-STATE-UNREADABLE", # could not read remote state; will recompute

    # -- STORE: compute succeeded, the write did not ------------------------
    "STORE-CHUNK-FAILED",      # a chunked write failed partway; the entry is
                               # incomplete on retrieval
    "STORE-FAILED",            # the backend refused the write
    "STORE-LOCK-FAILED",       # lock acquisition failed; proceeding unlocked
    "STORE-METADATA-INVALID",  # a stored entry's metadata did not validate
})


def doc_url(code: str) -> str:
    """The documentation anchor for *code*.

    Raises ``KeyError`` for an unregistered code so a typo fails at the emit
    site instead of shipping a link that 404s.
    """
    if code not in DIAGNOSTIC_CODES:
        raise KeyError(f"unknown diagnostic code: {code!r}")
    return f"{_DOCS_BASE}#{code.lower()}"
