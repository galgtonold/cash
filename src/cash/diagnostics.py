"""Stable identifiers for every warning Cash emits.

A warning without a handle is a warning nobody can look up: there is nothing to
search for, nothing to google, nothing to ask a colleague about. Every warning
in the ``CashWarning`` hierarchy (see ``cash.exceptions``) carries a code from
this module and a link to its section in ``docs/warnings.md``. The one
exception is ``cash.experimental``, whose import-time notice warns with a
plain ``FutureWarning`` outside that hierarchy on purpose -- it flags an
unstable API, not a diagnosable condition, so it gets no code.

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
#: here AND a section in ``docs/warnings.md``. Once both exist, the bijection
#: test at ``tests/docs/test_warning_codes_documented.py`` will fail if either
#: is missing; until that test lands (a later task in this plan), treat this
#: as a contract to honour by hand.
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

    # -- IMPURE: the function does something a cache hit will not repeat ----
    "IMPURE-OBSERVED-EFFECTS", # watching the first call caught effects static
                               # analysis could not see
    "IMPURE-SCOPE-MUTATION",   # calling it rewrites a global or captured
                               # variable, which can then no longer be tracked
    "IMPURE-SIDE-EFFECTS",     # static analysis found likely side effects

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


import warnings


def format_diagnostic(code: str, what: str, fix: str) -> str:
    """Render the three-part message body.

    Python's own machinery prints the location and the category name, so this
    starts at the code::

        foo.py:12: CashCacheIneffectiveWarning: [CACHE-THRASH] the cache is
        full at its 500 MB cap.
          Fix: raise max_cache_size, or cache fewer values.
          https://cash-lib.readthedocs.io/en/stable/warnings/#cache-thrash

    *what* is one sentence of what happened; *fix* is one imperative sentence.
    Everything else belongs in the doc section, which is the whole point — the
    message used to carry a paragraph because it had nowhere to point.
    """
    return f"[{code}] {what}\n  Fix: {fix}\n  {doc_url(code)}"


def warn_diagnostic(
    category: type[Warning],
    code: str,
    what: str,
    fix: str,
    *,
    stacklevel: int = 2,
) -> None:
    """Emit *category* carrying *code*, its rendered message, and ``.code``.

    Warns with an *instance* rather than a message string so the code survives
    to the handler: a caller can test ``w.message.code == "CACHE-THRASH"``
    instead of matching prose that is free to be reworded.

    Raises ``KeyError`` before emitting anything if *code* is unregistered.
    """
    message = format_diagnostic(code, what, fix)   # raises on an unknown code
    instance = category(message)
    instance.code = code
    warnings.warn(instance, stacklevel=stacklevel + 1)


def warn_diagnostic_explicit(
    category: type[Warning],
    code: str,
    what: str,
    fix: str,
    *,
    filename: str,
    lineno: int,
    registry: dict | None = None,
) -> None:
    """Emit a coded diagnostic blamed on an explicit *filename* and *lineno*.

    For sites that must point at the user's own cell rather than the frame that
    happened to call them -- ``warnings.warn``'s ``stacklevel`` cannot express
    "that line over there".

    ``warn_explicit`` takes ``(message, category)`` positionally and gives no
    way to pass a pre-built instance, so ``.code`` cannot ride along on the
    warning object here. Callers that need the code programmatically from these
    sites should read it from the rendered text, which always starts
    ``[CODE] ``. This asymmetry is why the test below pins it.
    """
    warnings.warn_explicit(
        format_diagnostic(code, what, fix),   # raises on an unknown code
        category,
        filename=filename,
        lineno=lineno,
        registry=registry,
    )


def warn_diagnostic_message(
    category: type[Warning], code: str, message: str, *, stacklevel: int = 2
) -> None:
    """Emit an already-rendered *message* carrying *code*.

    ``Cash._warn_once`` renders with :func:`format_diagnostic` itself, because
    it files the same text into ``cache_info()['warnings']`` before emitting and
    the log and the terminal must not drift apart. This keeps the ``.code``
    attribute and the registry check for that path.
    """
    if code not in DIAGNOSTIC_CODES:
        raise KeyError(f"unknown diagnostic code: {code!r}")
    instance = category(message)
    instance.code = code
    warnings.warn(instance, stacklevel=stacklevel + 1)
