"""Stable identifiers for every warning Cash emits.

A warning without a handle is a warning nobody can look up: there is nothing to
search for, nothing to google, nothing to ask a colleague about. Every warning
in the ``CashWarning`` hierarchy (see ``cash.exceptions``) carries a code from
this module and a link to its section in ``docs/warnings.md``.

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

import logging
import os
import re
import sys
import warnings
from typing import Any

# /en/latest/, not /en/stable/: the stable build is only rebuilt at a release,
# so a warning code documented since then would link to a missing anchor.
_DOCS_BASE = "https://cash-lib.readthedocs.io/en/latest/warnings/"

#: The installed package directory. A frame whose file lives under this is Cash
#: reporting on itself; anything else is the caller's own code.
#:
#: Matched as a path prefix, never as the substring "cash". A notebook statement
#: compiles under the pseudo-filename ``/<cash-{digest}>``, and that IS the
#: user's code -- a substring test would classify exactly the
#: frames we most want to blame as internal ones.
_CASH_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _is_cash_frame(frame) -> bool:
    """True if *frame* is executing Cash's own code."""
    try:
        return os.path.abspath(frame.f_code.co_filename).startswith(_CASH_ROOT)
    except (OSError, ValueError):
        # A synthetic or unresolvable filename is not Cash's; treating it as
        # the user's stops the walk rather than running off the top of the
        # stack, which is the safer failure.
        return False


def _stacklevel_of_first_user_frame() -> int:
    """The ``stacklevel`` that blames the nearest frame outside Cash.

    Hand-tuned ``stacklevel`` constants are unverifiable by reading and wrong
    often enough to matter: four separate diagnostics shipped pointing at a
    line inside ``core.py``, which tells a reader nothing they can act on. A
    constant also cannot be right for a site with two entry points at different
    depths -- and an over-deep one does not clamp to the outermost frame, it
    reports ``<sys>:0``.

    Counting: ``warnings.warn(..., stacklevel=1)`` blames the frame that calls
    it, so level 1 is the caller of this function (``warn_diagnostic``), which
    is itself Cash. Walking outward until the first non-Cash frame gives the
    level that names the user's own line, whatever the depth.

    Falls back to the outermost frame if every frame is Cash's -- a warning
    raised during ``import cash``, say. That is honest: there is no user frame
    to point at.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        if not _is_cash_frame(frame):
            return level
        frame = frame.f_back
        level += 1
    return level - 1


#: Standard-library machinery that calls user code without being it.
_MACHINERY = ("threading", "concurrent.futures", "multiprocessing", "asyncio")


def _machinery_roots() -> tuple[str, ...]:
    roots = []
    for name in _MACHINERY:
        path = getattr(sys.modules.get(name), "__file__", None)
        if path:
            path = os.path.abspath(path)
            roots.append(path if name == "threading" else os.path.dirname(path) + os.sep)
    return tuple(roots)


def _user_frame_level() -> int | None:
    """`_stacklevel_of_first_user_frame`, also passing over thread and process
    pool machinery -- or ``None`` when no frame on this thread's stack is the
    user's. A cached function first called in a ``ThreadPoolExecutor`` blamed
    CPython's ``concurrent/futures/thread.py``: a worker's stack
    has nothing of the user's above cash, and the line that submitted the
    work is on another thread."""
    roots = _machinery_roots()
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        if not _is_cash_frame(frame):
            try:
                path = os.path.abspath(frame.f_code.co_filename)
            except (OSError, ValueError):
                return level
            if not path.startswith(roots):
                return level
        frame = frame.f_back
        level += 1
    return None


def _shows_without_source_line(show: Any) -> Any:
    """*show*, a ``warnings._showwarnmsg``, made to leave out the source line
    under a cash warning.

    Python prints the blamed line after the message. Under a cash warning
    that is the user's call -- ``print(backtest(10))`` -- and says nothing
    the ``file:line`` before it does not, at the cost of a line per warning.
    A blank ``line`` is what tells the formatter there is none to print; any
    other warning goes through untouched.
    """

    def _showwarnmsg(msg: Any) -> None:
        if msg.line is None and _is_cash_category(msg.category):
            msg.line = ""
        show(msg)

    _showwarnmsg.cash_hook = True  # type: ignore[attr-defined]
    return _showwarnmsg


def _is_cash_category(category: Any) -> bool:
    from .exceptions import CashWarning

    return isinstance(category, type) and issubclass(category, CashWarning)


def _hide_source_lines() -> None:
    """Install `_shows_without_source_line` over the ``warnings`` module's
    display hook, once -- and again over whatever replaced it since."""
    show = getattr(warnings, "_showwarnmsg", None)
    if show is not None and not getattr(show, "cash_hook", False):
        warnings._showwarnmsg = _shows_without_source_line(show)  # type: ignore[attr-defined]


def _warn_at(instance: Warning, level: int | None, fallback: tuple[str, int] | None) -> bool:
    """Warn *instance* at *level* frames out -- counted from the caller of this
    function's caller -- or at *fallback* when there is no user frame.

    Returns whether the active warning filters let it through: False for an
    ``"ignore"`` filter. An ``"error"`` filter raises it instead of returning.
    """
    _hide_source_lines()
    if level is None and fallback is not None:
        filename, lineno = fallback
        module = filename[:-3] if filename.lower().endswith(".py") else filename
        shown = _filter_action(instance, module, lineno) != "ignore"
        warnings.warn_explicit(instance, type(instance), filename, lineno)
        return shown
    if level is None:
        level = _stacklevel_of_first_user_frame() - 1
    try:
        frame = sys._getframe(level)
        module, lineno = frame.f_globals.get("__name__", "<string>"), frame.f_lineno
    except ValueError:  # deeper than the stack: warnings blames `sys`
        module, lineno = "sys", 0
    shown = _filter_action(instance, module, lineno) != "ignore"
    warnings.warn(instance, stacklevel=level + 1)
    return shown


def _filter_action(instance: Warning, module: str, lineno: int) -> str:
    """The action the warning filters in force take on *instance* raised in
    *module* at *lineno*: the first filter that matches, as
    `warnings.warn_explicit` picks it, else the default action."""
    get_filters = getattr(warnings, "_get_filters", None)  # 3.14: per-context filters
    filters = get_filters() if get_filters is not None else warnings.filters
    text = str(instance)
    for action, msg, category, mod, ln in filters:
        if (
            (msg is None or msg.match(text))
            and isinstance(instance, category)
            and (mod is None or mod.match(module))
            and (ln == 0 or ln == lineno)
        ):
            return action
    return warnings.defaultaction


#: Every diagnostic code Cash can emit. Adding a warning means adding its code
#: here AND a section in ``docs/warnings.md``;
#: ``tests/docs/test_warning_codes_documented.py`` fails if either is missing.
#:
#: The gloss on each line is the one-sentence claim its doc section expands.
DIAGNOSTIC_CODES: frozenset[str] = frozenset(
    {
        # -- ANNOT: a ``# @cash:`` annotation Cash could not honour -------------
        "ANNOT-TTL-INVALID",  # `# @cash:ttl=` is not whole seconds; ignored
        "ANNOT-UNKNOWN-DIRECTIVE",  # `# @cash:<name>` is not a directive; ignored
        # -- CACHE: caching happened, or refused to, and it is worth saying -----
        "CACHE-ASYNC-GENERATOR",  # async generators are returned unwrapped
        "CACHE-EVICTED-RECOMPUTE",  # the cap had evicted a costly result, so it
        # was computed again
        "CACHE-IDENTITY-COUPLED",  # result is a live Figure/Axes; storing it would
        # detach the library's copy from yours
        "CACHE-IF-BYPASSED",  # result outgrew a chunk, so cache_if never ran
        "CACHE-FRESHNESS-COST",  # proving the entry fresh cost a serious share
        # of the compute it avoids
        "CACHE-IF-RAISED",  # the cache_if predicate raised
        "CACHE-LOOP-GROWTH",  # a loop is persisting every state of a growing
        # object, costing the sum of every snapshot
        "CACHE-NONDETERMINISTIC",  # a key computed again gave a different
        # result than the one stored under it
        "CACHE-NET-LOSS",  # key hashing has cost more than it has saved
        "CACHE-RETURNS-AWAITABLE",  # a sync wrapper returned a coroutine; not stored
        "CACHE-THRASH",  # at the cap, evicting within writes of storing
        "CACHE-CLEAR-INCOMPLETE",  # cache_clear() could not remove some entries,
        # which are still served
        "CACHE-DIR-NEW",  # cash created its default cache folder, and says where
        # (and names an existing cache nearby)
        "CACHE-DIR-UNWRITABLE",  # the cache directory cannot be written at all,
        # so nothing reaches disk this run
        "CACHE-PER-ITEM-RERUN",  # a costly step ran again over a whole
        # collection although most of its rows are unchanged
        "CACHE-NOT-WORTH-BYTES",  # more disk per second saved than cash will spend
        "CACHE-VALUE-TOO-BIG",  # too large for any persistent tier; RAM only
        "CACHE-WRITE-ABANDONED",  # a write was still running at the exit deadline,
        # so the process exited without storing it
        # -- CONFIG: a setting cash found but could not act on ------------------
        "CONFIG-INVALID",  # a value of the wrong type, or a config file that
        # is not valid TOML; ignored, defaults apply
        "CONFIG-UNKNOWN-KEY",  # a [tool.cash] / cash config key that is not a
        # setting, so it does nothing (usually a typo)
        "CONFIG-FILE-MISSING",  # Cash(config_path=...) names a file that does
        # not exist, so none of its settings apply
        # -- IMPURE: the function does something a cache hit will not repeat ----
        "IMPURE-HANDED-CALLABLE",  # an argument is a function of the user's
        # that changes state when called; the call runs uncached
        "IMPURE-OBSERVED-EFFECTS",  # watching the first call caught effects static
        # analysis could not see
        "IMPURE-SCOPE-MUTATION",  # calling it rewrites a global or captured
        # variable, which can then no longer be tracked
        "IMPURE-SIDE-EFFECTS",  # static analysis found likely side effects
        # -- KEY: something the result depends on is not in the cache key -------
        "KEY-AMBIENT-READ",  # the body reads the clock / environment / cwd /
        # a fresh uuid, so the first call's value freezes
        "KEY-BOOL-STATE-TOKEN",  # DataSource.state_token() returned a bool,
        # which cannot track changes
        "KEY-BUILD-FAILED",  # key construction raised
        "KEY-CALLABLE-HASHER",  # a hasher registered for every function /
        # method / partial; closures can share its value
        "KEY-DEPENDS-ON-OPAQUE",  # a declared depends_on= target has no readable
        # source, so editing it invalidates nothing
        "KEY-DYNAMIC-DEP-FAILED",  # a dynamic_depends_on resolver raised
        "KEY-FUNCTION-IMPURE",  # the key= function reads something besides its
        # arguments (a file, the clock, a random draw)
        "KEY-FUNCTION-RAISED",  # the key= function raised; the call ran uncached
        "KEY-FROZEN-MUTATED",  # a result declared frozen=True was modified
        "KEY-FROZEN-NO-EFFECT",  # frozen=True on a function whose result it cannot mark
        # after it was returned; keyed by content now
        "KEY-HELPERS-UNWALKABLE",  # the code it runs cannot all be found (a new
        # function on every read, or finding it raised); the function runs uncached
        "KEY-INSTANCE-STATE",  # a bound method's instance could not be hashed;
        # falling back to its process-local identity
        "KEY-LOOP-VAR-CHANGED",  # a loop body changes its loop variable in place
        # before work cached per pass, whose key took the variable as bound
        "KEY-ITERATOR-CONSUMED",  # building the key read an iterator argument
        # (a key= function, a registered hasher); the call ran uncached
        "KEY-MEMO-BEFORE-IMPORT",  # a functools memo of the user's held results
        # when cash was imported; a file it read then is not tracked
        "KEY-NETWORK-READ",  # the body fetches from a server; its answer is not
        # in the key, so it is served until a ttl= expires it
        "KEY-DYNAMIC-DEPENDENCY",  # code reached through an argument resolves a
        # dependency at runtime (getattr(m, name)(), eval)
        "KEY-OPAQUE-CALLABLE",  # a callable reached the call but its code could
        # not be hashed, so editing it changes nothing
        "KEY-STALE-CASH-KEY",  # one __cash_key__ stood for two different
        # contents, so one's results are served for the other
        "KEY-SOURCE-CHANGED",  # a cached function's or helper's file was edited
        # after import; keyed by the code actually running
        "KEY-UNHASHABLE-ARG",  # an argument could not be hashed; not cached
        "KEY-UNHASHABLE-CAPTURE",  # a value a closure reads could not be hashed;
        # not cached
        "KEY-UNHASHABLE-DEFAULT",  # a parameter default could not be hashed
        "KEY-UNHASHABLE-GLOBAL",  # a global the function reads could not be
        # hashed, so changing it invalidates nothing
        "KEY-UNRESOLVED-CALL",  # a call goes through a property or __getattr__
        # cash does not run, so the function it reaches is not in the key
        "KEY-UNTRACKABLE-DEP",  # the body picks code by a run-time name
        # (getattr(m, name)(), import_module, eval); the function runs uncached
        # -- NOTEBOOK: notebook-wide machinery, not one statement ---------------
        "NOTEBOOK-BAILOUT",  # cash hit an internal error, stepped aside, and
        # ran the cell uncached
        "NOTEBOOK-ANALYSIS-BUDGET",  # working out how to key a statement took
        # longer than caching it could save, so it ran uncached
        "NOTEBOOK-ANALYSIS-FAILED",  # a safety check on a statement raised, so
        # the statement ran uncached
        "NOTEBOOK-FILE-ROUNDTRIP",  # a statement reads back a file a statement
        # of this session wrote, and each new write makes it and what is built
        # from it run again
        "NOTEBOOK-FAILED-CELL",  # a cell raised part way, and a name it
        # changed is kept as that run left it, or could not be put back
        "NOTEBOOK-CHANGED-BELOW",  # a cell re-run on its own reads a value rebuilt
        # without the in-place change a cell below made to it
        "NOTEBOOK-CELL-SYNTAX",  # an upstream cell does not parse, so cells that
        # depend on it stop being tracked
        "NOTEBOOK-CLASS-KEPT",  # a class's own state a cell changes is kept
        # as it is for a re-run, as another cell changes it too
        "NOTEBOOK-FORWARD-READ",  # the cell reads a name only a cell
        # below binds; it runs uncached on the value in memory
        "NOTEBOOK-INPUT-UNBUILT",  # an input the cell's own cache hit was
        # to restore was not: the statement ran on the old value, unstored
        "NOTEBOOK-MAGIC-STALE",  # a name a magic bound is kept as it is,
        # though what the magic read changed since it ran, or is gone
        "NOTEBOOK-NO-LIVE-READER",  # cells above are read from the saved
        # file and nothing sends unsaved edits; shown once per session
        "NOTEBOOK-NOT-FOUND",  # no notebook cells to read; cells above the one
        # run are not re-checked, statements still cache on their own inputs
        "NOTEBOOK-RELOAD-STATE",  # an edited module was reloaded, or the
        # kernel restarted, and cash cannot rebuild the state a cell set on it
        "NOTEBOOK-THREAD-ALIVE",  # a thread the cell started still runs after
        # it ended; what it changes from then on is outside cash's view
        "NOTEBOOK-WORK-RUNNING",  # a thread or process a statement left
        # running has not finished; statements run rather than be served
        "NOTEBOOK-SAVEFIG-SKIP",  # refused to re-run plt.savefig() during
        # reconstruction; it would overwrite your chart
        "NOTEBOOK-SHARE-UNCHECKED",  # cash could not find every variable
        # sharing an object with a statement's outputs in time
        "NOTEBOOK-SHARED-KEPT",  # a value another name below still shares
        # is kept as it is rather than rebuilt for a cell re-run on its own
        # -- RANDOM: a cached value that randomness makes non-reproducible ------
        "RANDOM-HELD-GENERATOR",  # a cell re-run on its own draws from a generator
        # held by an object, which cash does not put back where a run from the top has it
        "RANDOM-REPLAYED",  # what you are seeing is a replay of an earlier
        # draw, not a fresh one
        "RANDOM-SEED-NONE",  # seed(None) cannot refresh cached values below
        "RANDOM-UNSEEDED",  # an unseeded draw is being cached and frozen
        # -- REMOTE: tracking a remote object's freshness -----------------------
        "REMOTE-FRESHNESS-COST",  # freshness checks cost more than they protect
        "REMOTE-SIZE-ONLY",  # tracked by size alone; a same-size edit is
        # invisible
        "REMOTE-STATE-UNREADABLE",  # could not read remote state; will recompute
        # -- STORE: compute succeeded, the write did not ------------------------
        "STORE-CODE-CHANGED",  # a code file changed on disk after the key
        # was read, so the result was not stored
        "STORE-CHUNK-FAILED",  # a chunked write failed partway; the entry is
        # incomplete on retrieval
        "CACHE-RESULT-SHARED",  # the result shares state with the caller's own object
        "STORE-FAILED",  # the backend refused the write
        "STORE-INPUT-CHANGED",  # a file the call read changed before it
        # returned, so the result was not stored
        "STORE-LOCK-FAILED",  # lock acquisition failed; proceeding unlocked
        "STORE-STATE-CHANGED",  # a global the key read changed (another
        # thread) while the call ran, so the result was not stored
        "STORE-WORK-RUNNING",  # work the call handed to a thread or pool was
        # still running when it returned, so the result was not stored
        "STORE-UNTRACKED-SOURCE",  # a cached callee depends on a dynamic source
        # that cannot be pickled, so the caller's result stays in RAM (or,
        # with no RAM tier, is not stored)
        "STORE-METADATA-INVALID",  # a stored entry's metadata did not validate
        "STORE-PARTS-SHARED",  # parts of the result share memory a hit
        # hands back as separate copies, so the result was not stored
        "STORE-RESULT-WRITTEN",  # a table kept uncopied was written past
        # pandas after its store, so the entry was dropped
    }
)


def doc_url(code: str) -> str:
    """The documentation anchor for *code*.

    Raises ``KeyError`` for an unregistered code so a typo fails at the emit
    site instead of shipping a link that 404s.
    """
    if code not in DIAGNOSTIC_CODES:
        raise KeyError(f"unknown diagnostic code: {code!r}")
    return f"{_DOCS_BASE}#{code.lower()}"


def format_diagnostic(code: str, what: str, fix: str) -> str:
    """Render the three-part message body.

    Python's own machinery prints the location and the category name, so this
    starts at the code::

        foo.py:12: CashCacheIneffectiveWarning: [CACHE-THRASH] the cache is
        full at its 500 MB cap.
          Fix: raise max_cache_size, or cache fewer values.
          https://cash-lib.readthedocs.io/en/latest/warnings/#cache-thrash

    *what* is one sentence of what happened; *fix* is one imperative sentence.
    Everything else belongs in the doc section the link points at, so the
    message stays short.

    This is the full form: what ``cache_info()['warnings']`` and logging keep,
    and what a warning shows under ``warning_detail = "full"``. By default a
    warning shows :func:`short_diagnostic` instead.
    """
    return f"[{code}] {what}\n  Fix: {fix}\n  {doc_url(code)}"


#: A sentence ends at ``.``/``!``/``?`` followed by space and what starts the
#: next one -- a capital, a backtick, a quote, ``(``. ``@cash.cache`` and
#: ``e.g. a`` do not end one.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`'\"(])")

#: Inline code as messages spell it, two backticks a side as in the docs: a
#: terminal shows them as typed, so the one-line form keeps one a side.
_INLINE_CODE = re.compile(r"``([^`]+)``")

#: An absolute path in a listed item, shortened to its last two parts.
_LONG_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s\\/,:]+[\\/]){2,}([^\s\\/,:]+[\\/][^\s\\/,]+)")


def short_path(text: str) -> str:
    """*text* with each absolute path cut to ``.../<dir>/<file>``."""
    return _LONG_PATH.sub(lambda m: ".../" + m.group(1).replace("\\", "/"), text)


def list_gist(items: list[str], noun: str = "finding") -> str:
    """``"3 findings, e.g. <first>"`` -- or the one item -- for a listed detail
    collapsed into one line, with its paths shortened."""
    if not items:
        return ""
    first = short_path(items[0].strip())
    if len(items) == 1:
        return first
    return f"{len(items)} {noun}s, e.g. {first}"


def short_diagnostic(code: str, what: str, fix: str | None = None, *, more: str | None = None) -> str:
    """The one-line form a warning shows by default::

        [CACHE-THRASH] the cache is full at its 500 MB cap. Fix: raise max_cache_size. (https://...#cache-thrash)

    The first sentence of *what*; an indented list below it (effects,
    findings) collapses to a count and its first item. All of *fix*
    follows, on the same line: the way out (``# @cash:allow-random``,
    "re-run cell 3") is what a reader acts on, and it is often the last
    clause. The rest is in the doc section the link points at; *more* names where the full text of this one is kept
    (``f.cache_info()['warnings']``). Inline code keeps one backtick a side.
    """
    head, _, rest = what.partition("\n")
    sentences = _SENTENCE_END.split(head.strip(), maxsplit=2)
    sentence = sentences[0]
    # "@cash.cache on f: result not cached." says what, not why: a first
    # sentence of a few words after its subject takes the next one along.
    if len(sentences) > 1 and len(sentence.rpartition(": ")[2].split()) < 6:
        sentence = f"{sentence} {sentences[1]}"
    # A heading line ("in helper (mod.py):") groups items; it is not one.
    items = [line.strip() for line in rest.splitlines() if line.strip() and not line.rstrip().endswith(":")]
    if items:
        sentence = f"{sentence.rstrip(' .:')}: {list_gist(items)}"
    if fix and fix.strip():
        sentence = f"{sentence} Fix: {' '.join(fix.split())}"
    where = doc_url(code) if more is None else f"{more}; {doc_url(code)}"
    sentence = _INLINE_CODE.sub(r"`\1`", sentence)
    return f"[{code}] {sentence} ({where})"


def full_warnings(config: Any = None) -> bool:
    """Does ``warning_detail = "full"`` govern the code running now?

    *config* when given; else the running call's instance, else the default
    instance -- both read ``CASH_WARNING_DETAIL`` and the config files --
    else the variable itself: resolving the config files here could warn,
    and so call back in.
    """
    if config is not None:
        return getattr(config, "warning_detail", "short") == "full"
    try:
        from . import _active  # cash._active imports the config, which imports this module

        config = _active.ACTIVE_CONFIG.get()
        if config is None:
            default = _active.default_cash()
            config = default.config if default is not None else None
    except (ImportError, AttributeError):  # warned while cash._active is still importing
        config = None
    if config is not None:
        return getattr(config, "warning_detail", "short") == "full"
    return os.environ.get("CASH_WARNING_DETAIL", "").strip().lower() == "full"


def render_diagnostic(code: str, what: str, fix: str, *, more: str | None = None) -> str:
    """What a warning shows: :func:`format_diagnostic` under
    ``warning_detail = "full"``, else :func:`short_diagnostic`."""
    if full_warnings():
        return format_diagnostic(code, what, fix)
    return short_diagnostic(code, what, fix, more=more)


def log_diagnostic(log: logging.Logger, code: str, what: str, fix: str) -> None:
    """Also send a diagnostic to logging -- when logging is configured.

    For the few advisories that should reach a log-only reader too. With no
    handler anywhere, which is every plain script, the record would go to
    logging's last-resort stderr handler and print the same text a second
    time right above the warning.
    """
    if log.hasHandlers():
        log.warning(format_diagnostic(code, what, fix))


def warn_diagnostic(
    category: type[Warning],
    code: str,
    what: str,
    fix: str,
    *,
    location: tuple[str, int] | None = None,
) -> None:
    """Emit *category* carrying *code*, its rendered message, and ``.code``.

    *location* ``(filename, lineno)`` blames that place instead of a frame,
    keeping ``.code`` (``warn_explicit`` takes the instance as its message).
    For a warning raised from cash's own machinery with no user frame worth
    naming: resolved "at emit time" from a notebook backend, the nearest frame
    outside Cash is ipykernel's, and every such warning read
    ``ipkernel.py:460: ...``.

    Warns with an *instance* rather than a message string so the code survives
    to the handler: a caller can test ``w.message.code == "CACHE-THRASH"``
    instead of matching prose that is free to be reworded.

    **The blamed frame is resolved at emit time**: without *location*, the
    warning names the nearest frame outside Cash, whatever the call depth --
    see ``_stacklevel_of_first_user_frame``.

    Raises ``KeyError`` before emitting anything if *code* is unregistered.
    """
    message = render_diagnostic(code, what, fix)  # raises on an unknown code
    instance = category(message)
    instance.code = code
    if location is not None:
        _hide_source_lines()
        warnings.warn_explicit(instance, category, filename=location[0], lineno=location[1], registry=None)
        return
    _warn_at(instance, _user_frame_level(), None)


def warn_diagnostic_message(
    category: type[Warning],
    code: str,
    message: str,
    *,
    fallback: tuple[str, int] | None = None,
) -> bool:
    """Emit an already-rendered *message* carrying *code*; whether the
    warning filters let it through (`_warn_at`).

    ``Notices.warn_once`` renders with :func:`format_diagnostic` itself, because
    it files the same text into ``cache_info()['warnings']`` before emitting and
    the log and the terminal must not drift apart. This keeps the ``.code``
    attribute and the registry check for that path.

    Blames the nearest frame outside Cash, the same as :func:`warn_diagnostic`
    -- or, with no frame of the user's on this thread's stack (a pool worker),
    the ``(filename, lineno)`` in *fallback*.
    """
    if code not in DIAGNOSTIC_CODES:
        raise KeyError(f"unknown diagnostic code: {code!r}")
    instance = category(message)
    instance.code = code
    return _warn_at(instance, _user_frame_level(), fallback)
