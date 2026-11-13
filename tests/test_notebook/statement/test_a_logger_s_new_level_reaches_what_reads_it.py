"""A statement reading a logger, or a value holding one, sees its new level.

A logger pickles by name (``getLogger(name)``), so a digest of a value
holding one holds nothing of its level, and the name finds it from anywhere.
``seen = level_of(cfg)`` -- ``cfg`` holding the logger three dicts deep --
was served 30 where plain Python gives 10 after ``log.setLevel(10)``, and
after ``logging.getLogger('app').setLevel(10)``, at the top level or in a
function, which names no variable at all. So was a statement reading
``seen`` below it. And a statement that set the level was served as a hit
when its cell ran again, leaving the level as it was.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import types
import warnings

import pytest

from tests._cell_driver import run_cash_cell

needs_monitoring = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sys.monitoring needs Python 3.12: below it what a statement finds by a lookup is not watched "
    "(NOTEBOOK-LOOKUPS-UNSEEN)",
)

NAME = "cash_test_logger_level"


@pytest.fixture(autouse=True)
def no_root_handlers(monkeypatch):
    # pytest's own handler on the root logger holds a stream, which no key
    # reads: the logger is then keyed without it, as in a notebook.
    monkeypatch.setattr(logging.root, "handlers", [])


def _setup(calls):
    return (
        "import logging\n"
        f"log = logging.getLogger({NAME!r})\n"
        "log.setLevel(30)\n"
        "cfg = {'a': {'b': {'c': {'log': log}}}}\n"
        f"CALLS = {str(calls)!r}\n"
        "def level_of(c):\n"
        "    with open(CALLS, 'a') as f:  # @cash:assume-safe\n"
        "        f.write('x')\n"
        "    return c['a']['b']['c']['log'].level\n"
        "def set_level(n):\n"
        f"    logging.getLogger({NAME!r}).setLevel(n)"
    )


@pytest.mark.parametrize(
    "change",
    ["log.setLevel(10)", f"logging.getLogger({NAME!r}).setLevel(10)", "set_level(10)"],
    ids=["by-variable", "by-name", "in-a-function"],
)
@pytest.mark.parametrize("reader", ["level_of(cfg)", "level_of({'a': {'b': {'c': {'log': log}}}})"])
def test_a_reader_runs_again_after_the_level_changes(cash_magics, cash_instance, tmp_path, change, reader):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    # Every call is kept as well: re-run, the reader's ``level_of(...)`` is
    # looked up in the call cache, keyed on the logger's name as the
    # statement is.
    cash_instance.config.call_cost_floor_seconds = 0.0
    calls = tmp_path / "calls"
    cells = [
        _setup(calls),
        f"seen = {reader}",
        "below = seen + 1",
        f"# served\nseen = {reader}",
        change,
        f"# again\nseen = {reader}",
        "# again\nbelow = seen + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 10
    assert cash_magics.shell.user_ns["below"] == 11
    assert calls.read_text(encoding="utf-8") == "xx"


@pytest.mark.parametrize(
    "change",
    [f"logging.getLogger({NAME!r}).setLevel(10)", "set_level(10)", "done = set_level(10)"],
    ids=["by-name", "in-a-function", "in-a-kept-call"],
)
def test_a_statement_setting_the_level_runs_again(cash_magics, cash_instance, tmp_path, change):
    """Its cell run again after the cell above set the level back: a hit
    would leave it there, as a stored pickle of the logger holds its name.
    So would a hit of the call that set it, kept by the call cache when the
    statement binds what it returns."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [_setup(tmp_path / "calls"), change, "seen = level_of(cfg)"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert logging.getLogger(NAME).level == 10
    assert cash_magics.shell.user_ns["seen"] == 10


#: The logger as a module's attribute: read through the module, no call.
LOGGER = logging.getLogger(NAME)


class Holder:
    """The logger as a class's attribute, and an object's."""

    LOG = LOGGER


HOLDER = types.SimpleNamespace(log=LOGGER)

LOOKUP_SETUP = (
    "import logging\n"
    f"import {__name__} as here\n"
    f"logging.getLogger({NAME!r}).setLevel(10)\n"
    "def level_now():\n"
    f"    return logging.getLogger({NAME!r}).level\n"
    "def level_of_item(_):\n"
    "    return level_now()\n"
    "def set_level(n):\n"
    f"    logging.getLogger({NAME!r}).setLevel(n)\n"
    "    return n"
)


@needs_monitoring
@pytest.mark.parametrize(
    "reader",
    [
        f"seen = logging.getLogger({NAME!r}).level",
        "seen = level_now()",
        "seen = here.LOGGER.level",
        "seen = max(map(level_of_item, [0]))",
        "seen = here.Holder.LOG.level",
        "seen = here.HOLDER.log.level",
        f"seen = logging.Logger.manager.loggerDict[{NAME!r}].level",
        f"seen = logging.Logger.manager.loggerDict.get({NAME!r}).level",
        f"seen = [lg.level for n, lg in logging.Logger.manager.loggerDict.items() if n == {NAME!r}][0]",
    ],
    ids=[
        "in-the-statement",
        "in-a-function",
        "a-module-attribute",
        "called-back-by-a-builtin",
        "a-class-attribute",
        "an-object-s-attribute",
        "the-registry-s-item",
        "the-registry-s-get",
        "the-registry-iterated",
    ],
)
def test_a_reader_looking_the_logger_up_runs_again(cash_magics, cash_instance, reader):
    """No input holds the logger: the statement, or a function it calls
    (from a builtin's code too), finds it by its name as it runs, or reads
    it through a module: an attribute of it, of a class or an object it
    holds, or an item of the registry of loggers, read by C."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [
        LOOKUP_SETUP,
        reader,
        "below = seen + 1",
        f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)",
        f"# again\n{reader}",
        "# again\nbelow = seen + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert cash_magics.shell.user_ns["below"] == 31


@needs_monitoring
@pytest.mark.parametrize(
    "setter",
    [f"logging.getLogger({NAME!r}).setLevel(10)", "set_level(10)", "done = set_level(10)"],
    ids=["in-the-statement", "in-a-function", "in-a-kept-call"],
)
def test_a_cell_setting_the_level_the_logger_had_runs_again(cash_magics, cash_instance, setter):
    """Run when the level was already 10, it changed nothing and was stored.
    The level then set to 30 elsewhere, a hit would leave it there."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [LOOKUP_SETUP, setter, f"seen = logging.getLogger({NAME!r}).level"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", cells=cells)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert logging.getLogger(NAME).level == 10
    assert cash_magics.shell.user_ns["seen"] == 10


#: More items than cash looks through for an object found by its name
#: (`handed_callables.ITEM_TYPES_AT`), the logger last.
BIG = [0] * 100_000 + [LOGGER]
#: Loggers each found by its own name.
MANY = [logging.getLogger(f"{NAME}.{i}") for i in range(20)] + [LOGGER]


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


@needs_monitoring
def test_a_container_too_big_to_look_through_is_never_served(cash_magics, cash_instance):
    """An item read by C from a container too big to look through: the
    statement runs every time, and a warning says why, rather than be
    served what it read before."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    reader = "seen = here.BIG[-1].level"
    cells = [LOOKUP_SETUP, reader, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", f"# again\n{reader}"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert "NOTEBOOK-ITEMS-UNREAD" in _codes(caught)


@needs_monitoring
def test_noting_past_the_budget_is_never_served(cash_magics, cash_instance, monkeypatch):
    """Checking the loggers a statement meets takes longer than its budget:
    it runs every time, and a warning says why."""
    from cash.notebook.statement.processor import StatementProcessor

    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    monkeypatch.setattr(StatementProcessor, "_noting_budget", staticmethod(lambda run: 0.0), raising=False)
    reader = "seen = here.MANY[-1].level"
    cells = [LOOKUP_SETUP, reader, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", f"# again\n{reader}"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert "NOTEBOOK-ANALYSIS-BUDGET" in _codes(caught)


#: A logger a library made, before any statement ran: found by the
#: statements below only through the library.
LIB_NAME = f"{NAME}.made_by_a_library"
logging.getLogger(LIB_NAME)


@pytest.fixture
def fresh_kernel(monkeypatch):
    """What a kernel that just started knows: no class seen pickled by its
    name, no library function seen returning one."""
    from cash import _weak, kept_state
    from cash.tracking import lookups_seen

    monkeypatch.setattr(kept_state, "_KINDS_BY_REFERENCE", set())
    monkeypatch.setattr(lookups_seen, "_RETURNERS", _weak.WeakSet())
    yield
    logging.getLogger(LIB_NAME).setLevel(logging.NOTSET)


FRESH_SETUP = "import logging, random\nclass Fake:\n    level = -1\ndef fake(n):\n    return Fake"


@needs_monitoring
@pytest.mark.usefixtures("fresh_kernel")
@pytest.mark.parametrize(
    "reader",
    [
        f"lv = [f({LIB_NAME!r}).level for f in (fake, logging.getLogger)][1]",
        f"lv = 0\nfor make in (fake, logging.getLogger):\n    lv = make({LIB_NAME!r}).level",
        f"lv = logging.Logger.manager.loggerDict[{LIB_NAME!r}].level",
        f"lv = [random.choice(s) for s in ([1], list(map(logging.getLogger, [{LIB_NAME!r}])))][1].level",
    ],
    ids=[
        "a-site-calling-another-first",
        "a-loop-calling-another-first",
        "the-registry",
        "a-return-giving-another-first",
    ],
)
def test_a_logger_a_fresh_kernel_finds_is_checked(cash_magics, cash_instance, reader):
    """In a kernel that saw no logger pickled yet, a statement finds one a
    library made: through a call made where another was called first, an
    item of the registry, or a return that gave another value first."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [FRESH_SETUP, reader, f"# elsewhere\nlogging.getLogger({LIB_NAME!r}).setLevel(40)", f"# again\n{reader}"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["lv"] == 40


@needs_monitoring
def test_a_kept_call_past_the_budget_is_never_served(cash_magics, cash_instance, monkeypatch):
    """The budget ran out before the kept calls of the statement started:
    what they find is unseen, so none is served once the level changed."""
    from cash.notebook.statement.processor import StatementProcessor

    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    setup = f"import logging\nlogging.getLogger({NAME!r}).setLevel(20)\ndef level_of(n):\n    return logging.getLogger(n).level"
    first = f"a = level_of('{NAME}.other')\nb = level_of({NAME!r})\nab = (level_of('{NAME}.other'), level_of({NAME!r}))"
    cells = [setup, first, f"logging.getLogger({NAME!r}).setLevel(40)", f"c = level_of({NAME!r}) + 0"]
    run_cash_cell(cash_magics, setup, cells=cells)
    with monkeypatch.context() as patched:
        patched.setattr(StatementProcessor, "_noting_budget", staticmethod(lambda run: 0.0), raising=False)
        run_cash_cell(cash_magics, first, cells=cells)
    for cell in cells[2:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["c"] == 40


def test_the_logger_s_attributes_are_all_described():
    """A standard library logger is keyed by the attributes its output
    follows (`kept_state.LOGGER_STATE`), its levels found enabled left out.
    A Python that adds one fails here, rather than key loggers without it."""
    from cash import kept_state

    described = set(kept_state.LOGGER_STATE) | set(kept_state.LOGGER_DERIVED)
    for logger in (logging.Logger(f"{NAME}.described"), logging.RootLogger(logging.WARNING)):
        assert set(vars(logger)) == described, sorted(set(vars(logger)) ^ described)
    assert "disable" in dir(logging.Logger.manager)


WORK_SETUP = f"import logging\nlog = logging.getLogger('{NAME}.work')\nlog.setLevel(logging.INFO)"


@pytest.mark.parametrize(
    "work",
    [
        "log.info('start')\nx = sum(range(10_000))",
        "def work(n):\n    log.info('working %s', n)\n    return sum(range(n))\nx = work(10_000)",
    ],
    ids=["the-cell-logs", "a-function-it-calls-logs"],
)
def test_work_that_logs_is_served_on_the_next_run_all(cash_magics, cash_instance, work):
    """Logging fills the logger's levels found enabled, and changes nothing
    else: run all again, the work is served."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [WORK_SETUP, work]
    try:
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    finally:
        logging.getLogger(f"{NAME}.work").setLevel(logging.NOTSET)
    # A ``def`` runs every time; what logs, and the work, are served.
    served = statuses[1:] if work.startswith("def ") else statuses
    assert served and all(status == "RESTORED" for status in served), statuses


def test_a_logger_s_digest_follows_what_its_output_does():
    """What changes a logger's output changes its digest; logging, which
    only notes levels found enabled, does not. A subclass defined elsewhere
    is keyed by all its attributes."""
    from cash import kept_state

    logger = logging.getLogger(f"{NAME}.digest")
    handler = logging.StreamHandler()
    try:
        before = kept_state.state_digest(logger)
        logger.info("a")
        logger.isEnabledFor(logging.DEBUG)
        assert kept_state.state_digest(logger) == before
        changes = [
            lambda: logger.setLevel(logging.ERROR),
            lambda: logger.addHandler(handler),
            lambda: handler.setLevel(logging.CRITICAL),
            lambda: handler.setFormatter(logging.Formatter("%(message)s")),
            lambda: setattr(logger, "propagate", False),
            lambda: logging.disable(logging.INFO),
        ]
        for change in changes:
            change()
            after = kept_state.state_digest(logger)
            assert after != before
            before = after
    finally:
        logging.disable(logging.NOTSET)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    class Mine(logging.Logger):
        pass

    logging.setLoggerClass(Mine)
    try:
        mine = logging.getLogger(f"{NAME}.mine")
    finally:
        logging.setLoggerClass(logging.Logger)
    mine.extra = 1
    held = kept_state.state_digest(mine)
    mine.extra = 2
    assert kept_state.state_digest(mine) != held


SWAP_SETUP = (
    "import logging, io\n"
    f"log = logging.getLogger('{NAME}.swap')\n"
    "log.setLevel(logging.INFO)\n"
    "log.propagate = False\n"
    "class Mine(logging.StreamHandler):\n"
    "    pass\n"
    "first = io.StringIO()\n"
    "second = io.StringIO()\n"
)


@pytest.mark.parametrize(
    ("handler", "swap"),
    [
        ("logging.StreamHandler(first)", "log.handlers[0].setStream(second)"),
        ("Mine(first)", "log.handlers[0].setStream(second)"),
    ],
    ids=["set-stream", "a-subclass-of-yours"],
)
def test_a_handler_set_to_write_elsewhere_is_written_to(cash_magics, cash_instance, handler, swap):
    """A handler given another stream (an ``io.StringIO``, which has no
    name) changes the logger's state: the statement logging after it runs,
    and the new stream gets the line."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    logs = "log.info('w%s', 5)\nx = 10"
    cells = [f"{SWAP_SETUP}log.addHandler({handler})", logs, swap, logs.replace("x", "y"), "out = second.getvalue()"]
    try:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    finally:
        # pytest gives each non-propagating logger its capture handlers.
        swapped = logging.getLogger(f"{NAME}.swap")
        swapped.handlers.clear()
        swapped.propagate = True
        swapped.setLevel(logging.NOTSET)
    assert cash_magics.shell.user_ns["out"] == "w5\n"


def test_a_logger_whose_handler_holds_each_record_says_why_it_runs_again(cash_magics, cash_instance):
    """A ``MemoryHandler`` holds each record logged, so a statement that
    logs changes the logger and runs every time: its reason names the
    handler."""
    from cash.tracking import reference_states

    logger = logging.getLogger(f"{NAME}.memory")
    held = logging.handlers.MemoryHandler(10_000)
    logger.addHandler(held)
    try:
        assert "MemoryHandler holds each record it logs" in reference_states._describe(logger)
    finally:
        logger.removeHandler(held)
    assert "holds each record" not in reference_states._describe(logger)


class Gathering(logging.StreamHandler):
    """A handler keeping each record it handles, as pytest's does."""

    def __init__(self, stream):
        super().__init__(stream)
        self.records = []

    def emit(self, record):
        self.records.append(record)
        super().emit(record)


class Shaded(logging.Formatter):
    """A formatter holding a mapping of its own."""

    def __init__(self):
        super().__init__("%(levelname)s %(message)s")
        self.shades = {logging.INFO: "plain"}


def test_a_handler_of_yours_is_keyed_by_what_it_writes_to_not_what_it_gathered():
    """A handler subclass that keeps each record (as pytest's does), with a
    formatter subclass holding a mapping, keeps the logger's digest while
    it logs; set to write to another stream, the digest changes."""
    import io

    from cash import kept_state

    logger = logging.getLogger(f"{NAME}.gathering")
    handler = Gathering(io.StringIO())
    handler.setFormatter(Shaded())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        before = kept_state.state_digest(logger)
        logger.info("a")
        assert handler.records and kept_state.state_digest(logger) == before
        handler.setStream(io.StringIO())
        assert kept_state.state_digest(logger) != before
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
