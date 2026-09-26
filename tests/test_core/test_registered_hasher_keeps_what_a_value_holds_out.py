"""A value a registered hasher keys is not searched for code.

``cash.register_hasher(logging.Logger, lambda log: "any-logger")`` is how the
docs keep a logger out of the key, and the value channel did use the hasher.
The code channel still searched the logger, and a logger holds its manager,
so it reached every logger, handler, filter and stream in the process. A
handler that held a bound builtin (``self.write = stream.write``) made every
call warn that code it reached was not in the key, and one that held a
function of the user's put that function in the key of every call taking a
logger. In a Jupyter kernel, ``logging.basicConfig()`` alone was enough to
set off the warning.
"""

from __future__ import annotations

import io
import logging
import uuid
import warnings

import pytest

from cash import Cash, FileBackend


def upper(text):
    return text.upper()


def lower(text):
    return text.lower()


class Echo(logging.Handler):
    """A handler of the user's own that holds a callable."""

    def __init__(self, write):
        super().__init__()
        self.write = write

    def emit(self, record):
        self.write(self.format(record))


@pytest.fixture
def logger():
    log = logging.getLogger(f"cash-test.{uuid.uuid4().hex}")
    yield log
    for handler in log.handlers[:]:
        log.removeHandler(handler)


def _cash(tmp_path, *, register):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    if register:
        c.register_hasher(logging.Logger, lambda log: "any-logger")
    return c


def _opaque_warnings(caught):
    return [str(w.message) for w in caught if "KEY-OPAQUE-CALLABLE" in str(w.message)]


def test_a_handler_holding_a_builtin_does_not_warn_through_a_registered_logger(tmp_path, logger):
    logger.addHandler(Echo(io.StringIO().write))
    c = _cash(tmp_path, register=True)

    @c.cache
    def fit_logged(data, log):
        log.info("fitting %d rows", len(data))
        return sum(data) / len(data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_logged([1, 2, 3], logger)
        fit_logged([1, 2, 3], logging.getLogger("cash-test.other"))

    assert _opaque_warnings(caught) == []
    assert fit_logged.cache_info()["hits"] == 1


def test_a_registered_logger_inside_an_argument_is_not_searched_either(tmp_path, logger):
    logger.addHandler(Echo(io.StringIO().write))
    c = _cash(tmp_path, register=True)

    class Job:
        def __init__(self, log):
            self.log = log

    @c.cache
    def run(job):
        job.log.info("running")
        return 1

    @c.cache
    def run_all(logs):
        return len(logs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(Job(logger))
        run_all([logger])

    assert _opaque_warnings(caught) == []


@pytest.mark.parametrize(
    ("register", "hits"),
    [(True, 1), (False, 0)],
    ids=["registered-logger-hits", "control-unregistered-logger-misses"],
)
def test_code_a_registered_value_holds_is_not_in_the_key(tmp_path, logger, register, hits):
    """With the hasher, swapping what the logger's handler holds is not a new
    call. The control: without it, the handler's function is reached and is
    in the key, so the same swap misses."""
    handler = Echo(upper)
    logger.addHandler(handler)
    c = _cash(tmp_path, register=register)

    @c.cache
    def fit_logged(data, log):
        log.info("fitting %d rows", len(data))
        return sum(data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_logged([1, 2, 3], logger)
        handler.write = lower
        fit_logged([1, 2, 3], logger)

    assert fit_logged.cache_info()["hits"] == hits
