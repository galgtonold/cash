"""A cached function that logs through a module-level logger, in a real kernel.

With ``logging.basicConfig()`` the root handler writes to ipykernel's output
stream, and that stream holds the thread that watches the process's output
file descriptors. cash counted the bound methods of that thread's lock as user
code it could not hash, so every call of such a function warned
KEY-OPAQUE-CALLABLE about ``lock.acquire`` and ``lock.release``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(90)]

_SETUP = (
    "import logging, warnings\n"
    "import cash\n"
    "logging.basicConfig(level=logging.INFO)\n"
    "log = logging.getLogger('work')\n"
    "\n"
    "@cash.cache\n"
    "def step(n):\n"
    "    log.info('step %d', n)\n"
    "    return n + 1\n"
)
_CALLS = (
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    step(1)\n"
    "    step(1)\n"
    "OPAQUE = [str(w.message)[:120] for w in caught if 'KEY-OPAQUE-CALLABLE' in str(w.message)]\n"
    "HITS = step.cache_info()['hits']\n"
    "WATCHED = logging.getLogger().handlers[0].stream._should_watch\n"
)


def test_basic_config_and_a_module_logger_do_not_warn(nb_runner, monkeypatch):
    # ipykernel does not watch the output descriptors when it sees
    # PYTEST_CURRENT_TEST, which a kernel started from a test inherits; a
    # kernel Jupyter starts does watch them, and that is the shape to test.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    nb_runner.create_notebook([_SETUP, _CALLS])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()

    assert nb_runner.peek("WATCHED") == "True", "premise: the kernel's stream watches its descriptors"
    assert nb_runner.peek("OPAQUE") == "[]"
    assert nb_runner.peek("HITS") == "1"
