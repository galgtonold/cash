"""One-line clock and environment helpers, reached through this module.

A real importable module (not collected by pytest), so a cached function in a
test can call them as ``_clock_helpers_fixture.now()``.
"""

import os
import time

ENV_NAME = "CASH_TEST_APP_MODE"


def now():
    return time.time()


def env(name):
    return os.environ.get(name)


def mode():
    return os.environ.get(ENV_NAME)


class Clock:
    @staticmethod
    def static():
        return time.time()
