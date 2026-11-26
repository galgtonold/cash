"""``use_locking`` is a setting like any other.

Compute-once locking was only a ``Cash(...)`` argument: a service that
writes ``@cash.cache``, as the docs show, could not turn it on without
building an instance and rewriting every decorator. It is now read from
code (``cash.configure``, ``Cash(...)``), ``CASH_USE_LOCKING`` and
``[tool.cash]``, on every call.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import cash
from cash import Cash
from tests._scripts import run_python

N = 4

SCRIPT = """
import os, threading, sys
from concurrent.futures import ThreadPoolExecutor
import cash
{setup}
LOG = os.path.join(os.getcwd(), "runs.log")
barrier = threading.Barrier({n})
entered = threading.Barrier({n})

@cash.cache
def paid_call(x):
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    try:
        entered.wait({hold})   # still running when the other callers arrive
    except threading.BrokenBarrierError:
        pass
    return x

def call(_):
    barrier.wait()
    return paid_call(1)

with ThreadPoolExecutor({n}) as ex:
    assert list(ex.map(call, range({n}))) == [1] * {n}
print("RUNS", os.path.getsize(LOG))
"""


def _runs(tmp_path, setup="", env=None, cache_dir=..., hold=0.5):
    """How many callers ran the body. Each one in the body holds it until
    all ``N`` are in, or *hold* seconds pass: one who is let in only after
    the value is stored is served it, as it should be."""
    script = SCRIPT.format(setup=setup, n=N, hold=hold)
    (tmp_path / "main.py").write_text(script, encoding="utf-8")
    kwargs = {} if cache_dir is ... else {"cache_dir": cache_dir}
    out = run_python("main.py", cwd=tmp_path, env=env, **kwargs).stdout
    return int(out.split("RUNS")[1])


@pytest.mark.timeout(120)
def test_without_the_setting_every_concurrent_caller_computes(tmp_path):
    """The control: the script does overlap its callers. Every caller waits
    in the body for all of them, so none can arrive after the first one has
    stored however slowly a loaded runner schedules it; a cash that let
    only some in would hold the body half a minute and count fewer."""
    assert _runs(tmp_path, hold=30) == N


@pytest.mark.timeout(120)
def test_configure_turns_it_on_for_the_default_instance(tmp_path):
    assert _runs(tmp_path, setup="cash.configure(use_locking=True)") == 1


@pytest.mark.timeout(120)
def test_the_environment_variable_turns_it_on(tmp_path):
    assert _runs(tmp_path, env={"CASH_USE_LOCKING": "1"}) == 1


@pytest.mark.timeout(120)
def test_pyproject_turns_it_on(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.cash]\nuse_locking = true\n", encoding="utf-8")
    assert _runs(tmp_path, cache_dir=None) == 1


def _overlapping_calls(app: Cash) -> int:
    runs = []
    barrier = threading.Barrier(N)
    release = threading.Event()

    @app.cache
    def body(x):
        runs.append(x)
        release.wait(0.5)
        return x

    def call(_):
        barrier.wait()
        return body(1)

    with ThreadPoolExecutor(N) as ex:
        assert list(ex.map(call, range(N))) == [1] * N
    return len(runs)


@pytest.mark.timeout(60)
def test_the_constructor_argument_is_the_setting(tmp_path):
    app = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, use_locking=True)
    assert app.config.use_locking is True and app.use_locking is True
    assert _overlapping_calls(app) == 1


@pytest.mark.timeout(60)
def test_reconfigure_switches_a_running_instance(tmp_path):
    """Read on every call: a function decorated before the switch follows it."""
    app = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    assert app.use_locking is False
    app.reconfigure(use_locking=True)
    assert app.use_locking is True
    assert _overlapping_calls(app) == 1
    app.use_locking = False
    assert app.config.use_locking is False


def test_a_value_that_is_not_a_flag_is_refused(tmp_path):
    """Checked like every boolean setting: a flag, or a string an
    environment variable could hold (``"no"`` is off)."""
    app = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    with pytest.raises(ValueError, match="use_locking"):
        app.reconfigure(use_locking="sometimes")
    with pytest.raises(ValueError, match="use_locking"):
        cash.configure(use_locking=2)
    app.reconfigure(use_locking="no")
    assert app.use_locking is False
