"""``cash.disabled()`` and ``cash.configure(...)`` reach the pool workers
started after them.

The testing guide isolates a suite with ``cash.configure(cache_dir=tmp)``
and switches the cache off with ``with cash.disabled():``. Both changed only
the process that called them: a spawned worker (the default start method on
Windows and macOS, and forkserver on Linux since 3.14) built its settings
from the environment and the config files again. Under ``cash.disabled()`` a
pooled seed-ignoring function returned the value stored earlier, twice, so
the determinism test the guide warns about passed on broken code; under
``configure(cache_dir=tmp)`` the workers read the project's cache. Only
``CASH_DISABLE`` / ``CASH_CACHE_DIR`` in the environment reached them.

Each case runs a real script in a fresh interpreter, so the pool starts from
a parent in the state a user's program is in. Executions are counted with
``os.write`` to a file, which a cache hit cannot replay.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

_SCRIPT = textwrap.dedent("""
    import json, os, sys, time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import cash


    def runs():
        try:
            return os.path.getsize(os.environ["COUNTER"])
        except FileNotFoundError:
            return 0


    @cash.cache(assume_safe=True)
    def draw(n):
        fd = os.open(os.environ["COUNTER"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.write(fd, b"x")
        os.close(fd)
        time.sleep(0.15)                      # past the persistence floor
        return os.urandom(8).hex()            # the bug under test: n is ignored


    def pooled(ctx, kind, args):
        if kind == "executor":
            with ProcessPoolExecutor(1, mp_context=ctx) as ex:
                return list(ex.map(draw, args))
        if kind == "joblib":
            from joblib import Parallel, delayed
            # n_jobs=1 would run the calls in this process.
            return Parallel(n_jobs=2, backend="loky")(delayed(draw)(a) for a in args)
        with ctx.Pool(1) as pool:
            return pool.map(draw, args)


    def phase(ctx, kind, args):
        before = runs()
        values = pooled(ctx, kind, args)
        return runs() - before, values


    if __name__ == "__main__":
        method, kind, scenario = sys.argv[1:4]
        ctx = mp.get_context(method)
        out = {}
        if scenario == "disabled":
            # A warm entry first -- through the pool, so a forkserver is
            # already running before any setting changes.
            _, (warm,) = phase(ctx, kind, [3])
            with cash.disabled():
                out["inside"], values = phase(ctx, kind, [3, 3])
            out["inside_distinct"] = len({warm, *values}) == 3
            out["after"], (again,) = phase(ctx, kind, [3])
            out["after_is_warm"] = again == warm
        elif scenario == "cache_dir":
            _, (warm,) = phase(ctx, kind, [3])
            cash.configure(cache_dir=os.environ["OTHER_DIR"])
            out["moved"], values = phase(ctx, kind, [3, 3])
            out["moved_fresh"] = values[0] != warm and values[0] == values[1]
            other = os.environ["OTHER_DIR"]
            out["other_used"] = os.path.isdir(other) and any(os.scandir(other))
        elif scenario == "disabled_first":
            # The pool's first use is inside the block (joblib keeps its
            # workers between calls, so only workers it starts there count).
            warm = draw(3)
            with cash.disabled():
                out["inside"], values = phase(ctx, kind, [3, 3])
            out["inside_distinct"] = len({warm, *values}) == 3
        elif scenario == "enabled":
            # CASH_DISABLE=1 in the environment, caching forced back on.
            with cash.disabled(False):
                out["forced_on"], values = phase(ctx, kind, [3, 3])
            out["forced_on_same"] = values[0] == values[1]
            out["after"], _ = phase(ctx, kind, [3, 3])
        print("RESULT", json.dumps(out))
""")


def _run(tmp_path, method, kind, scenario, **env_extra):
    script = tmp_path / "job.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(
        CASH_CACHE_DIR=str(tmp_path / ".cash"),
        COUNTER=str(tmp_path / "runs.bin"),
        OTHER_DIR=str(tmp_path / "other-cache"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.update(env_extra)
    run = subprocess.run(
        [sys.executable, str(script), method, kind, scenario],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    line = next((ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, run.stdout + run.stderr
    return json.loads(line[len("RESULT ") :])


def _methods():
    available = multiprocessing.get_all_start_methods()
    cases = [("spawn", "executor"), ("spawn", "pool")]
    if "forkserver" in available:
        cases.append(("forkserver", "executor"))
    if sys.platform.startswith("linux"):
        cases.append(("fork", "pool"))  # already inherited the parent's memory; must stay right
    return cases


@pytest.mark.parametrize(("method", "kind"), _methods())
def test_disabled_reaches_the_workers_and_ends_with_the_block(tmp_path, method, kind):
    out = _run(tmp_path, method, kind, "disabled")
    assert out["inside"] == 2 and out["inside_distinct"], (
        f"under cash.disabled() a {method} worker served the cached value: {out}"
    )
    assert out["after"] == 0 and out["after_is_warm"], f"workers stayed uncached after the block: {out}"


@pytest.mark.parametrize(("method", "kind"), _methods())
def test_configured_cache_dir_is_the_workers_cache_dir(tmp_path, method, kind):
    out = _run(tmp_path, method, kind, "cache_dir")
    assert out["moved"] == 1 and out["moved_fresh"] and out["other_used"], (
        f"a {method} worker kept using the project's cache after cash.configure(cache_dir=...): {out}"
    )


def test_disabled_false_turns_the_cache_on_in_workers_of_a_disabled_run(tmp_path):
    out = _run(tmp_path, "spawn", "executor", "enabled", CASH_DISABLE="1")
    assert out["forced_on"] == 1 and out["forced_on_same"], out
    assert out["after"] == 2, f"the environment's CASH_DISABLE=1 did not come back after the block: {out}"


def test_a_joblib_worker_started_under_disabled_runs_uncached(tmp_path):
    pytest.importorskip("joblib")
    out = _run(tmp_path, "spawn", "joblib", "disabled_first")
    assert out["inside"] == 2 and out["inside_distinct"], out


# --- What the parent hands on, in this process ---------------------------------


@pytest.fixture
def fresh_default():
    import cash

    cash.reset_session()
    yield cash
    cash.reset_session()


def _handed_on():
    from cash import _active

    return dict(multiprocessing.current_process()._config.get(_active._SETTINGS_KEY) or {})


def test_nested_disabled_blocks_hand_on_the_setting_in_force(fresh_default):
    cash = fresh_default
    with cash.disabled():
        assert _handed_on()["disable"] is True
        with cash.disabled(False):
            assert _handed_on()["disable"] is False
        assert _handed_on()["disable"] is True
    assert _handed_on()["disable"] is False


def test_a_relative_cache_dir_is_handed_on_as_absolute(fresh_default, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh_default.configure(cache_dir="rel")
    assert _handed_on()["cache_dir"] == os.path.join(os.getcwd(), "rel")


def test_reset_session_stops_handing_settings_on(fresh_default, tmp_path):
    fresh_default.configure(cache_dir=str(tmp_path))
    fresh_default.reset_session()
    assert _handed_on() == {}


@pytest.mark.parametrize("built_before", [True, False], ids=["default-built-first", "default-built-after"])
def test_unpickling_what_was_handed_on_applies_it(fresh_default, tmp_path, built_before):
    """What a spawned child does with the process object it is sent: its
    default may exist already (the main script it imported used cash) or be
    built later (the task imports cash)."""
    import pickle

    from cash import _active

    cash = fresh_default
    cash.configure(cache_dir=str(tmp_path / "parent"), disable=True)
    payload = pickle.dumps(multiprocessing.current_process()._config[_active._SETTINGS_KEY])
    cash.reset_session()  # the child's own default, from the environment and files
    if built_before:
        child = cash._get_global_cash()
        assert not child.config.disable
        pickle.loads(payload)
    else:
        pickle.loads(payload)
        child = cash._get_global_cash()
    assert child.config.disable is True
    assert child.config.cache_dir == str(tmp_path / "parent")
