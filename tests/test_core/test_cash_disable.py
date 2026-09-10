"""``CASH_DISABLE=1`` runs everything uncached -- what a test suite needs.

CAS-121, round-17 tester r17s4 (F13, F20). A test that calls a cached kernel
twice with the same seed and asserts equality passed even when the kernel
ignored its seed: the second call was a hit, on a fresh cache too. There was
no supported way to run a suite uncached; the tester wrote a pytest plugin to
get one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from cash import Cash

pytestmark = pytest.mark.core


def _run(tmp_path, body, **env_extra):
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / ".cash")
    env.update(env_extra)
    return subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, cwd=str(tmp_path), env=env,
                          encoding="utf-8", errors="replace")


_JOB = """
    import time
    import cash

    calls = []

    @cash.cache(assume_safe=True)
    def work(n):
        calls.append(n)
        time.sleep(0.2)             # past the persistence floor
        return n * 2

    work(1); work(1); work(1)
    print("BODY", len(calls), work.explain(1).reason)
"""


def test_disabled_runs_the_body_every_time_and_writes_nothing(tmp_path):
    out = _run(tmp_path, _JOB, CASH_DISABLE="1", CASH_SUMMARY="1")
    assert "BODY 3 disabled" in out.stdout, out.stdout + out.stderr
    cache = tmp_path / ".cash"
    assert not cache.exists() or not any(cache.iterdir()), "something was cached"
    assert "caching disabled" in out.stderr and "3 calls ran uncached" in out.stderr


def test_unset_caches_as_usual(tmp_path):
    """The control: without the variable, the same script hits."""
    out = _run(tmp_path, _JOB)
    assert "BODY 1 " in out.stdout, out.stdout + out.stderr


def test_the_vacuous_determinism_test_is_caught(tmp_path):
    """The tester's case: a kernel that ignores its seed.

    Cached, "same seed gives the same answer" passes -- it compares one result
    with itself. Disabled, it fails, which is the point of running CI that way.
    """
    body = """
        import numpy as np
        import cash

        @cash.cache(allow_random=True)
        def simulate(n, seed=0):
            return float(np.random.default_rng().random())   # seed ignored

        print("EQUAL", simulate(3, seed=1) == simulate(3, seed=1))
    """
    pytest.importorskip("numpy")
    assert "EQUAL True" in _run(tmp_path, body).stdout
    assert "EQUAL False" in _run(tmp_path, body, CASH_DISABLE="1").stdout


def test_configure_flips_it_at_runtime(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    calls = []

    @c.cache(assume_safe=True)
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(1)
    assert len(calls) == 1
    c.config.disable = True                  # what cash.configure(disable=True) sets
    f(1)
    f(1)
    assert len(calls) == 3, "disable did not take effect on the next call"
    c.config.disable = False
    f(1)
    assert len(calls) == 3, "re-enabled, the stored entry is served again"


def test_cash_on_declines_when_disabled(cash_magics, capsys):
    cash_magics._cash_instance.config.disable = True
    cash_magics.cash_on("")
    assert not cash_magics._auto_cache_enabled
    assert "caching is disabled" in capsys.readouterr().out
