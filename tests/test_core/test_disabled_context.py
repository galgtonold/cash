"""`cash.disabled()` switches caching off for a block and puts back what was there.

Round 19 (r19s3): the testing guide's ``no_cache`` fixture ended with
``cash.configure(disable=False)``. Under ``CASH_DISABLE=1`` -- the job meant
to run with no cache -- that switched caching ON from the first test that used
the fixture, and every later test read and wrote the cache.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import cash

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


@pytest.fixture
def restore_disable():
    c = cash._get_global_cash()
    before = c.config.disable
    yield c
    cash.configure(disable=before)


@pytest.mark.parametrize("start", [False, True])
def test_the_setting_in_force_before_the_block_is_the_one_after_it(restore_disable, start):
    c = restore_disable
    cash.configure(disable=start)
    with cash.disabled():
        assert c.config.disable is True
    assert c.config.disable is start
    with cash.disabled(False):
        assert c.config.disable is False
    assert c.config.disable is start


def test_it_is_restored_when_the_block_raises(restore_disable):
    c = restore_disable
    cash.configure(disable=False)
    with pytest.raises(RuntimeError), cash.disabled():
        raise RuntimeError("boom")
    assert c.config.disable is False


CONFTEST = '''\
import cash
import pytest

@pytest.fixture
def no_cache():
    with cash.disabled():
        yield
'''

TEST_A = '''\
def test_a(no_cache):
    assert True
'''

TEST_B = '''\
import cash

calls = []

@cash.cache
def f(x):
    calls.append(x)
    return x * 2

def test_b_runs_uncached():
    f(1)
    f(1)
    assert calls == [1, 1], "CASH_DISABLE=1 was switched off by the fixture"
'''


def test_the_documented_fixture_leaves_a_cash_disable_run_uncached(tmp_path):
    for name, text in (("conftest.py", CONFTEST), ("test_a.py", TEST_A), ("test_b.py", TEST_B)):
        (tmp_path / name).write_text(textwrap.dedent(text), encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CASH_", "PYTEST_XDIST"))}
    env.update(CASH_DISABLE="1", CASH_CACHE_DIR=str(tmp_path / ".cash"))
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-p", "no:randomly", "test_a.py", "test_b.py"],
                       cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    written = [f for _, _, files in os.walk(tmp_path / ".cash") for f in files if f.endswith(".entry")]
    assert not written, f"a CASH_DISABLE=1 run wrote {written}"
