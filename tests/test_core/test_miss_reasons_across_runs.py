"""A fresh process says why its first call missed, from what earlier runs stored.

Round 18, all five testers: in a script run every first miss read "no entry
yet: ... no earlier run left one on disk" -- after a code edit, after a TTL
expiry, after new arguments -- while the earlier run's entry was on disk. The
reasons only knew this process's history. Each persisted store now records its
key beside the cache (one small file per function), and a miss with no
in-process history reads it: the same key stored before is "ttl expired" or
"entry gone"; another key of the same function says which part moved.

Every step is a fresh process on one cache; the child prints its
`cache_info()["miss_reasons"]`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = textwrap.dedent('''
    import json, sys, time
    import cash

    TTL = {ttl}

    @cash.cache(ttl=TTL)
    def f(x):
        time.sleep(0.2)  # @cash:assume-safe
        return x * {factor}

    arg = int(sys.argv[1])
    explanation = f.explain(arg)
    f(arg)
    print(json.dumps({{"reasons": f.cache_info()["miss_reasons"],
                       "explain": explanation.details.get("why", explanation.reason)}}))
''')


def _write(proj, factor=2, ttl=None):
    (proj / "job.py").write_text(JOB.format(factor=factor, ttl=ttl), encoding="utf-8")


def _run(proj, arg=1):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py", str(arg)], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_first_run_has_no_earlier_entry(tmp_path):
    _write(tmp_path)
    out = _run(tmp_path)
    assert out["reasons"] == {"no entry yet": 1}


def test_a_code_edit_is_named_in_the_next_process(tmp_path):
    _write(tmp_path, factor=2)
    _run(tmp_path)
    _write(tmp_path, factor=3)
    out = _run(tmp_path)
    assert out["reasons"] == {"code or state changed": 1}, out
    assert "earlier run" in out["explain"]


def test_new_arguments_are_named_in_the_next_process(tmp_path):
    _write(tmp_path)
    _run(tmp_path, arg=1)
    out = _run(tmp_path, arg=2)
    assert out["reasons"] == {"new arguments": 1}, out


def test_a_ttl_expiry_is_named_in_the_next_process(tmp_path):
    """r18s4's shape: the backend drops an expired entry on read, so the lookup
    alone sees "absent"."""
    _write(tmp_path, ttl=1)
    _run(tmp_path)
    time.sleep(1.5)
    out = _run(tmp_path)
    assert out["reasons"] == {"ttl expired": 1}, out
    assert "ttl=1" in out["explain"]


def test_a_cleared_entry_is_named_as_gone(tmp_path):
    _write(tmp_path)
    _run(tmp_path)
    for entry in (tmp_path / ".cash").glob("*.entry"):
        entry.unlink()
    out = _run(tmp_path)
    assert out["reasons"] == {"entry gone": 1}, out


def test_a_warm_run_still_hits(tmp_path):
    """Control: recording keys changes nothing about hitting."""
    _write(tmp_path)
    _run(tmp_path)
    out = _run(tmp_path)
    assert out["reasons"] == {}, out
