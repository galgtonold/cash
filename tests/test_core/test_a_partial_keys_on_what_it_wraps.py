"""A cached ``functools.partial`` keys on what it wraps, not on an address.

Found while attacking the decorator before round 26: `cash.cache(partial(slow,
1))` took its cache namespace from `repr(partial)`, which holds the wrapped
function's memory address -- so every process wrote a fresh namespace and none
of them ever hit. (`docs/decorator.md` promises these "degrade to a stable
identity-based fallback", which held for builtins but not for a partial over a
Python function.) The doubled `functools.functools.` in the name came from the
same repr.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent('''
    import functools, json, time
    import cash
    cash.configure(cache_dir=CACHE)

    def slow(a, b):
        time.sleep(0.3)
        return a + b

    cached = cash.cache(functools.partial(slow, 1))
    value = cached(2)
    print(json.dumps({"value": value, "misses": cached.cache_info()["misses"]}))
''')


@pytest.mark.timeout(300)
def test_a_second_process_hits(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")

    def run():
        import json
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                              timeout=180, cwd=str(tmp_path))
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert run() == {"value": 3, "misses": 1}
    assert run() == {"value": 3, "misses": 0}, "the second process missed again"


def test_the_name_does_not_carry_an_address():
    from cash.core import Cash

    name = Cash._get_func_key(functools.partial(test_the_name_does_not_carry_an_address, 1))
    assert "0x" not in name, name
    assert "functools.functools" not in name, name


def test_two_partials_of_one_function_are_two_namespaces():
    from cash.core import Cash

    def base(a, b):
        return a + b

    one = Cash._get_func_key(functools.partial(base, 1))
    two = Cash._get_func_key(functools.partial(base, 2))
    assert one != two, one
