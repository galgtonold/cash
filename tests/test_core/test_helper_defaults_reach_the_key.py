"""A helper's parameter defaults reach its caller's key.

CAS-112, round-17 tester r17s3: 8 wrong answers in 8 from a service whose
ridge penalty was a helper's DEFAULT:

    # helpers.py
    ALPHA = 1.0
    def shrink(v, alpha=ALPHA):      # the constant is a default, not a read
        return v / (1 + alpha)

    @cash.cache
    def score(v): return shrink(v)   # edit ALPHA -> old answer

A default is evaluated at ``def`` time and lives on the function object. The
helper's source text does not change when ALPHA does, and global folding
looks at names the body reads -- so nothing saw it. The cached function's own
defaults were already keyed; helpers' were not.

Fresh process per run, bytecode caching off.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

MAIN = textwrap.dedent('''
    import json, sys, time
    import cash
    from helpers import shrink, scaled

    @cash.cache
    def score(v):
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.25)
        return [shrink(v), scaled(v)]

    print(json.dumps(score(100.0)))
''')


def _project(tmp_path, alpha, factor=2.0):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "constants.py").write_text(f"FACTOR = {factor}\n", encoding="utf-8")
    (proj / "helpers.py").write_text(textwrap.dedent(f"""
        from constants import FACTOR

        ALPHA = {alpha}

        def shrink(v, alpha=ALPHA):          # a default bound to a module constant
            return v / (1 + alpha)

        def _make(k):
            def scaled(v, k=k):              # a closure's default, set by a factory
                return v * k
            return scaled

        scaled = _make(FACTOR)               # default bound to an IMPORTED name
    """), encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    return proj


def _run(tmp_path, proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out = subprocess.run([sys.executable, "main.py"], cwd=str(proj),
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1]), "RAN" in out.stderr


def test_a_helper_default_bound_to_a_constant_invalidates(tmp_path):
    """THE BUG: ALPHA 1.0 -> 3.0 served the old answer."""
    proj = _project(tmp_path, alpha=1.0)
    first, _ = _run(tmp_path, proj)
    assert first[0] == 50.0

    _project(tmp_path, alpha=3.0)
    second, ran = _run(tmp_path, proj)

    assert second[0] == 25.0, "a helper's default was not in the key"
    assert ran


def test_a_factory_default_from_another_module_invalidates(tmp_path):
    """The default arrives through an import and a closure factory."""
    proj = _project(tmp_path, alpha=1.0, factor=2.0)
    first, _ = _run(tmp_path, proj)
    assert first[1] == 200.0

    _project(tmp_path, alpha=1.0, factor=5.0)
    second, ran = _run(tmp_path, proj)

    assert second[1] == 500.0
    assert ran


def test_unchanged_defaults_still_hit(tmp_path):
    """The control: folding defaults must not cost the hit."""
    proj = _project(tmp_path, alpha=1.0)
    _run(tmp_path, proj)
    again, ran = _run(tmp_path, proj)

    assert again == [50.0, 200.0]
    assert not ran, "unchanged helper defaults recomputed"
