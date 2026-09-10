"""A callable built at import time is keyed by everything it runs WITH, not just its code.

Round 18 (r18s5): a module-level callable read inside a cached function, or
used as a parameter default, was keyed by its code alone. What it was built
with never reached the key:

* a factory closure's captured value (`CLIP = make_clipper(-3, 3)`);
* a `functools.partial`'s arguments, and -- as a default or an argument --
  the body of the function it wraps, which was reported as uncomputable
  (KEY-OPAQUE-CALLABLE) and whose only silencer, `mark_opaque(partial)`,
  silenced every partial in the process;
* a pre-built bound method's instance state (`F = S(2).f`).

Each shape is edited twice, in fresh processes on one cache: A changes the
value baked in (2 -> 3), B the body (`x * k` -> `x * k + 1`). Values [1, 2, 3]
give 12 before, 18 after A, 21 after B. The second (unedited) run must hit.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# Several fresh interpreters per case: past the suite-wide 30 s on a loaded runner.
pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SHAPES = {
    "lambda_reads_const": "K = {K}\nF = lambda x: {BODY}\n",
    "lambda_default": "F = lambda x, k={K}: {BODY}\n",
    "factory_lambda": "def make(k):\n    return lambda x: {BODY}\nF = make({K})\n",
    "partial": "import functools\ndef base(x, k):\n    return {BODY}\nF = functools.partial(base, k={K})\n",
    "bound_method": (
        "class S:\n    def __init__(self, k):\n        self.k = k\n"
        "    def f(self, x):\n        k = self.k\n        return {BODY}\nF = S({K}).f\n"
    ),
    "factory_closure": "def make(k):\n    def f(x):\n        return {BODY}\n    return f\nF = make({K})\n",
}

AS_GLOBAL = textwrap.dedent('''
    import sys, time
    import cash
    import cbmod

    @cash.cache
    def run(values):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return sum(cbmod.F(v) for v in values)

    print(run([1, 2, 3]))
''')

AS_DEFAULT = textwrap.dedent('''
    import sys, time
    import cash
    from cbmod import F

    @cash.cache
    def run(values, fn=F):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)  # @cash:assume-safe
        return sum(fn(v) for v in values)

    print(run([1, 2, 3]))
''')


def _module(template, k, body):
    if template.startswith("K ="):
        body = body.replace("k", "K")
    return template.format(K=k, BODY=body)


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr, p.stderr


@pytest.mark.parametrize("where", ["global", "default"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_value_a_callable_was_built_with_and_its_body_reach_the_key(tmp_path, shape, where):
    template = SHAPES[shape]
    (tmp_path / "job.py").write_text(AS_GLOBAL if where == "global" else AS_DEFAULT, encoding="utf-8")
    (tmp_path / "cbmod.py").write_text(_module(template, 2, "x * k"), encoding="utf-8")

    first, _, _ = _run(tmp_path)
    warm, warm_ran, warm_err = _run(tmp_path)
    assert first == warm == "12"
    assert not warm_ran, "the unedited second run did not hit"
    assert "KEY-OPAQUE-CALLABLE" not in warm_err

    (tmp_path / "cbmod.py").write_text(_module(template, 3, "x * k"), encoding="utf-8")
    assert _run(tmp_path)[0] == "18", "the value the callable was built with did not reach the key"

    (tmp_path / "cbmod.py").write_text(_module(template, 3, "x * k + 1"), encoding="utf-8")
    assert _run(tmp_path)[0] == "21", "the callable's body did not reach the key"


def test_mark_opaque_partial_no_longer_silences_every_partial(tmp_path):
    """The old advice for KEY-OPAQUE-CALLABLE was `cash.mark_opaque(functools.partial)`.
    It exempted every partial in the process, so one over a function the user
    then edited was served stale, silently. A partial is keyed by its function now,
    and the mark cannot switch that off."""
    (tmp_path / "cbmod.py").write_text(
        "import functools\ndef base(x, k):\n    return x * k\nF = functools.partial(base, k=2)\n",
        encoding="utf-8",
    )
    job = AS_DEFAULT.replace("import cash\n", "import cash, functools\ncash.mark_opaque(functools.partial)\n")
    (tmp_path / "job.py").write_text(job, encoding="utf-8")
    assert _run(tmp_path)[0] == "12"
    (tmp_path / "cbmod.py").write_text(
        "import functools\ndef base(x, k):\n    return x * k + 1\nF = functools.partial(base, k=2)\n",
        encoding="utf-8",
    )
    assert _run(tmp_path)[0] == "15"
