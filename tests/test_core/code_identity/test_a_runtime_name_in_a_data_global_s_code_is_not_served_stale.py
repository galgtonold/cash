"""Code a data global carries that picks code by a run-time name is never
served stale.

A cached function reads ``HANDLERS["a"].run(y)``, and ``run`` -- a method of
an object or class a module global holds -- calls ``getattr(MOD, name)()``.
The key cannot name the function the name picks (KEY-DYNAMIC-DEPENDENCY), so
the call is served only on a trace of what it ran (Python 3.12+, where a
lookup by name is recorded even with ``CASH_TRACE=0``) or runs uncached.

The lookup was noted while the global's code was folded into the key. On
3.12+ a fold of the same helper is remembered and replayed on the next call
(`helper_fold_memo`), and the replay left the lookup out: a later call
was stored with no trace, and served after an edit to the function picked.

Fresh interpreters on one cache: five calls, an edit, the same five calls.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

SCRIPT = textwrap.dedent("""
    import importlib
    import sys

    import cash
    import other as MOD


    class Handler:
        name = "fun" + "1"

        @classmethod
        def run(cls, y):
            return y + {LOOKUP}


    HANDLERS = {"a": {HELD}}


    def dispatch(y):
        return HANDLERS["a"].run(y)


    @cash.cache
    def f(y):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return dispatch(y)


    print(*[f(y) for y in range(2, 7)])
""")

LOOKUPS = {
    "getattr": "getattr(MOD, cls.name)()",
    "import_module": "importlib.import_module('oth' + 'er').fun1()",
}


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("held", ["Handler", "Handler()"], ids=["class", "instance"])
@pytest.mark.parametrize("lookup", sorted(LOOKUPS))
def test_an_edit_to_the_function_a_held_object_picks_by_name_recomputes(tmp_path, trace, held, lookup):
    (tmp_path / "main.py").write_text(
        SCRIPT.replace("{LOOKUP}", LOOKUPS[lookup]).replace("{HELD}", held), encoding="utf-8"
    )
    env = {"PYTHONWARNINGS": "always", "CASH_TRACE": trace}

    def run(ret):
        (tmp_path / "other.py").write_text(f"def fun1():\n    return {ret}\n", encoding="utf-8")
        proc = run_python("main.py", cwd=tmp_path, env=env)
        return proc.stdout.strip(), proc.stderr

    out, err = run(10)
    assert out == "12 13 14 15 16"
    assert "KEY-DYNAMIC-DEPENDENCY" in err, err
    out, err = run(20)
    assert out == "22 23 24 25 26", f"served the result from before the edit: {out}"
