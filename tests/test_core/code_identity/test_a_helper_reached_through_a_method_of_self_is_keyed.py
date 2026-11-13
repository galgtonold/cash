"""A helper a method reaches through another method of ``self`` is keyed.

A cached function is handed an object (as ``self`` of a decorated method, as
the instance of a bound method passed to ``cash.cache``, as an argument, or
reads one from a module global), and calls ``obj.fit(x)``, which calls
``self.step(x)``, which calls ``H.scale(x)`` in a module of the project.

The object's code is keyed by what its methods load by name: ``H``, never
``H.scale``. On Python 3.12+ the trace of what the call ran names
``scale`` and checks it; before 3.12, or with ``CASH_TRACE=0``, nothing did,
and an edit to ``scale`` was served the result from before it. Without a
trace the helpers the purity walk of each method reaches are in the key.

Fresh interpreters on one cache: a run, a run that hits, an edit to the
helper that must recompute.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

HELPERS = "def scale(x):\n    return x * {K}\n"

MODEL = textwrap.dedent("""
    import helpers as H


    class Model:
        def __init__(self, bias):
            self.bias = bias

        def step(self, x):
            return H.scale(x) + self.bias

        def fit(self, x):
            return self.step(x) + 1
""")

SHAPES = {
    "decorated-method": """
        class Est(Model):
            @cash.cache
            def run(self, x):
                print("[RUN]", file=sys.stderr)  # @cash:assume-safe
                return self.fit(x)

        print(Est(5).run(3))
    """,
    "bound-method": """
        class Est(Model):
            def run(self, x):
                print("[RUN]", file=sys.stderr)  # @cash:assume-safe
                return self.fit(x)

        print(cash.cache(Est(5).run)(3))
    """,
    "argument": """
        @cash.cache
        def run(model, x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return model.fit(x)

        print(run(Model(5), 3))
    """,
    "data-global": """
        MODELS = [Model(5)]

        @cash.cache
        def run(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return MODELS[0].fit(x)

        print(run(3))
    """,
}


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_edit_to_a_helper_a_method_of_self_reaches_recomputes(tmp_path, shape, trace):
    script = "import sys\nimport cash\nfrom model import Model\n" + textwrap.dedent(SHAPES[shape])
    (tmp_path / "main.py").write_text(script, encoding="utf-8")
    (tmp_path / "model.py").write_text(MODEL, encoding="utf-8")
    env = {"CASH_TRACE": trace}

    def run(k):
        (tmp_path / "helpers.py").write_text(HELPERS.format(K=k), encoding="utf-8")
        proc = run_python("main.py", cwd=tmp_path, env=env)
        return proc.stdout.strip(), "[RUN]" in proc.stderr

    assert run(2) == ("12", True)
    assert run(2) == ("12", False), "an unedited run must hit"
    out, ran = run(10)
    assert out == "36", f"served the result from before the edit to the helper: {out}"
    assert ran
