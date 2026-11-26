"""A helper that code handed to a call reaches is checked even when nothing
recorded the body.

A cached function is handed an object and calls ``model.fit(x)``, which calls
``H.scale(x)`` in a module of the project. The object's code is keyed by what
its methods load by name: ``H``, never ``H.scale``. For a recorded call the
record of what ran names ``scale`` and checks it before the entry is served.

Some bodies leave no record: a generator's runs after the call, as the caller
iterates; so does the body of a generator expression a plain function
returns; and another tool may hold the monitoring ids. Each of them was
stored by its key alone, and an edit to ``scale`` was served the result from
before it. Now a generator's key holds those helpers, and an entry stored
without a record holds a digest of them that every lookup builds again; where
they cannot be read, the call is not stored and says so.

Fresh interpreters on one cache: a run, a second run, an edit to the helper.
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

        def fit(self, x):
            return H.scale(x) + self.bias
""")

HEAD = "import sys\nimport cash\nfrom model import Model\n"

#: shape -> (script, is a second run served?)
SHAPES = {
    "generator-of-an-argument": (
        """
        @cash.cache
        def run(model, xs):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            for x in xs:
                yield model.fit(x)

        print(list(run(Model(5), (3, 4))))
        """,
        True,
    ),
    "returned-iterator-of-an-argument": (
        """
        def note():
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe

        @cash.cache
        def run(model, xs):
            note()
            return (model.fit(x) for x in xs)

        print(list(run(Model(5), (3, 4))))
        """,
        True,
    ),
    "generator-of-a-data-global": (
        """
        MODELS = [Model(5)]

        @cash.cache
        def run(xs):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            for x in xs:
                yield MODELS[0].fit(x)

        print(list(run((3, 4))))
        """,
        True,
    ),
    "returned-iterator-of-a-data-global": (
        """
        MODELS = [Model(5)]

        def note():
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe

        @cash.cache
        def run(xs):
            note()
            return (MODELS[0].fit(x) for x in xs)

        print(list(run((3, 4))))
        """,
        True,
    ),
    "an-argument-with-the-monitoring-ids-taken": (
        """
        for tool in (3, 4):
            sys.monitoring.use_tool_id(tool, "another-tool")

        @cash.cache
        def run(model, xs):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return [model.fit(x) for x in xs]

        print(run(Model(5), (3, 4)))
        """,
        True,
    ),
}


def _runner(tmp_path, shape):
    (tmp_path / "main.py").write_text(HEAD + textwrap.dedent(SHAPES[shape][0]), encoding="utf-8")
    (tmp_path / "model.py").write_text(MODEL, encoding="utf-8")

    def run(k):
        (tmp_path / "helpers.py").write_text(HELPERS.format(K=k), encoding="utf-8")
        proc = run_python("main.py", cwd=tmp_path)
        return proc.stdout.strip(), "[RUN]" in proc.stderr, proc.stderr

    return run


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_an_edit_to_the_helper_recomputes(tmp_path, shape):
    run = _runner(tmp_path, shape)
    out, ran, _ = run(2)
    assert (out, ran) == ("[11, 13]", True)
    out, ran, _ = run(2)
    assert out == "[11, 13]"
    assert ran is not SHAPES[shape][1], "a second run is served exactly where the helper is checked"
    out, ran, _ = run(10)
    assert out == "[35, 45]", f"served the result from before the edit to the helper: {out}"
    assert ran


UNREADABLE = """
from cash.decorator.runtime import KeyBuilder


def unreadable(self, spec, args, kwargs):
    raise RuntimeError("cannot read the helpers")


KeyBuilder.carried_helpers_digest = unreadable
"""


def test_helpers_that_cannot_be_read_leave_the_call_unstored(tmp_path):
    """The digest an unrecorded entry needs cannot be built: not stored,
    and KEY-UNTRACKABLE-DEP says why."""
    script = HEAD + UNREADABLE + textwrap.dedent(SHAPES["returned-iterator-of-an-argument"][0])
    (tmp_path / "main.py").write_text(script, encoding="utf-8")
    (tmp_path / "model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "helpers.py").write_text(HELPERS.format(K=2), encoding="utf-8")
    for _ in range(2):
        proc = run_python("main.py", cwd=tmp_path)
        assert proc.stdout.strip() == "[11, 13]"
        assert "[RUN]" in proc.stderr, "stored although nothing checks the helpers"
        assert "KEY-UNTRACKABLE-DEP" in proc.stderr, proc.stderr
        assert "calls helpers" in proc.stderr
        assert "run uncached" in proc.stderr


def test_a_recorded_call_is_still_served_and_still_checked(tmp_path):
    """The control: the same call returning a list leaves a record."""
    script = """
        @cash.cache
        def run(model, xs):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            return [model.fit(x) for x in xs]

        print(run(Model(5), (3, 4)))
    """
    (tmp_path / "main.py").write_text(HEAD + textwrap.dedent(script), encoding="utf-8")
    (tmp_path / "model.py").write_text(MODEL, encoding="utf-8")

    def run(k):
        (tmp_path / "helpers.py").write_text(HELPERS.format(K=k), encoding="utf-8")
        proc = run_python("main.py", cwd=tmp_path)
        return proc.stdout.strip(), "[RUN]" in proc.stderr, proc.stderr

    assert run(2)[:2] == ("[11, 13]", True)
    out, ran, err = run(2)
    assert (out, ran) == ("[11, 13]", False)
    assert "KEY-UNTRACKABLE-DEP" not in err
    assert run(10)[:2] == ("[35, 45]", True)
