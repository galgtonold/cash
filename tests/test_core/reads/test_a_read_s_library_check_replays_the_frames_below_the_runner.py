"""A read's library check walks only the frames above cash's runner again.

Whether a read is a library's own (`incidental_read`) walked the whole stack
for every file: ~50 frames in a notebook, 60 us a read, 1.2 s of a cell that
reads 20,000 files. The frames below cash's frame above the reading code (a
statement's runner) do not change while it runs, so what they say is kept and
replayed for the next read from under the same frame. Each case here is
answered the same with the replay as by the full walk: a library calling cash
and reading its own package's file, an installed module being imported, a
package metadata lookup, and the user's own data.
"""

from __future__ import annotations

import types

import pytest

from cash._paths import normalize_path
from cash.tracking import read_classification as rc


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    site = tmp_path / "site"
    package = site / "fakelib"
    package.mkdir(parents=True)
    module = types.ModuleType("fakelib")
    module.__path__ = [str(package)]
    module.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(__import__("sys").modules, "fakelib", module)
    root = rc.norm_dir(str(site))
    monkeypatch.setattr(rc, "installed_roots", lambda: (root,))
    return package


def _frames(below_runner):
    """``below_runner(call)`` -> a stack: [what below_runner builds] -> a cash
    runner -> a notebook statement -> cash's hook -> the check."""
    seen = {}
    # cash's hook asks; the test's own frames are below the runner, where
    # they are walked like any caller's.
    hook_globals = {"__name__": "cash.notebook.fake_hook", "rc": rc}
    exec("def hook(path, anchor):\n    return rc.incidental_read(path, None, anchor)\n", hook_globals)
    statement_globals = {"__name__": "__main__", "hook": hook_globals["hook"]}
    exec("def statement(path, anchor):\n    return hook(path, anchor)\n", statement_globals)
    runner_globals = {"__name__": "cash.notebook.fake_runner", "statement": statement_globals["statement"]}
    exec("def runner(path, anchor):\n    return statement(path, anchor)\n", runner_globals)

    def ask(path):
        anchor = [None]
        full = below_runner(runner_globals["runner"], path, None)
        first = below_runner(runner_globals["runner"], path, anchor)
        built = anchor[0] is not None
        again = below_runner(runner_globals["runner"], path, anchor)
        seen.update(full=full, first=first, again=again, built=built)
        return seen

    return ask


def _from_user(runner, path, anchor):
    return runner(path, anchor)


def _from_library(runner, path, anchor):
    lib = {"__name__": "fakelib.core", "runner": runner}
    exec("def drive(path, anchor):\n    return runner(path, anchor)\n", lib)
    return lib["drive"](path, anchor)


def _from_import(origin):
    def call(runner, path, anchor):
        spec = types.SimpleNamespace(origin=origin, _initializing=True)
        lib = {"__name__": "fakelib", "__spec__": spec, "runner": runner, "args": (path, anchor), "out": {}}
        exec(compile("out['why'] = runner(*args)\n", origin, "exec"), lib)
        return lib["out"]["why"]

    return call


def _from_metadata(runner, path, anchor):
    meta = {"__name__": "importlib.metadata.fake", "runner": runner}
    exec("def lookup(path, anchor):\n    return runner(path, anchor)\n", meta)
    return meta["lookup"](path, anchor)


@pytest.mark.parametrize(
    "case, below, inside, expected",
    [
        ("user data", _from_user, False, None),
        ("a library's own file", _from_library, True, "library resource"),
        ("a library reading user data", _from_library, False, None),
        ("an installed module being imported", "import", False, "library import"),
        ("a metadata lookup", _from_metadata, False, "package metadata"),
    ],
)
def test_the_replay_answers_as_the_full_walk(fake_install, tmp_path, case, below, inside, expected):
    if below == "import":
        below = _from_import(str(fake_install / "__init__.py"))
    path = normalize_path(str((fake_install if inside else tmp_path) / "data.csv"))
    seen = _frames(below)(path)
    assert seen["built"], case
    assert seen["full"] == seen["first"] == seen["again"] == expected, (case, seen)


def test_a_suspended_frame_is_not_an_anchor(fake_install, tmp_path):
    runner_globals = {"__name__": "cash.notebook.fake_runner", "rc": rc}
    exec("def runner(path, anchor):\n    yield rc.incidental_read(path, None, anchor)\n", runner_globals)
    # The runner itself is a generator: its callers change between resumptions.
    statement = {"__name__": "__main__", "runner": runner_globals["runner"]}
    exec("def statement(path, anchor):\n    return next(runner(path, anchor))\n", statement)
    anchor = [None]
    assert statement["statement"](normalize_path(str(tmp_path / "x.csv")), anchor) is None
    assert anchor[0] is None or anchor[0][0].f_code.co_name != "runner"
