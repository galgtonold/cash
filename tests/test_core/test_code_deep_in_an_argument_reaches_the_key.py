"""User code nested many containers deep in an argument is keyed by its code.

The search for code stopped eight containers down without a word, so a
function ten lists deep was keyed by name and an edit to it served the old
result. Later it stopped at a hundred, with a warning. It now reaches code
however deep it is held.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import types
import warnings

import pytest

from cash import Cash, FileBackend

HELPER = """
    def fn(x):
        return x * 1
"""

JOB = """
    import sys
    import cash
    from helper import fn

    @cash.cache
    def run(tree):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        for _ in range(12):
            tree = tree[0]
        return tree(10)

    tree = fn
    for _ in range(12):
        tree = [tree]
    print(run(tree))
"""


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    past = time.time() - 30  # saved a while before the run, like an ordinary edit
    os.utime(path, (past, past))


def _run(proj):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(proj / ".cash"))
    p = subprocess.run([sys.executable, "job.py"], cwd=str(proj), env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.split(), p.stderr.count("[RUN]")


@pytest.mark.timeout(300)
def test_editing_a_function_twelve_lists_deep_recomputes(tmp_path):
    _write(tmp_path / "helper.py", HELPER)
    _write(tmp_path / "job.py", JOB)
    assert _run(tmp_path) == (["10"], 1)
    assert _run(tmp_path) == (["10"], 0)
    _write(tmp_path / "helper.py", HELPER.replace("x * 1", "x * 2"))
    assert _run(tmp_path) == (["20"], 1)


def _deep(value, depth):
    for _ in range(depth):
        value = [value]
    return value


def _fileless_function(module_name: str, body: str):
    """``fn`` defined in a registered module with no file, as a notebook
    cell or ``exec`` defines it: pickled by name, keyed by its code."""
    module = sys.modules.get(module_name) or types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(textwrap.dedent(body), module.__dict__)
    return module.fn


def test_a_function_far_deeper_than_one_search_stretch_is_keyed_by_its_code(tmp_path):
    """The search stopped 100 containers down and keyed what lay deeper by
    name, so editing a function 150 lists deep served the old result."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def run(tree):
        while isinstance(tree, list):
            tree = tree[0]
        return tree(10)

    name = "_cash_test_deep_code_arg"
    try:
        fn = _fileless_function(name, "def fn(x):\n    return x * 1\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert run(_deep(fn, 150)) == 10
        assert not [w for w in caught if "KEY-OPAQUE-CALLABLE" in str(w.message)]
        fn = _fileless_function(name, "def fn(x):\n    return x * 2\n")
        assert run(_deep(fn, 150)) == 20
    finally:
        sys.modules.pop(name, None)


def test_a_deep_value_of_plain_data_does_not_warn(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def run(tree):
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(_deep(1, 20))
    assert not [w for w in caught if "deep" in str(w.message)]
