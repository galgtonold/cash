"""A cached function without readable source keys its helpers and globals too.

``python - <<EOF``, ``python -c`` and ``exec`` give a function whose source
``inspect`` cannot read. Its own bytecode was keyed, but the walk that finds
its helpers and the fold of the globals it reads both start from the source,
so neither ran: editing a helper or a global it reads served the old result,
silently. Both now start from the bytecode when there is no source.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
import types
import warnings

import pytest

from cash import Cash

SCRIPT = """
import sys
import cash
RATE = {rate}
def helper(x):
    return x * {k}
@cash.cache
def f(x):
    print("[RUN]", file=sys.stderr)
    return helper(x) + RATE
print(f(2))
"""


def _module(name: str, body: str, monkeypatch) -> types.ModuleType:
    mod = types.ModuleType(name)
    exec(compile(textwrap.dedent(body), "<stdin>", "exec"), mod.__dict__)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_a_global_it_reads_reaches_the_key(tmp_path, monkeypatch):
    mod = _module(
        "sourceless_globals",
        """
        RATE = 0
        def f(x):
            return x + RATE
        """,
        monkeypatch,
    )
    with pytest.raises(OSError):
        inspect.getsource(mod.f)  # the premise: no source
    f = Cash(cache_dir=str(tmp_path / "cache")).cache(mod.f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert f(2) == 2
        mod.RATE = 5
        assert f(2) == 7, "a rebound global was served the old result"


def test_a_helper_it_calls_reaches_the_key(tmp_path, monkeypatch):
    mod = _module(
        "sourceless_helpers",
        """
        def helper(x):
            return x * 10
        def f(x):
            return helper(x)
        """,
        monkeypatch,
    )
    f = Cash(cache_dir=str(tmp_path / "cache")).cache(mod.f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert f(2) == 20
        exec(compile("def helper(x):\n    return x * 11\n", "<stdin>", "exec"), mod.__dict__)
        assert f(2) == 22, "an edited helper was served the old result"


def test_a_cached_function_it_calls_is_an_edge(tmp_path, monkeypatch):
    mod = _module(
        "sourceless_edges",
        """
        RATE = 2
        def inner(x):
            return x * RATE
        def outer(x):
            return inner(x) + 1
        """,
        monkeypatch,
    )
    c = Cash(cache_dir=str(tmp_path / "cache"))
    mod.inner = c.cache(mod.inner)
    mod.outer = c.cache(mod.outer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mod.outer(1) == 3
        mod.RATE = 5
        assert mod.outer(1) == 6, "the caller kept its result after its cached callee's global changed"


def _stdin_run(tmp_path, rate, k):
    env = {k_: v for k_, v in os.environ.items() if not k_.startswith("CASH_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", CASH_CACHE_DIR=str(tmp_path / ".cash"))
    p = subprocess.run(
        [sys.executable, "-"],
        input=SCRIPT.format(rate=rate, k=k),
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.mark.timeout(120)
def test_a_heredoc_script_recomputes_after_an_edit(tmp_path):
    assert _stdin_run(tmp_path, 0, 10) == ("20", True)
    assert _stdin_run(tmp_path, 0, 10) == ("20", False), "the unedited run did not hit"
    assert _stdin_run(tmp_path, 0, 11) == ("22", True), "an edited helper was served the old result"
    assert _stdin_run(tmp_path, 5, 10) == ("25", True), "an edited global was served the old result"


def test_what_the_bytecode_plainly_writes_is_left_out():
    """Without source the bytecode decides what is a write: `G.append(v)` and
    `G[k] = v` are kept out of the key (they would move it on every call),
    and a global only read, or stored as a value (`d[k] = G`), stays in.
    `G9[0] = v` has a constant key, which 3.14 loads with its own opcode."""
    from cash.decorator.globals_fold import _bytecode_mutated_globals

    ns: dict = {}
    src = (
        "def f(k, v, d):\n    G1.append(v)\n    G2[k] = v\n    d[k] = G3\n"
        "    G4.x = v\n    d.y = G5\n    del G6[k]\n    G9[0] = v\n    return G7[k] + len(G8)\n"
    )
    exec(compile(src, "<stdin>", "exec"), ns)
    names = {f"G{i}" for i in range(1, 10)}
    assert _bytecode_mutated_globals((ns["f"].__code__,), names) == {"G1", "G2", "G4", "G6", "G9"}
