"""Every helper a cached function reaches is in its key, however deep.

The walk that finds helpers used to stop six calls below the cached function.
A helper past that was not keyed at all, so editing it served the result from
before the edit. Which helpers fell past the cut also depended on the order
the walk met them in, and that order came from a set, so it moved with
``PYTHONHASHSEED``: a deep helper graph was keyed differently in every process
and never hit after a restart.

The visited set already ends cycles; the only thing a bound can still be for
is code that makes a new function on every read, where the walk would never
end. That call runs uncached, with a warning, instead of leaving code out.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import cash
from cash import Cash
from cash.analysis.purity_analyzer import PurityAnalyzer, get_analyzer
from cash.exceptions import CashWarning

pytestmark = [pytest.mark.core]

CHAIN = 12  # h0 .. h11: twice as deep as the old cut


def _helpers(step: int) -> str:
    """``h0 .. h11``, each calling the next; the last one's factor is *step*.

    The cached function calls ``h0 .. h4`` directly, so ``h5`` is reachable
    from five callers at five depths: the shape where the old walk's result
    depended on the order it met them in."""
    lines = [f"def h{i}(x):\n    return h{i + 1}(x)\n\n" for i in range(CHAIN - 1)]
    lines.append(f"def h{CHAIN - 1}(x):\n    return x * {step}\n")
    return "\n".join(lines)


JOB = """\
import os
import sys
import time

import cash

from {helpers} import h0, h1, h2, h3, h4


@cash.cache
def total(x):
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")  # @cash:assume-safe
    os.close(fd)  # @cash:assume-safe
    return h0(x) + h1(x) + h2(x) + h3(x) + h4(x)


print("ANSWER", total(1))
"""


def _write(project: Path, step: int) -> None:
    (project / "deep_helpers.py").write_text(_helpers(step), encoding="utf-8")
    (project / "job.py").write_text(JOB.format(helpers="deep_helpers"), encoding="utf-8")


def _run(project: Path, seed: str = "0") -> str:
    src = str(Path(cash.__file__).resolve().parents[1])  # the cash under test
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(project / ".cash")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONHASHSEED"] = seed
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "job.py", str(project / "runs")],
        cwd=str(project),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.split("ANSWER")[-1].strip()


def test_every_helper_of_a_deep_chain_is_keyed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    name = f"_deep_chain_{tmp_path.name}"
    monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{name}.py").write_text(
        _helpers(1) + "\n\ndef entry(x):\n    return h0(x) + h1(x) + h2(x) + h3(x) + h4(x)\n",
        encoding="utf-8",
    )
    module = importlib.import_module(name)

    keyed = get_analyzer().analyze(module.entry).helper_source_hashes

    assert {f"{name}.h{i}" for i in range(CHAIN)} <= set(keyed), sorted(keyed)


@pytest.mark.timeout(300)
def test_an_edit_deep_in_the_chain_invalidates_the_next_run(tmp_path):
    _write(tmp_path, step=1)
    assert _run(tmp_path) == "5"
    assert _run(tmp_path) == "5"
    assert (tmp_path / "runs").read_bytes() == b"x"  # the second run was served from disk

    _write(tmp_path, step=100)  # h11, twelve calls below `total`
    assert _run(tmp_path) == "500"
    assert (tmp_path / "runs").read_bytes() == b"xx"


@pytest.mark.timeout(300)
def test_the_key_is_the_same_under_every_hash_seed(tmp_path):
    """Stored under one seed, served under every other: the helpers keyed
    do not depend on the order a set gives in this process."""
    _write(tmp_path, step=1)
    assert _run(tmp_path, seed="0") == "5"
    for seed in ("1", "2", "3", "4"):
        assert _run(tmp_path, seed=seed) == "5"
    assert (tmp_path / "runs").read_bytes() == b"x", "a later process under another seed missed"


ENDLESS = """\
import {name}


def __getattr__(attr):
    if attr != "step":
        raise AttributeError(attr)

    def step(x):
        return {name}.step(x - 1) if x else 0

    return step


def total(x):
    return {name}.step(x)
"""


def test_a_walk_with_no_end_runs_uncached_with_a_warning(tmp_path, monkeypatch):
    """Each read of ``step`` makes a new function that reads ``step``: the
    walk can never finish, so the call is not keyed by the part it saw."""
    monkeypatch.setattr(PurityAnalyzer, "_WALK_LIMIT", 40)  # the real bound only costs time
    monkeypatch.syspath_prepend(str(tmp_path))
    name = f"_endless_{tmp_path.name}"
    monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{name}.py").write_text(ENDLESS.format(name=name), encoding="utf-8")
    module = importlib.import_module(name)

    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    total = c.cache(module.total)
    with pytest.warns(CashWarning, match="do not end"):
        assert total(3) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert total(3) == 0

    info = total.cache_info()
    assert info["hits"] == 0
    assert info["miss_reasons"] == {"helpers could not all be keyed": 2}
    assert total.explain(3).details["error"] == "helpers could not all be keyed"
