"""Code of a script that no module in ``sys.modules`` holds is keyed by its file.

Found walking nanoGPT: a driver runs the training script with
``runpy.run_path(path, run_name="__main__")``. While it runs, ``__main__`` in
``sys.modules`` is the script, so a function of the driver's that ran inside
a traced call (a ``print`` it installed) was named ``("code", "__main__",
name)`` and looked up in the script in the next process: it always read as
changed, and the call recomputed in every process.

Each case runs in fresh interpreters on one cache: the second process hits;
editing the driver's function, or the script's, recomputes. A script run as
``python train.py`` still hits as before.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

TRAIN = textwrap.dedent("""
    import sys
    import __main__
    import cash

    def helper(x):
        return x * {FACTOR}

    STEP = "helper"

    @cash.cache
    def run(x):
        print("running")
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return getattr(__main__, STEP)(x) + 1

    if __name__ == "__main__":
        print(run(5))
""")

DRIVER = textwrap.dedent("""
    import builtins
    import runpy

    _print = builtins.print

    def logged_print(*args, **kwargs):
        return _print("{TAG}", *args, **kwargs)

    builtins.print = logged_print
    runpy.run_path("train.py", run_name="__main__")
""")


def _write(tmp_path, factor=3, tag="[log]"):
    (tmp_path / "train.py").write_text(TRAIN.format(FACTOR=factor), encoding="utf-8")
    (tmp_path / "driver.py").write_text(DRIVER.format(TAG=tag), encoding="utf-8")


def _run(tmp_path, script):
    proc = run_python(script, cwd=tmp_path)
    return proc.stdout.strip().splitlines()[-1], proc.stderr


def test_a_driver_function_that_ran_inside_hits_in_the_next_process(tmp_path):
    _write(tmp_path)
    out, err = _run(tmp_path, "driver.py")
    assert out == "[log] 16" and "[RUN]" in err
    out, err = _run(tmp_path, "driver.py")
    assert out == "[log] 16"
    assert "[RUN]" not in err, f"the next process ran the call again:\n{err[-1500:]}"


def test_editing_the_driver_or_the_script_recomputes(tmp_path):
    _write(tmp_path)
    _run(tmp_path, "driver.py")
    _write(tmp_path, tag="[LOG]")
    out, err = _run(tmp_path, "driver.py")
    assert "[RUN]" in err, "an edit to the driver's function served the old entry"
    assert out == "[LOG] 16"
    _write(tmp_path, factor=4, tag="[LOG]")
    out, err = _run(tmp_path, "driver.py")
    assert "[RUN]" in err and out == "[LOG] 21", f"an edit to the script served the old result: {out}"


def test_the_script_run_directly_still_hits(tmp_path):
    _write(tmp_path)
    out, err = _run(tmp_path, "train.py")
    assert out == "16" and "[RUN]" in err
    out, err = _run(tmp_path, "train.py")
    assert out == "16" and "[RUN]" not in err, err[-1500:]
