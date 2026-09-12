"""A helper edited on disk after import is keyed by the code that runs.

CAS-110, round-17 tester r17s3. The ordinary deploy sequence:

    process A: import app              (helper.bump is `x + 1`)
               -- new files land: helper.py now says `x + 100` --
               compute(1) -> 2, stored under the NEW source's key
    process B: compute(1) -> HIT, 2.   The right answer is 101.

A helper's digest is computed lazily, at the first call that needs it, from
the text on disk -- which by then described the new code while the old code
object was the one executing.

Now a helper whose file changed after the process started is compiled from
disk (the whole file, so the compiler sees the module context the import did)
and compared with the loaded code; if they differ, it is keyed by the loaded
bytecode, and KEY-SOURCE-CHANGED says so. The restarted process keys the new
code by its source, a different key, and computes.

Process A is held between import and its first call with a handshake file,
not a sleep, and bytecode caching is off so a stale `.pyc` cannot fake either
outcome (r17s3 hit that trap once).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.core

OLD = "def bump(x):\n    return x + 1\n"
NEW = "def bump(x):\n    return x + 100\n"

MAIN = textwrap.dedent('''
    import os, sys, time
    import cash
    from helper import bump

    @cash.cache
    def compute(x):
        print("COMPUTE", file=sys.stderr, flush=True)  # @cash:assume-safe
        time.sleep(0.25)
        return bump(x)

    if __name__ == "__main__":
        if len(sys.argv) > 1 and sys.argv[1] == "wait":
            open("ready", "w").close()
            for _ in range(400):
                if os.path.exists("go"):
                    break
                time.sleep(0.025)
        print(compute(1))
''')


def _project(tmp_path, helper_text):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "helper.py").write_text(helper_text, encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    return proj


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _edit(path, text):
    """Rewrite *path* and make sure its mtime moves, even on a coarse clock."""
    before = os.stat(path).st_mtime_ns
    path.write_text(text, encoding="utf-8")
    if os.stat(path).st_mtime_ns == before:
        os.utime(path, ns=(before + 1_000_000, before + 1_000_000))


def _run(proj, env, *argv):
    return subprocess.run([sys.executable, "main.py", *argv], cwd=str(proj),
                          capture_output=True, text=True, env=env)


def test_an_edit_between_import_and_first_call_is_not_served_to_the_restart(tmp_path):
    """THE BUG: the restarted process got the old code's answer."""
    proj = _project(tmp_path, OLD)
    env = _env(tmp_path)

    a = subprocess.Popen([sys.executable, "main.py", "wait"], cwd=str(proj),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    for _ in range(400):
        if (proj / "ready").exists():
            break
        time.sleep(0.025)
    assert (proj / "ready").exists(), "process A never reached its first call"
    _edit(proj / "helper.py", NEW)             # the deploy lands under A
    (proj / "go").write_text("", encoding="utf-8")
    a_out, a_err = a.communicate(timeout=60)

    assert a_out.strip() == "2", "A runs the code it imported"
    assert "KEY-SOURCE-CHANGED" in a_err, "nothing said the file changed under A"

    b = _run(proj, env)                         # the restart
    assert b.stdout.strip() == "101", "the restarted process was served the old code's answer"
    assert "COMPUTE" in b.stderr


def test_an_edit_before_the_process_starts_is_ordinary(tmp_path):
    """Control: the normal restart order computes once, then hits."""
    proj = _project(tmp_path, OLD)
    env = _env(tmp_path)
    _edit(proj / "helper.py", NEW)

    first = _run(proj, env)
    second = _run(proj, env)

    assert first.stdout.strip() == second.stdout.strip() == "101"
    assert "COMPUTE" not in second.stderr, "an ordinary restart stopped hitting"
    assert "KEY-SOURCE-CHANGED" not in first.stderr + second.stderr


def test_an_unedited_helper_is_silent(tmp_path):
    """Control: no edit, no notice, and the second run hits."""
    proj = _project(tmp_path, OLD)
    env = _env(tmp_path)

    first = _run(proj, env)
    second = _run(proj, env)

    assert first.stdout.strip() == second.stdout.strip() == "2"
    assert "COMPUTE" not in second.stderr
    assert "KEY-SOURCE-CHANGED" not in first.stderr + second.stderr


# ---------------------------------------------------------------------------
# The decorated function's OWN body (round 18: r18s4's deploy race, r18s1's
# "start the run, keep editing"). The helper fix above did not reach it: the
# root's identity was pinned at its FIRST CALL, from the text on disk, so an
# edit between import and that call keyed the old code's result by the new
# text. Now the pin is taken when the decorator runs, from the text the import
# compiled. The body sleeps past the 0.1 s persistence floor: a call that
# never reaches disk cannot show a cross-process stale entry, which is how the
# tester's "no network" control passed while the bug was still there.
# ---------------------------------------------------------------------------

OWN_OLD = textwrap.dedent('''
    import sys, time
    import cash

    @cash.cache
    def compute(x):
        print("COMPUTE", file=sys.stderr, flush=True)  # @cash:assume-safe
        time.sleep(0.25)  # @cash:assume-safe
        return x * 14

    def unrelated():
        return 1
''')
OWN_NEW = OWN_OLD.replace("x * 14", "x * 3")
OWN_ELSEWHERE = OWN_OLD.replace("return 1", "return 2")

OWN_MAIN = textwrap.dedent('''
    import os, sys, time
    from app import compute

    if len(sys.argv) > 1 and sys.argv[1] == "wait":
        open("ready", "w").close()
        for _ in range(400):
            if os.path.exists("go"):
                break
            time.sleep(0.025)
    print(compute(3))
''')

# The same shape with the decorated function in the script itself.
SCRIPT_OLD = OWN_OLD + textwrap.dedent('''
    if __name__ == "__main__":
        import os
        if len(sys.argv) > 1 and sys.argv[1] == "wait":
            open("ready", "w").close()
            for _ in range(400):
                if os.path.exists("go"):
                    break
                time.sleep(0.025)
        print(compute(3))
''')


def _own_project(tmp_path, layout):
    proj = tmp_path / "proj"
    proj.mkdir()
    if layout == "module":
        (proj / "app.py").write_text(OWN_OLD, encoding="utf-8")
        (proj / "main.py").write_text(OWN_MAIN, encoding="utf-8")
        return proj, proj / "app.py"
    (proj / "main.py").write_text(SCRIPT_OLD, encoding="utf-8")
    return proj, proj / "main.py"


def _edit_under_a(proj, env, edited, new_text):
    """Start A, let it import, edit *edited* on disk, then let A make its first call."""
    a = subprocess.Popen([sys.executable, "main.py", "wait"], cwd=str(proj),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    for _ in range(400):
        if (proj / "ready").exists():
            break
        time.sleep(0.025)
    assert (proj / "ready").exists(), "process A never reached its first call"
    _edit(edited, new_text)
    (proj / "go").write_text("", encoding="utf-8")
    return a.communicate(timeout=60)


@pytest.mark.parametrize("layout", ["module", "script"])
def test_an_edit_to_the_cached_function_itself_is_not_served_to_the_restart(tmp_path, layout):
    """THE BUG (round 18): the restart got 42, the OLD body's answer; 9 is right."""
    proj, edited = _own_project(tmp_path, layout)
    env = _env(tmp_path)
    new_text = edited.read_text(encoding="utf-8").replace("x * 14", "x * 3")

    a_out, a_err = _edit_under_a(proj, env, edited, new_text)
    assert a_out.strip() == "42", "A runs the code it imported"

    b = _run(proj, env)
    assert b.stdout.strip() == "9", "the restarted process was served the old body's answer"
    assert "COMPUTE" in b.stderr
    assert "KEY-SOURCE-CHANGED" in a_err, "nothing said the file changed under A"


IMPORT_WINDOW = textwrap.dedent('''
    import os, sys, time
    import cash

    if os.environ.get("WAIT_IN_IMPORT"):     # a slow import above the def
        open("ready", "w").close()
        for _ in range(400):
            if os.path.exists("go"):
                break
            time.sleep(0.025)


    @cash.cache
    def compute(x):
        print("COMPUTE", file=sys.stderr, flush=True)  # @cash:assume-safe
        time.sleep(0.25)  # @cash:assume-safe
        return x * 14
''')


def test_an_edit_while_the_module_is_still_importing_is_not_served_to_the_restart(tmp_path):
    """Round 20 (r20s1): the edit landed after Python compiled the module but
    before its ``@cash.cache`` line ran, so the pin taken at decoration read the
    NEW text for the OLD code. KEY-SOURCE-CHANGED fired, and the old body's
    answer was stored under the new text's key anyway."""
    proj = tmp_path / "proj"
    proj.mkdir()
    app = proj / "app.py"
    app.write_text(IMPORT_WINDOW, encoding="utf-8")
    (proj / "main.py").write_text("from app import compute\nprint(compute(3))\n", encoding="utf-8")
    env = _env(tmp_path)

    a = subprocess.Popen([sys.executable, "main.py"], cwd=str(proj), text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=dict(env, WAIT_IN_IMPORT="1"))
    for _ in range(400):
        if (proj / "ready").exists():
            break
        time.sleep(0.025)
    assert (proj / "ready").exists(), "process A never reached the window"
    _edit(app, IMPORT_WINDOW.replace("x * 14", "x * 3"))
    (proj / "go").write_text("", encoding="utf-8")
    a_out, a_err = a.communicate(timeout=60)
    assert a_out.strip() == "42", "A runs the code it compiled"

    b = _run(proj, env)
    assert b.stdout.strip() == "9", "the restarted process was served the old body's answer"
    assert "COMPUTE" in b.stderr


def test_an_edit_elsewhere_in_the_same_file_still_hits_after_the_restart(tmp_path):
    """Control: the cached function did not change, so its entry stays valid and
    nothing is said. Guards against "fixing" this by refusing anything stored
    after its file moved."""
    proj, edited = _own_project(tmp_path, "module")
    env = _env(tmp_path)

    a_out, a_err = _edit_under_a(proj, env, edited, OWN_ELSEWHERE)
    assert a_out.strip() == "42"
    assert "KEY-SOURCE-CHANGED" not in a_err

    b = _run(proj, env)
    assert b.stdout.strip() == "42"
    assert "COMPUTE" not in b.stderr, "an unchanged function stopped hitting"


def test_a_closure_in_an_edited_file_is_compared_without_raising(tmp_path, monkeypatch):
    """The comparison built a function from each candidate code object with
    `types.FunctionType(code, {})`, which raises for a nested function with
    free variables -- so a cached closure whose file changed after import
    failed its key build (KEY-BUILD-FAILED) instead of being compared."""
    import importlib

    from cash.source_norm import loaded_code_matches_disk

    src = "def make(k):\n    def inner(x):\n        return x * k\n    return inner\n"
    (tmp_path / "closmod_r18.py").write_text(src, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("closmod_r18")
    try:
        inner = mod.make(3)
        _edit(tmp_path / "closmod_r18.py", src + "\nOTHER = 1\n")    # inner unchanged
        assert loaded_code_matches_disk(inner) is True
        _edit(tmp_path / "closmod_r18.py", src.replace("x * k", "x + k"))
        assert loaded_code_matches_disk(inner) is False
    finally:
        sys.modules.pop("closmod_r18", None)


def test_an_unedited_cached_function_hits_across_processes(tmp_path):
    """Control: decoration-time pinning keeps the key byte-stable run to run."""
    proj, _ = _own_project(tmp_path, "module")
    env = _env(tmp_path)

    first = _run(proj, env)
    second = _run(proj, env)
    assert first.stdout.strip() == second.stdout.strip() == "42"
    assert "COMPUTE" not in second.stderr
    assert "KEY-SOURCE-CHANGED" not in first.stderr + second.stderr
