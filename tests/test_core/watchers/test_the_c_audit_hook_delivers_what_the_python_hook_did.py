"""cash's audit hook in C (``caudit.c``) hands cash's consumers exactly the
events its Python hook handed them, and counts new code as it did.

The C hook filters, before any Python runs, the audited events no consumer
wants (an ``id()``, an import, a compile): in a fresh interpreter with the
extension and in one without (``CASH_PURE_PYTHON=1``) the same script must
see the same deliveries, the same new-code counts, and no consumer frame
under a profiler.
"""

from __future__ import annotations

import json

import pytest

from cash import _native
from tests._scripts import run_python

_SCRIPT = r"""
import json, os, sys, types
from cash import _native
from cash.tracking import io_watch
from cash.notebook import held_functions as hf

got = []
for event in ("open", "os.listdir", "os.scandir", "exec", "compile", "builtins.id"):
    io_watch.subscribe(event, lambda args, event=event: got.append((event, args[0] if isinstance(args[0], str) else type(args[0]).__name__)))
io_watch.subscribe("os.remove", lambda args: got.append(("os.remove", "outside")), outside_scopes=True)

def run(tag):
    got.append(("--", tag))
    with open("f.txt", "w") as f:
        f.write("x")
    os.listdir(".")
    id(got)
    exec("y = 1", {})
    os.remove("f.txt")

run("no scope")
io_watch.hold()
run("held")
io_watch.release()
run("released")
io_watch.watch_outside_scopes()
run("outside")
io_watch.hold()
# a consumer that raises is skipped, as before
io_watch.subscribe("os.listdir", lambda args: 1 / 0)
run("held again")

counts = []
start = hf._new_code()
for step in ("exec", "def", "lambda", "FunctionType", "id"):
    if step == "exec":
        exec("pass", {})
    elif step == "def":
        def g():
            pass
    elif step == "lambda":
        (lambda: 1)()
    elif step == "FunctionType":
        types.FunctionType((lambda: 2).__code__, {})
    else:
        id(counts)
    counts.append(hf._new_code() - start)

# a profiler does not step into the consumers, as it does not into a Python hook
frames = []
sys.setprofile(lambda frame, event, arg: frames.append(frame.f_code.co_name))
os.listdir(".")
sys.setprofile(None)
c_hook = io_watch._C_HOOK is not None
print(json.dumps({"got": got, "counts": counts, "lambda_frames": frames.count("<lambda>"), "c_hook": c_hook}))
"""


@pytest.mark.skipif(_native.kernels is None, reason=f"no C extension: {_native.REASON}")
def test_the_c_hook_delivers_and_counts_as_the_python_hook(tmp_path):
    (tmp_path / "s.py").write_text(_SCRIPT, encoding="utf-8")
    with_c = json.loads(run_python("s.py", cwd=tmp_path).stdout.splitlines()[-1])
    without = json.loads(run_python("s.py", cwd=tmp_path, env={"CASH_PURE_PYTHON": "1"}).stdout.splitlines()[-1])
    assert with_c.pop("c_hook") is True
    assert without.pop("c_hook") is False
    assert with_c == without
    assert with_c["lambda_frames"] == 0
    events = [event for event, _ in with_c["got"]]
    assert {"open", "os.listdir", "exec", "builtins.id", "os.remove"} <= set(events)
    assert with_c["counts"][0] == 1 and with_c["counts"][-1] >= 2  # an exec, a FunctionType()


@pytest.mark.skipif(_native.kernels is None, reason=f"no C extension: {_native.REASON}")
def test_the_c_hook_dispatches_only_the_events_a_table_names():
    from cash.tracking import io_watch

    if io_watch._C_HOOK is None:
        pytest.skip("the Python hook is installed in this process")
    installed, names = _native.kernels.audit_state()
    assert installed
    assert set(names) == {*io_watch._held_table, *io_watch._idle_table}
    assert "builtins.id" not in names
