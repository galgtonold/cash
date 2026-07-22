"""Adversarial probes, wave 2 (2026-07-02): channels no prior sweep covered.

Process-global state, function-object rebinding via patching, control-flow
definitions, script deps, statement-level skip effectiveness, restart+persist
with user classes, class-creation side effects, and time nondeterminism.

Each test = one mechanism:

 1. test_env_var_edit_invalidates_reader        — os.environ channel: edit the
        env-setting cell; the env-reading consumer must recompute (run_all).
 2. test_decimal_context_edit_invalidates       — decimal.getcontext().prec
        process-global; edit precision; consumer must show new precision.
 3. test_monkeypatch_class_method_edit          — Tool.scale patched in its own
        cell; edit the patch; isolated re-run of the call site must update.
 4. test_conditional_def_flag_edit              — if flag: def f()... else:
        def f()...; edit flag; isolated re-run of call site must switch body.
 5. test_class_method_edit_invalidates_instance_call — edit a method body;
        isolated re-run of m.predict(1) call cell must see the new body.
 6. test_starred_unpack_alias_rerun_idempotent  — a, *rest = data; rest.append;
        isolated re-run idempotent (starred-unpack alias channel).
 7. test_exception_object_crossing_cells        — except .. as e: err = e; err
        consumed downstream; re-run + restart+persist must not crash.
 8. test_run_magic_script_edit_invalidates      — %run script.py defines K;
        script edited on disk; run_all must reflect the new K.
 9. test_init_subclass_registry_rerun_idempotent — __init_subclass__ registers
        into an upstream dict; isolated re-run of the class cell idempotent.
10. test_time_cell_edit_gives_fresh_stamp       — edit the time.time() cell;
        recompute must produce a NEW stamp (not a restored stale one).
11. test_comment_only_edit_stays_cached         — EFFECTIVENESS: comment-only
        edit of an expensive cell must not recompute it.
12. test_whitespace_only_edit_stays_cached      — EFFECTIVENESS: trailing
        blank-line edit must not recompute.
13. test_print_edit_keeps_expensive_stmt_cached — EFFECTIVENESS: statement-level
        promise; editing the print in a two-statement cell must not recompute
        the expensive first statement.
14. test_user_class_instances_restart_persist   — restart+persist with __main__
        class instances: correct values, no AttributeError (CAS-93 family).
15. test_lambda_crossing_cells_restart_persist  — lambda-valued var + restart
        with persist: graceful re-execution, correct value.
"""

import time

import pytest

pytestmark = [pytest.mark.timeout(120)]

MISS = "Executing (cache miss)"
REEXEC = "Auto-executing upstream statement:"


def _p(path) -> str:
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# 1. os.environ process-global channel
# ---------------------------------------------------------------------------

def test_env_var_edit_invalidates_reader(nb_runner):
    nb_runner.create_notebook([
        "import os\nos.environ['CASH_PROBE_MODE'] = 'fast'",
        "mode = os.environ.get('CASH_PROBE_MODE', 'none')\n"
        "factor = 10 if mode == 'fast' else 99\n"
        "print('factor=', factor)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "factor= 10" in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, "import os\nos.environ['CASH_PROBE_MODE'] = 'slow'")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "factor= 99" in out, (
        f"env-var edit not propagated: consumer served stale factor. Got {out!r}"
    )


# ---------------------------------------------------------------------------
# 2. decimal context precision (process-global numeric state)
# ---------------------------------------------------------------------------

def test_decimal_context_edit_invalidates(nb_runner):
    nb_runner.create_notebook([
        "from decimal import Decimal, getcontext\ngetcontext().prec = 4",
        "r = Decimal(1) / Decimal(7)\nprint('r=', r)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "r= 0.1429" in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, "from decimal import Decimal, getcontext\ngetcontext().prec = 8")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "r= 0.14285714" in out, (
        f"decimal-context edit not propagated: consumer served stale precision. Got {out!r}"
    )


# ---------------------------------------------------------------------------
# 3. monkeypatched class method edited
# ---------------------------------------------------------------------------

def test_monkeypatch_class_method_edit(nb_runner):
    nb_runner.create_notebook([
        "class Tool:\n    def scale(self, v):\n        return v + 1",
        "Tool.scale = lambda self, v: v * 2",
        "t = Tool()\nprint('scaled=', t.scale(5))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "scaled= 10" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, "Tool.scale = lambda self, v: v * 3")
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "scaled= 15" in out, (
        f"edited monkeypatch not picked up by isolated call-site re-run. Got {out!r}"
    )


# ---------------------------------------------------------------------------
# 4. conditional function definition, flag edited
# ---------------------------------------------------------------------------

def test_conditional_def_flag_edit(nb_runner):
    nb_runner.create_notebook([
        "flag = True",
        "if flag:\n    def f():\n        return 1\nelse:\n    def f():\n        return 2",
        "print('f()=', f())",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "f()= 1" in nb_runner.get_output(3)

    nb_runner.set_cell_source(1, "flag = False")
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "f()= 2" in out, (
        f"flag edit did not swap the conditionally-defined function. Got {out!r}"
    )


# ---------------------------------------------------------------------------
# 5. class method body edited, instance call site isolated re-run
# ---------------------------------------------------------------------------

def test_class_method_edit_invalidates_instance_call(nb_runner):
    nb_runner.create_notebook([
        "class Model:\n    def predict(self, x):\n        return x + 1",
        "m = Model()",
        "print('pred=', m.predict(1))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "pred= 2" in nb_runner.get_output(3)

    nb_runner.set_cell_source(
        1, "class Model:\n    def predict(self, x):\n        return x + 100"
    )
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "pred= 101" in out, (
        f"edited method body not seen by isolated re-run of the call site. Got {out!r}"
    )


# ---------------------------------------------------------------------------
# 6. starred unpack alias mutation
# ---------------------------------------------------------------------------

def test_starred_unpack_alias_rerun_idempotent(nb_runner):
    nb_runner.create_notebook([
        "data = [1, 2, 3, 4]",
        "a, *rest = data\nrest.append(9)\nprint('rest=', rest)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "rest= [2, 3, 4, 9]" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "rest= [2, 3, 4, 9]" in out, (
        f"starred-unpack re-run not idempotent: {out!r}"
    )


# ---------------------------------------------------------------------------
# 7. exception object crossing cells (+ restart/persist)
# ---------------------------------------------------------------------------

def test_exception_object_crossing_cells(nb_runner):
    nb_runner.create_notebook([
        "err = None\ntry:\n    1 / 0\nexcept ZeroDivisionError as e:\n    err = e",
        "print('etype=', type(err).__name__)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "etype= ZeroDivisionError" in nb_runner.get_output(2)

    nb_runner.run_cell(2)
    assert "etype= ZeroDivisionError" in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "etype= ZeroDivisionError" in out, (
        f"exception-holding var broke after restart+persist: {out!r}"
    )


# ---------------------------------------------------------------------------
# 8. %run script dependency
# ---------------------------------------------------------------------------

def test_run_magic_script_edit_invalidates(nb_runner, tmp_path):
    script = tmp_path / "helper_script.py"
    script.write_text("K = 3\n", encoding="utf-8")
    sp = _p(script)
    nb_runner.create_notebook([
        f"%run {sp}",
        "print('K2=', K * 2)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "K2= 6" in nb_runner.get_output(2)

    time.sleep(1.1)  # generous mtime gap
    script.write_text("K = 50\n", encoding="utf-8")
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "K2= 100" in out, (
        f"%run script edited on disk but consumer served stale K. Got {out!r}"
    )


# ---------------------------------------------------------------------------
# 9. __init_subclass__ registry (class-creation side effect)
# ---------------------------------------------------------------------------

def test_init_subclass_registry_rerun_idempotent(nb_runner):
    nb_runner.create_notebook([
        "registry = []",
        "class PluginBase:\n"
        "    def __init_subclass__(cls, **kw):\n"
        "        super().__init_subclass__(**kw)\n"
        "        registry.append(cls.__name__)",
        "class AlphaPlugin(PluginBase):\n    pass",
        "print('plugins=', registry)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "plugins= ['AlphaPlugin']" in nb_runner.get_output(4)

    # Isolated re-run of the subclass cell, then the consumer: must not double.
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "plugins= ['AlphaPlugin']" in out, (
        f"__init_subclass__ registry accumulated on isolated re-run: {out!r}"
    )


# ---------------------------------------------------------------------------
# 10. time.time cell edited -> fresh stamp
# ---------------------------------------------------------------------------

def test_time_cell_edit_gives_fresh_stamp(nb_runner):
    nb_runner.create_notebook([
        "import time\nstamp = time.time()",
        "print('stamp=', stamp)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(2)

    time.sleep(0.3)
    # A REAL code change (not comment-only) forces recompute of the stamp.
    nb_runner.set_cell_source(1, "import time\nstamp = time.time() + 0.0")
    nb_runner.run_all()
    second = nb_runner.get_output(2)
    s1 = first.split("stamp=")[-1].strip()
    s2 = second.split("stamp=")[-1].strip()
    assert s1 != s2, (
        f"edited time-stamp cell served a STALE stamp: {s1!r} == {s2!r}"
    )


# ---------------------------------------------------------------------------
# 11./12. comment-only / whitespace-only edit should stay cached
# ---------------------------------------------------------------------------

def test_comment_only_edit_stays_cached(nb_runner):
    # sum over 5M ints takes >>10ms, so the statement clears the cost-model
    # floor and actually lands in the cache on the first run.
    nb_runner.create_notebook([
        "base = sum(i * i for i in range(5000000))  # v1",
        "print('base=', base)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "base=" in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, "base = sum(i * i for i in range(5000000))  # v2")
    nb_runner.run_all()
    raw1 = nb_runner.get_raw_output(1)
    assert "base= 41666654166667500000" in nb_runner.get_output(2)
    assert "below 10ms floor" not in raw1, "probe invalid: compute under cost floor"
    assert MISS not in raw1, (
        "EFFECTIVENESS: comment-only edit recomputed the expensive statement "
        f"(markers: {[l for l in raw1.splitlines() if MISS in l]})"
    )


def test_whitespace_only_edit_stays_cached(nb_runner):
    nb_runner.create_notebook([
        "base2 = sum(i * i for i in range(5000000))",
        "print('base2=', base2)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "base2=" in nb_runner.get_output(2)

    nb_runner.set_cell_source(1, "base2 = sum(i * i for i in range(5000000))\n")
    nb_runner.run_all()
    raw1 = nb_runner.get_raw_output(1)
    assert "base2= 41666654166667500000" in nb_runner.get_output(2)
    assert "below 10ms floor" not in raw1, "probe invalid: compute under cost floor"
    assert MISS not in raw1, (
        "EFFECTIVENESS: whitespace-only edit recomputed the expensive statement "
        f"(markers: {[l for l in raw1.splitlines() if MISS in l]})"
    )


# ---------------------------------------------------------------------------
# 13. statement-level skip: edit only the print in a two-statement cell
# ---------------------------------------------------------------------------

def test_print_edit_keeps_expensive_stmt_cached(nb_runner):
    nb_runner.create_notebook([
        "acc = sum(i * i for i in range(5000000))\nprint('acc ready A')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "acc ready A" in nb_runner.get_output(1)

    nb_runner.set_cell_source(1, "acc = sum(i * i for i in range(5000000))\nprint('acc ready B')")
    nb_runner.run_cell(1)
    out = nb_runner.get_output(1)
    raw = nb_runner.get_raw_output(1)
    assert "acc ready B" in out
    # Attribute the misses per statement: between "Statement: acc = ..." and the
    # next "[CACHE DEBUG] Statement:" there must be no cache-miss execution.
    misses = raw.count(MISS)
    acc_recomputed = False
    seg = raw.split("Statement: acc = ")
    if len(seg) > 1:
        tail = seg[1].split("Statement:")[0]
        acc_recomputed = MISS in tail
    assert not acc_recomputed and misses <= 1, (
        f"EFFECTIVENESS (statement-level promise): editing only the print "
        f"statement recomputed the expensive first statement "
        f"(acc_recomputed={acc_recomputed}, total misses={misses})"
    )


# ---------------------------------------------------------------------------
# 14. restart + persist with __main__ class instances
# ---------------------------------------------------------------------------

def test_user_class_instances_restart_persist(nb_runner):
    nb_runner.create_notebook([
        "class Point:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n"
        "        self.y = y",
        "pts = [Point(1, 2), Point(3, 4)]",
        "print('sx=', sum(p.x for p in pts))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "sx= 4" in nb_runner.get_output(3)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "sx= 4" in out, (
        f"__main__-class instances broke after restart+persist: {out!r}"
    )


# ---------------------------------------------------------------------------
# 15. lambda-valued var + restart/persist
# ---------------------------------------------------------------------------

def test_lambda_crossing_cells_restart_persist(nb_runner):
    nb_runner.create_notebook([
        "sq = lambda x: x * x",
        "print('sq4=', sq(4))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "sq4= 16" in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    assert "sq4= 16" in out, (
        f"lambda-valued var broke after restart+persist: {out!r}"
    )
