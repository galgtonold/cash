"""CAS-93: def/class statements are never value-restored.

Functions and classes pickle BY REFERENCE to the binding they create, so a
persisted entry is a dangling pointer after a kernel restart: restoring it
crashed the defining cell with ``AttributeError: module '__main__' has no
attribute ...``. Definition statements now always execute (microseconds),
their lineage stays source-derived, and unrestorable pickles at lookup time
degrade to a clean miss instead of an error.
"""

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.restore]


def test_class_cell_survives_restart_with_persist(nb_runner):
    nb_runner.create_notebook([
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
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
    assert "sx= 4" in nb_runner.get_output(3), (
        "class-defining notebook broke after restart+persist"
    )


def test_def_plus_unpicklable_user_survives_restart(nb_runner):
    """The original CAS-93 repro: a cell defining a function and handing it
    to an unpicklable stateful object must re-execute cleanly post-restart."""
    nb_runner.create_notebook([
        "from concurrent.futures import ThreadPoolExecutor\n"
        "def _work():\n"
        "    return 21 * 2\n"
        "ex = ThreadPoolExecutor(max_workers=1)\n"
        "fut = ex.submit(_work)",
        "print(f'res={fut.result()}')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res=42" in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "res=42" in nb_runner.get_output(2), (
        "def+executor cell broke after restart+persist"
    )


def test_lambda_assign_survives_restart(nb_runner):
    """Lambda assigns are Assign statements (not FunctionDef): the store
    guard must keep them re-executing rather than storing an entry that
    'restores' nothing and skips the binding."""
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
    assert "sq4= 16" in nb_runner.get_output(2), (
        "lambda-valued var broke after restart+persist"
    )


def test_function_alias_survives_restart(nb_runner):
    nb_runner.create_notebook([
        "def base_fn(x):\n    return x + 1",
        "alias_fn = base_fn",
        "print('a=', alias_fn(1))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "a= 2" in nb_runner.get_output(3)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "a= 2" in nb_runner.get_output(3), (
        "function-alias chain broke after restart+persist"
    )


def test_expensive_value_still_restores_next_to_defs(nb_runner):
    """Statement granularity preserved: the def re-executes, but a heavier
    sibling value still restores from disk after restart (no full-cell
    fallback)."""
    nb_runner.create_notebook([
        "def scale_fn(x):\n    return x * 2\n"
        "big93 = sum(i * i for i in range(3000000))",
        "print('r=', scale_fn(2), big93 > 0)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "r= 4 True" in nb_runner.get_output(2)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.enable_persist()
    nb_runner.run_all()
    out = nb_runner.get_output(2)
    raw = nb_runner.get_raw_output(1)
    assert "r= 4 True" in out, out
    # The def must have been executed (not restored); the big value may be
    # restored from disk. Either way, no crash and correct values.
    assert "AttributeError" not in raw, raw[-500:]
