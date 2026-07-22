"""Adversarial probes, wave 4 (2026-07-02): decorator <-> notebook interplay.

The decorator's arg hash prefers a ``_cash_lineage_hash`` attribute over
content (core.py get_arg_hash). Inside a cash-enabled notebook the lineage
store attaches that attribute to user objects — a shallow copy carries the
attribute in ``__dict__`` while diverging in content, so a decorated call with
the copy may key on the STALE inherited lineage.

 1. test_copied_object_decorated_call_distinct — copy.copy(obj) + mutate, then
        decorated calls with original and copy must return different results.
 2. test_notebook_defined_decorated_fn_edit    — @cash.cache function defined
        in a cell; edit its body cell; downstream call must recompute.
 3. test_decorated_call_seen_by_lineage        — decorated result feeding a
        notebook variable: edit decorated fn -> downstream cell updates.
"""

import pytest

pytestmark = [pytest.mark.timeout(120)]


def test_copied_object_decorated_call_distinct(nb_runner):
    nb_runner.create_notebook([
        "import cash\nimport copy",
        "class Cfg:\n    def __init__(self, v):\n        self.v = v",
        "cfg = Cfg(5)",
        "cfg2 = copy.copy(cfg)\ncfg2.v = 99",
        "@cash.cache\ndef evaluate(c):\n    return c.v * 2",
        "print('r=', evaluate(cfg), evaluate(cfg2))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    out = nb_runner.get_output(6)
    assert "r= 10 198" in out, (
        f"decorated call with a copied+mutated object served the ORIGINAL's "
        f"cached result (stale inherited _cash_lineage_hash?). Got {out!r}"
    )


def test_notebook_defined_decorated_fn_edit(nb_runner):
    nb_runner.create_notebook([
        "import cash",
        "@cash.cache\ndef score(x):\n    return x + 1",
        "print('s=', score(10))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "s= 11" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, "@cash.cache\ndef score(x):\n    return x + 100")
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "s= 110" in out, (
        f"edited notebook-defined @cash.cache function served stale result: {out!r}"
    )


def test_decorated_call_seen_by_lineage(nb_runner):
    nb_runner.create_notebook([
        "import cash",
        "@cash.cache\ndef base_value():\n    return 7",
        "v = base_value()",
        "print('v2=', v * 2)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "v2= 14" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, "@cash.cache\ndef base_value():\n    return 9")
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "v2= 18" in out, (
        f"decorated-function edit not propagated through notebook lineage to "
        f"the isolated downstream run: {out!r}"
    )
