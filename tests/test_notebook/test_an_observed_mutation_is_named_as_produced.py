"""A call seen changing its argument names that argument as what it produced.

``sc.pp.normalize_total(adata)`` is a bare call; nothing in
its code says it writes ``adata``. Cash sees it at runtime -- the argument's
content changed -- and treats ``adata`` as an output from then on. The badge
row of that first run still said "Produced -": the observed name reached the
lineage and the cache decision but not the row.
"""

from cash.notebook.cache_status import CacheStatus


def test_the_first_run_names_the_argument_it_changed(cash_magics):
    ns = cash_magics.shell.user_ns
    exec("def normalize(d):\n    d['x'] = [v / 2 for v in d['x']]\n", ns)
    ns["data"] = {"x": [2.0, 4.0]}
    metrics = cash_magics._statement_processor.process_statement("normalize(data)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert ns["data"]["x"] == [1.0, 2.0]
    assert "data" in metrics["evaluated_vars"], metrics["evaluated_vars"]


def test_a_call_that_changes_nothing_names_nothing(cash_magics):
    ns = cash_magics.shell.user_ns
    exec("def look(d):\n    return len(d['x'])\n", ns)
    ns["data"] = {"x": [2.0, 4.0]}
    metrics = cash_magics._statement_processor.process_statement("look(data)")
    assert "data" not in metrics.get("evaluated_vars", []), metrics["evaluated_vars"]
