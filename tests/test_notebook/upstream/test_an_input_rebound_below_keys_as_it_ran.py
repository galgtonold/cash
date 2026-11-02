"""The simulation keys a statement on what its input held when it ran.

``housing_prepared = preprocessing.fit_transform(housing)`` ran on a
transformer holding no object of a notebook class. A cell below rebinds
``preprocessing`` to one that holds ``ClusterSimilarity``. The simulation,
keying the first statement again, walked the value the name holds NOW for
the notebook code it reaches, folded that class's methods into a key the
run never had, and so never found the statement's entry: what the run
recorded under it (the objects it shared, the files it read) was not
replayed, and the lineages built from there differed from the run's. A
first run of the notebook from top to bottom then rebuilt dozens of
statements, every search fitted above included.
"""

from __future__ import annotations

from cash.notebook.cache_key import CacheKeyContext, compute_cache_key

SOURCE = (
    "def scale(v):\n"
    "    return v * 2\n"
    "class Step:\n"
    "    def run(self, v):\n"
    "        return scale(v)\n"
    "class Box:\n"
    "    def __init__(self, items):\n"
    "        self.items = items\n"
    "    def fit(self):\n"
    "        return len(self.items)\n"
)
CODE = "res = pre.fit()"


def _namespace() -> dict:
    ns: dict = {"__name__": "__main__"}
    exec(compile(SOURCE, "<cell>", "exec"), ns)
    return ns


def _key(ns, variable_lineage, virtual_lineage=None) -> str:
    ctx = CacheKeyContext(variable_lineage=variable_lineage, user_ns=ns, virtual_lineage=virtual_lineage)
    return compute_cache_key(CODE, {"pre"}, ctx=ctx, outputs={"res"}).cache_key


def test_the_simulation_keys_the_value_the_statement_ran_on():
    ns = _namespace()
    lineages = {"Box": "box0", "Step": "step0", "scale": "scale0"}
    ns["pre"] = ns["Box"]([1, 2, 3])
    ran = _key(ns, {**lineages, "pre": "pre-before"})
    ns["pre"] = ns["Box"]([ns["Step"]()])  # the cell below
    now = {**lineages, "pre": "pre-after"}
    assert _key(ns, now, {**lineages, "pre": "pre-before"}) == ran


def test_the_value_the_name_holds_now_keys_its_own_statement():
    """The control: the statement below, on the rebound value, reaches the
    class it holds -- an edit of ``scale`` re-keys it."""
    ns = _namespace()
    lineages = {"Box": "box0", "Step": "step0", "scale": "scale0"}
    ns["pre"] = ns["Box"]([ns["Step"]()])
    now = {**lineages, "pre": "pre-after"}
    assert _key(ns, now, dict(now)) != _key(ns, {**now, "scale": "scale1"}, {**now, "scale": "scale1"})
