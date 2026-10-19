"""A name several changed partners move on gets the same lineage in every
process.

The walk started from the statement's outputs in their set's order, which
follows the process's string hashes. ``for est in estimators: est.fit(X,
y)`` moves the four models and the list, and the loop variable left holding
the last model is moved on by each of them in turn: after a restart it came
out with another lineage, and every entry storing it with its outputs
missed (E8: ``voting_clf.fit``, 31 s).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    from cash.notebook.statement.derivation_edges import bump_derived_lineages

    models = ["random_forest_clf", "extra_trees_clf", "svm_clf", "mlp_clf"]
    edges = {"estimators": set(models), "estimator": {"mlp_clf"}}
    for m in models:
        edges[m] = {"estimators", "estimator"}
    lineage = {name: "L-" + name for name in [*models, "estimators", "estimator"]}
    changed = set(models) | {"estimators"}
    bump_derived_lineages(
        edges, lineage, changed, changed, record=lineage.__setitem__, present=lambda name: True
    )
    print(lineage["estimator"])
    """
)


def _lineage_with_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_the_lineage_holds_across_hash_seeds():
    assert len({_lineage_with_seed(seed) for seed in ("0", "1", "2", "3", "4", "5")}) == 1
