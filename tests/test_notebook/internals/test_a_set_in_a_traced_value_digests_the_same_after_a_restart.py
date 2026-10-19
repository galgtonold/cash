"""A set held inside a library object digests the same in every process.

sklearn's ``@validate_params`` functions keep ``StrOptions`` objects in their
closure, each holding a set of strings. Pickled as they were, their order
followed the process's string hashes, so the trace check saw
``cross_val_score`` as changed after every restart and the fit ran again.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace as Options

from cash.tracking.trace_items import value_digest

_SCRIPT = textwrap.dedent(
    """
    import functools
    from types import SimpleNamespace as Options  # pickled, as sklearn's are

    from cash.tracking.trace_items import value_digest

    def checked(constraints):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **k):
                assert constraints
                return fn(*a, **k)
            return wrapper
        return deco

    @checked({"scoring": [Options(options={"accuracy", "neg_root_mean_squared_error", "r2", "f1", "roc_auc"})]})
    def score(x):
        return x

    print(value_digest(score))
    """
)


def _digest_with_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, check=True)
    digest = out.stdout.strip()
    assert not digest.startswith("opaque")
    return digest


def test_the_digest_holds_across_hash_seeds():
    digests = {_digest_with_seed(seed) for seed in ("0", "1", "2", "3")}
    assert len(digests) == 1


def test_a_different_set_still_digests_differently():
    assert value_digest(Options(o={"a", "b"})) != value_digest(Options(o={"a", "c"}))
    assert value_digest(Options(o=frozenset({"a"}))) != value_digest(Options(o={"a"}))
