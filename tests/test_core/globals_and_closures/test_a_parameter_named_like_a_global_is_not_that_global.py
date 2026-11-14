"""A parameter shadows the module global of the same name: a driver called
from a module-level loop over that name hits on repeat.

``for sampler in samplers: resample(sampler, X, y)`` leaves a module global
``sampler`` beside the parameter of ``def resample(sampler, X, y)``. The
helper walk resolved ``sampler.fit_resample`` in the module's globals, so the
global's bound method was keyed as a binding the function calls: each call
keyed whichever sampler the loop last bound, and missed with "a global it
reads changed". A call through a parameter or local is the argument's code,
folded from the argument; a real global read still invalidates.
"""

from __future__ import annotations

import json
import textwrap

from tests._scripts import run_python

_LOOP = """
import json

import cash

c = cash.Cash(register_magic=False)
RUNS = []
FACTOR = 2


class Sampler:
    def __init__(self, k):
        self.k = k

    def fit_resample(self, X, y):
        return [x * self.k for x in X], list(y)


@c.cache
def resample(sampler, X, y):
    RUNS.append(1)  # @cash:assume-safe
    return sampler.fit_resample(X, y)


@c.cache
def scaled(sampler, X):
    RUNS.append(1)  # @cash:assume-safe
    return [x * FACTOR for x in sampler.fit_resample(X, X)[0]]


samplers = [Sampler(1), Sampler(2), Sampler(3)]
X, y = [1, 2], [0, 1]
out = {}
for round_ in range(2):
    for sampler in samplers:
        resample(sampler, X, y)
    out[f"round {round_}"] = len(RUNS)
resample(samplers[0], X, y)
out["after the loop"] = len(RUNS)
out["hits"] = resample.cache_info()["hits"]
RUNS.clear()
scaled(samplers[0], X)
scaled(samplers[0], X)
FACTOR = 3
got = scaled(samplers[0], X)
out["scaled runs"] = len(RUNS)
out["scaled"] = got
print(json.dumps(out))
"""


def test_a_loop_over_a_global_named_as_the_parameter_hits_on_repeat(tmp_path):
    (tmp_path / "loop.py").write_text(textwrap.dedent(_LOOP), encoding="utf-8")
    first = json.loads(run_python("loop.py", cwd=tmp_path).stdout.strip().splitlines()[-1])
    assert first["round 0"] == 3
    assert first["round 1"] == 3
    assert first["after the loop"] == 3
    assert first["hits"] == 4
    # A global the body does read still invalidates: FACTOR 2 -> 3.
    assert first["scaled runs"] == 2
    assert first["scaled"] == [3, 6]
    again = json.loads(run_python("loop.py", cwd=tmp_path).stdout.strip().splitlines()[-1])
    assert again["round 0"] == 0
    assert again["hits"] == 7
    assert again["scaled runs"] == 0
