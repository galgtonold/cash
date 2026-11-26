"""A global the call trace digests that holds a module's global RNG hits in
a new process.

Every ``scipy.stats`` distribution holds ``np.random.mtrand._rand`` (its
``_random_state``), the generator ``np.random.normal`` draws from. A call
whose trace read such a global -- lmfit's ``conf_interval`` reads
``scipy.stats.f`` -- digested that generator's state, fresh entropy in
every process, so the call missed after each restart with "global f
changed". The trace now digests the module's global RNG by the module's
name, as the key does; the global RNG channel accounts for its state. A
generator of the value's own still digests by its state, and seeded draws
behave as plain Python.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


DIGEST = """
    import random
    import sys

    import numpy as np

    from cash.tracking.trace_items import value_digest


    class Holder:
        def __init__(self, rng):
            self.rng = rng


    print(
        value_digest(Holder(np.random.mtrand._rand)),
        value_digest(Holder(random._inst)),
        value_digest(np.random.mtrand._rand),
        value_digest(Holder(np.random.RandomState(int(sys.argv[1])))),
    )
"""


def test_the_module_rng_digests_alike_in_every_process(tmp_path):
    pytest.importorskip("numpy")
    (tmp_path / "digest.py").write_text(textwrap.dedent(DIGEST), encoding="utf-8")

    def digests(seed):
        return run_python("digest.py", str(seed), cwd=tmp_path).stdout.split()

    first, again, other = digests(0), digests(0), digests(1)
    assert first == again
    assert not any(d.startswith("opaque:") for d in first)
    # A generator of the value's own is data: by its state.
    assert first[:3] == other[:3] and first[3] != other[3]


def test_a_scipy_distribution_digests_alike_in_every_process(tmp_path):
    pytest.importorskip("scipy.stats")
    (tmp_path / "digest.py").write_text(
        "from scipy.stats import f\nfrom cash.tracking.trace_items import value_digest\nprint(value_digest(f))\n",
        encoding="utf-8",
    )
    runs = {run_python("digest.py", cwd=tmp_path).stdout.strip() for _ in range(2)}
    assert len(runs) == 1 and not next(iter(runs)).startswith("opaque:"), runs


HELPERS = """
    import numpy as np
    from scipy.stats import f


    class Holder:
        def __init__(self, rng):
            self.rng = rng
            self.k = 2.0


    HOLD = Holder(np.random.mtrand._rand)


    def conf(z):
        return float(f.ppf(z, 2, 3)) + HOLD.k


    def draw(z):
        return float(HOLD.rng.random()) + z


    def draw_directly(z):
        return float(np.random.random()) + z
"""

MAIN = """
    import sys
    import warnings

    import numpy as np

    import cash
    import helpers

    warnings.simplefilter("always")


    @cash.cache
    def run(z, name):
        # By a run-time name: what it reaches is the trace's to name.
        return getattr(helpers, name)(z)


    name, seed = sys.argv[1], sys.argv[2]
    if seed != "none":
        np.random.seed(int(seed))
    with warnings.catch_warnings(record=True) as caught:
        got = [run(0.5, name)] if name == "conf" else [run(0.0, name), run(0.0, name)]
        after = float(np.random.random())
    same = None
    if seed != "none":
        np.random.seed(int(seed))
        same = [*got, after] == [float(np.random.random()) for _ in range(3)]
    info = run.cache_info()
    codes = sorted({str(getattr(w.message, "code", "")) for w in caught} - {""})
    print(same, info["hits"], info["misses"], info["miss_reasons"], codes)
"""


def _runs(folder: Path, *argvs: tuple[str, ...]) -> list[str]:
    folder.mkdir()
    (folder / "helpers.py").write_text(textwrap.dedent(HELPERS), encoding="utf-8")
    (folder / "main.py").write_text(textwrap.dedent(MAIN), encoding="utf-8")
    env = {"CASH_TRACE": "1"}
    return [run_python("main.py", *argv, cwd=folder, env=env).stdout.strip().splitlines()[-1] for argv in argvs]


def test_a_traced_call_reading_a_scipy_distribution_hits_in_a_new_process(tmp_path):
    pytest.importorskip("scipy.stats")
    assert _runs(tmp_path / "conf", ("conf", "none"), ("conf", "none")) == [
        "None 0 1 {'no entry yet': 1} []",
        "None 1 0 {} []",
    ]


def test_draws_from_the_held_module_rng_behave_as_drawing_from_the_module(tmp_path):
    pytest.importorskip("scipy.stats")
    argvs = [("0",), ("0",), ("1",), ("none",), ("none",)]
    held = _runs(tmp_path / "held", *[("draw", *a) for a in argvs])
    direct = _runs(tmp_path / "direct", *[("draw_directly", *a) for a in argvs])
    # Seeded: each draw is plain Python's, the stream moved on after a hit.
    assert all(line.startswith("True ") for line in held[:3]), held
    # Hits, misses and warnings, seeded and unseeded, are those of drawing
    # from ``np.random`` itself.
    assert held == direct
    assert held[1].split()[1] != "0", "the second process must hit"
