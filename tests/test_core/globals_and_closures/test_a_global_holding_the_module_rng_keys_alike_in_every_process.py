"""A global that holds a module's global RNG keys alike in every process.

``from scipy.stats import norm`` binds a distribution object that holds
``np.random.mtrand._rand``, the generator ``np.random.normal`` draws from. A
cached function reading ``norm`` keyed on that generator's state, which is
fresh entropy in every process: every run missed with "global norm changed".
The module's global RNG is the global channel's (seeded, captured and
replayed as the module's), so a value holding it keys it by the module's
name. A generator of the value's own still keys by its state.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core]

JOB = """
    import random
    import sys

    import numpy as np

    import cash


    class Dist:
        def __init__(self, rng):
            self._random_state = rng
            self.loc = 0.0

        def shift(self, x):
            return x + self.loc


    HOLDERS = {
        "numpy": Dist(np.random.mtrand._rand),
        "random": Dist(random._inst),
        "own": Dist(np.random.RandomState(int(sys.argv[2]) if len(sys.argv) > 2 else 0)),
    }
    DIST = HOLDERS[sys.argv[1]]


    @cash.cache
    def conf(z):
        return DIST.shift(z)


    conf(1.0)
    info = conf.cache_info()
    print(info["misses"], info["changed"])
"""


def _runs(folder: Path, *argvs: tuple[str, ...]) -> list[str]:
    folder.mkdir()
    (folder / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    return [run_python("job.py", *argv, cwd=folder).stdout.strip().splitlines()[-1] for argv in argvs]


def test_a_value_holding_the_module_rng_hits_in_a_new_process(tmp_path):
    pytest.importorskip("numpy")
    assert _runs(tmp_path / "numpy", ("numpy",), ("numpy",)) == ["1 {}", "0 {}"]
    assert _runs(tmp_path / "random", ("random",), ("random",)) == ["1 {}", "0 {}"]
    # A generator of the value's own is data it was built with: by its state.
    assert _runs(tmp_path / "own", ("own", "0"), ("own", "0"), ("own", "1")) == [
        "1 {}",
        "0 {}",
        "1 {'global DIST changed': 1}",
    ]


def test_a_scipy_distribution_global_hits_in_a_new_process(tmp_path):
    pytest.importorskip("scipy.stats")
    (tmp_path / "job.py").write_text(
        textwrap.dedent(
            """
            import cash
            from scipy.stats import norm


            @cash.cache
            def conf(z):
                return float(norm.ppf(0.5 + z / 2.0))


            conf(0.95)
            print(conf.cache_info()["misses"])
            """
        ),
        encoding="utf-8",
    )
    runs = [run_python("job.py", cwd=tmp_path).stdout.strip().splitlines()[-1] for _ in range(2)]
    assert runs == ["1", "0"]
