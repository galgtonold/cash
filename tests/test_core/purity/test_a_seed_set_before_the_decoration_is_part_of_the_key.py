"""A seed set before the decoration, or before ``import cash``, is part of the key.

``np.random.seed(int(sys.argv[1]))`` at module level, above a decorated
function that draws, was never seen: the seed watch began at the first
decoration, after the seed. Run with seed 1 and then seed 2 over one cache,
both printed seed 1's draw, and the decoration said the draw was unseeded.
The same held for ``random.seed``, and for a seed set before ``import cash``.
And no seeded call hit in the next process: the "this function draws" marker
a later process keys by was kept in RAM only.

Each program runs in its own process over one cache, as a script is run
again with another argument. A truly unseeded stream is the control: its
first value is frozen and replayed, with RANDOM-UNSEEDED.
"""

from __future__ import annotations

import json
import random
import textwrap

import pytest

from tests._scripts import run_python

np = pytest.importorskip("numpy")

#: Counts executions of the body: an append a hit skips (see the testing guide).
COUNT = """
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")  # @cash:assume-safe
    os.close(fd)  # @cash:assume-safe
"""

SEED_AFTER_IMPORT = """
import json, os, sys, warnings
warnings.simplefilter("always")
import numpy as np, random, cash
np.random.seed(int(sys.argv[1]))
random.seed(int(sys.argv[1]))
"""

SEED_BEFORE_IMPORT = """
import json, os, sys, warnings
warnings.simplefilter("always")
import numpy as np, random
np.random.seed(int(sys.argv[1]))
random.seed(int(sys.argv[1]))
import cash
"""

UNSEEDED = """
import json, os, sys, warnings
warnings.simplefilter("always")
import numpy as np, random, cash
"""

BODY = """
@cash.cache
def draw_np():
COUNT    return float(np.random.normal())

@cash.cache
def draw_std():
COUNT    return random.random()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = {"np": draw_np(), "std": draw_std()}
out["unseeded_warned"] = any("RANDOM-UNSEEDED" in str(w.message) for w in caught)
print(json.dumps(out))
"""


def _program(tmp_path, head: str):
    script = tmp_path / "run.py"
    log = tmp_path / "calls.log"
    script.write_text(
        textwrap.dedent(head) + f"LOG = {str(log)!r}\n" + textwrap.dedent(BODY).replace("COUNT", COUNT.lstrip("\n")),
        encoding="utf-8",
    )

    def run(seed: int) -> dict:
        before = log.stat().st_size if log.exists() else 0
        done = run_python(script, str(seed), cwd=tmp_path, cache_dir=tmp_path / ".cash", timeout=180)
        got = json.loads(done.stdout.strip().splitlines()[-1])
        got["executions"] = (log.stat().st_size if log.exists() else 0) - before
        got["stderr"] = done.stderr
        return got

    return run


def _plain(seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)
    return {"np": float(np.random.normal()), "std": random.random()}


@pytest.mark.timeout(300)
@pytest.mark.parametrize("head", [SEED_AFTER_IMPORT, SEED_BEFORE_IMPORT], ids=["after_import", "before_import"])
def test_each_seed_gets_its_own_draw_and_a_repeated_seed_hits(tmp_path, head):
    run = _program(tmp_path, head)
    runs = [run(seed) for seed in (1, 2, 1, 1)]
    for seed, got in zip((1, 2, 1, 1), runs):
        want = _plain(seed)
        assert (got["np"], got["std"]) == (want["np"], want["std"]), f"seed {seed} served another seed's draw: {got}"
        assert not got["unseeded_warned"] and "RANDOM-UNSEEDED" not in got["stderr"], got["stderr"]
    assert runs[-1]["executions"] == 0, "the same seed a third time recomputed instead of hitting"


@pytest.mark.timeout(300)
def test_an_unseeded_draw_is_still_frozen_and_warned(tmp_path):
    """Control: no seed anywhere, so the first value is replayed."""
    run = _program(tmp_path, UNSEEDED)
    first, second = run(1), run(2)
    assert (second["np"], second["std"]) == (first["np"], first["std"])
    assert second["executions"] == 0
    assert "RANDOM-UNSEEDED" in first["stderr"]
