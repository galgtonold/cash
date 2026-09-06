"""An unseeded draw is replayed however cheap it is.

`docs/known-limitations.md` used to claim the opposite: that a statement below
the ~10 ms cost floor "is never cached at all, so `random.random()` *does* give
you a new number each time". The inference is wrong — freezing does not require
caching. The RNG rewind replays the value whether or not it was ever stored, so
the cost floor governs persistence, not whether you see the same number twice.

Pinned here because the wrong version is the intuitive one: anybody reasoning
from "not cached, therefore recomputed" will write it again.

**Compare the printed number, never the raw output.** `get_output` also carries
the stderr warning, and that text legitimately differs between runs
(`RANDOM-UNSEEDED` on the first, `RANDOM-REPLAYED` after). Comparing raw output
measures the warning rather than the draw — which is how the first version of
this probe reported "it redrew" while the value was frozen.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = "import cash\nimport random\nimport numpy as np\nimport time"

EXPENSIVE = (
    "time.sleep(0.2)\n"
    "v = float(np.random.rand(200_000).sum())\n"
    "print('EXPENSIVE', v)"
)
TRIVIAL = "v = random.random()\nprint('TRIVIAL', v)"


def _value(raw: str, tag: str) -> str:
    for line in raw.splitlines():
        if line.startswith(tag + " "):
            return line.split(None, 1)[1].strip()
    raise AssertionError(f"no {tag!r} line in output: {raw!r}")


def _two_runs(nb_runner, src: str, tag: str) -> tuple[str, str]:
    nb_runner.create_notebook([SETUP, src])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    first = _value(nb_runner.get_output(2), tag)
    nb_runner.run_all()
    return first, _value(nb_runner.get_output(2), tag)


def test_an_expensive_unseeded_draw_is_frozen(nb_runner):
    """The control. If this ever stops freezing, the test below proves nothing
    — it would be passing because the rewind is off, not because cheap draws
    are held."""
    first, second = _two_runs(nb_runner, EXPENSIVE, "EXPENSIVE")
    assert first == second, "the expensive draw was redrawn; the rewind is not running"


def test_a_trivial_unseeded_draw_is_frozen_just_the_same(nb_runner):
    """The claim the docs got wrong. Nothing in this cell but the draw and a
    print, so no other statement can be pushing it over the cost floor."""
    first, second = _two_runs(nb_runner, TRIVIAL, "TRIVIAL")
    assert first == second, (
        "a sub-cost-floor draw was redrawn. If this is now intended, "
        "docs/known-limitations.md must be updated to match — it currently "
        "documents the frozen behaviour."
    )
