"""A cached call must not hijack pyplot's current figure (CAS-243).

Found by adversarial probing, and it wrote a genuinely wrong PNG.

`statement/processor.py` refuses to cache an identity-coupled object because
the RAM tier deep-copies on store and `Figure.__setstate__` re-registers the
COPY as pyplot's *current figure* — so a later bare `plt.savefig()` writes the
cache's snapshot instead of the figure the user drew on, on the FIRST run and
silently. Routing calls through the decorator skipped that guard.

The measurement that caught it, and that this pins: save the same figure twice
in one cell, once through the object and once through the global. They must be
byte-identical. Before the fix they differed (5437B vs 8460B) and
`plt.gcf() is fig` was False.

Deliberately end-to-end rather than a unit test: the defect only appears once a
real kernel, a real backend tier and real pyplot state are all in play.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

pytest.importorskip("matplotlib")

_HEAD = ("import time, matplotlib\nmatplotlib.use('Agg')\n"
         "import matplotlib.pyplot as plt")
# Creation is cached; the DRAWING happens later, in a separate statement. That
# split is the vulnerable arrangement — a function that draws and returns in one
# go cannot expose it.
_DEFS = ("def new_fig(n):\n    time.sleep(0.2)\n    fig, ax = plt.subplots()\n"
         "    return fig, ax")


def _run(nb_runner, tmp_path, loop_src, tag):
    obj, glb = tmp_path / f"{tag}_o.png", tmp_path / f"{tag}_g.png"
    nb_runner.create_notebook([
        _HEAD,
        _DEFS,
        "holder = []",
        loop_src,
        "fig, ax = holder[-1]\nax.bar(['a', 'b'], [3, 6])",
        f"fig.savefig(r'{obj}')\nplt.savefig(r'{glb}')\n"
        "print('IS_CURRENT', plt.gcf() is fig)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(6)
    assert obj.exists() and glb.exists(), "no PNG written; the probe is vacuous"
    return obj.read_bytes() == glb.read_bytes(), out


@pytest.mark.xfail(
    reason="CAS-243 Task 5: CallUnit._storable is a deliberate stub returning "
    "True (post-execution refusals -- including this identity-coupled-result "
    "guard -- are Task 6's job, not reimplemented ahead of it here). The "
    "decorator-fallback path (CallCache.resolve with no registered site) "
    "still applies _is_storable/identity_coupled_reason and is unaffected; "
    "only a call routed through a real CallSite lost the guard. Remove this "
    "marker once Task 6 lands CallUnit's own post-execution refusal.",
    strict=True,
)
def test_intercepted_figure_call_does_not_hijack_pyplot(nb_runner, tmp_path):
    same, out = _run(
        nb_runner, tmp_path,
        "# @cash:cache-calls\nfor n in [1]:\n    holder.append(new_fig(n))",
        "directive",
    )
    assert "IS_CURRENT True" in out, (
        "plt.gcf() is not the figure the user holds -- a bare plt.savefig() "
        f"will write the cache's copy instead:\n{out}"
    )
    assert same, (
        "fig.savefig() and plt.savefig() wrote DIFFERENT images: pyplot's "
        "current figure is a restored copy, so the user's drawing is lost"
    )


def test_ground_truth_without_caching(nb_runner, tmp_path):
    """Positive control: the arrangement is sound when nothing is cached.

    Without it, a bug that broke figures generally would still satisfy the
    test above by making both arms equally wrong.
    """
    same, out = _run(
        nb_runner, tmp_path,
        "%cash_off\nfor n in [1]:\n    holder.append(new_fig(n))",
        "plain",
    )
    assert "IS_CURRENT True" in out, out
    assert same, "the control arm is broken; the comparison proves nothing"
