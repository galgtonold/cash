"""CAS-220 blast-radius guard: fixing in-loop draws must not stop loops caching.

The CAS-220 defect itself (a chart drawn inside a ``for`` body is silently
written blank on warm re-runs) is NOT covered here, and deliberately so: this
suite drives ``NotebookClient``, which does not reproduce it. A test written
here passes with and without the fix and would be pure decoration. The bug's
oracle is the real-jupyter-server reproducer (`repro_blank_charts_min3.py`);
the mechanism is pinned by unit tests in
``tests/test_notebook/test_loop_draw_receiver_routing.py``.

What this file DOES cover is the other side of the fix. CAS-220 is repaired by
routing in-loop method calls to skip-cache when the receiver is a live
Figure/Axes. If that condition were drawn any wider it would disable
per-iteration caching for ordinary loop bodies -- trading a correctness bug for
a silent performance regression. These tests fail if that happens.
"""
import pytest

pytestmark = pytest.mark.libraries

SETUP = "import cash\n%cash_on"


@pytest.mark.timeout(180)
def test_an_ordinary_loop_body_still_restores_on_a_warm_run(nb_runner):
    """A non-drawing accumulator loop must not re-execute its body when warm.

    The call counter is an in-kernel list rather than a file: appending to a log
    from inside the body is itself file I/O, which suppresses caching on its own
    and would make this test measure its own instrumentation rather than the
    loop.
    """
    nb_runner.create_notebook([
        SETUP,
        "CALLS = []\n"
        "def work(x):\n"
        "    CALLS.append(x)\n"
        "    return x * 2\n"
        "print('defined')",
        "out = []\n"
        "for i in range(5):\n"
        "    out.append(work(i))\n"
        "print('sum', sum(out))",
        "print('calls', len(CALLS))",
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    cold = nb_runner.get_output(4)

    nb_runner.run_all()
    warm = nb_runner.get_output(4)

    assert 'calls' in cold, f"counter cell produced no reading: {cold!r}"
    assert warm.strip() == cold.strip(), (
        f"warm pass changed the call count ({cold.strip()!r} -> {warm.strip()!r}): "
        f"per-iteration caching for an ordinary, non-drawing loop regressed"
    )


@pytest.mark.timeout(180)
def test_a_loop_calling_a_dataframe_method_still_caches(nb_runner):
    """A non-coupled receiver in a loop body must keep today's behaviour.

    ``df.head()`` is a genuinely receiver-pure call. The identity-coupled
    predicate must return False for a DataFrame, so this loop is untouched by
    the fix -- the same discriminator CAS-194 used to tell ``ax.hist()`` (draws)
    apart from ``df.hist()`` (pure).
    """
    pytest.importorskip("pandas")

    nb_runner.create_notebook([
        SETUP,
        "import pandas as pd\n"
        "df = pd.DataFrame({'v': range(20)})\n"
        "SEEN = []\n"
        "print('setup')",
        "for _ in range(3):\n"
        "    SEEN.append(len(df.head()))\n"
        "print('seen', SEEN)",
        "print('n', len(SEEN))",
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    cold = nb_runner.get_output(4)

    nb_runner.run_all()
    warm = nb_runner.get_output(4)

    assert warm.strip() == cold.strip(), (
        f"warm pass changed the accumulator ({cold.strip()!r} -> {warm.strip()!r}): "
        f"a pure DataFrame method call in a loop body is being treated as a draw"
    )
