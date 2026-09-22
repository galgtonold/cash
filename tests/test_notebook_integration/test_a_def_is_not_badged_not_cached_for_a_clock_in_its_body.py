"""A ``def`` is not refused for what its body does when CALLED.

Round 30, r30s1: "every ``def`` whose body calls time.time() shows
``NOT CACHED: def f(...)  (0.00s) - time.time``. A def is never something I
wanted cached; the row reads as if cash refuses to cache my function. The
calls to that function ARE cached. Noise, one row per def per cell."

The clock runs when the function is called, and the call is judged on its
own. What runs at definition time -- decorators, default arguments,
annotations -- still counts, as does a class body.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\n%cash_badge print\nimport time"


def test_a_clock_in_the_body_does_not_refuse_the_def(nb_runner):
    nb_runner.create_notebook([
        SETUP,
        "def f(x):\n    t = time.time()\n    return x + t * 0\nprint('F', f(1))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert "F 1" in nb_runner.get_output(2), raw
    assert "NOT CACHED: def f" not in raw, raw


def test_the_call_is_still_refused(nb_runner):
    """Control: reading the clock is still not cacheable where it happens."""
    nb_runner.create_notebook([
        SETUP,
        "def f(x):\n    return x + time.time()\nv = f(1)",
        "print('V', v)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(3).split("V ")[1].splitlines()[0]

    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.get_output(3).split("V ")[1].splitlines()[0] != first, (
        "a call reading the clock was served from the cache:\n"
        + nb_runner.get_raw_output(2))


@pytest.mark.parametrize("code, label", [
    ("def h(x, when=time.time()):\n    return x\nprint('H', h(1))", "def h"),
    ("class C:\n    made = time.time()\nprint('C', C.made > 0)", "class C"),
])
def test_what_runs_at_definition_time_still_counts(nb_runner, code, label):
    nb_runner.create_notebook([SETUP, code])
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert f"NOT CACHED: {label}" in raw, raw
