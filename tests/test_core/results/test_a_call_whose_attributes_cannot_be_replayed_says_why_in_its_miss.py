"""A call that binds an attribute a hit cannot replay is not stored, and its
miss says why: the ``cash.calls`` line and ``explain`` name the replay's
refusal, as the IMPURE-OBSERVED-EFFECTS warning does when only the first
call shows the change, not only that the argument changed in place."""

from __future__ import annotations

import textwrap
import warnings

from tests._scripts import run_python


class Opaque:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return (Opaque, (self.value,))


class Est:
    def __init__(self):
        self.k = 2


REASON = (
    "a hit cannot replay it: the object it bound to est.coef_ is a Opaque, which pickles through its own __reduce__"
)


def test_explain_names_the_replay_refusal(disk_cash):
    @disk_cash.cache
    def fit(est, x):
        est.coef_ = Opaque(x * est.k)
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit(Est(), 3)
        fit(Est(), 3)
    assert fit.cache_info()["miss_reasons"] == {"no entry yet": 1, "not stored last time": 1}
    why = fit.explain(Est(), 3).details["why"]
    assert "it changed its argument 'est' in place, and " + REASON in why, why


def test_the_cash_calls_line_names_the_replay_refusal(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            import cash

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)

            class Opaque:
                def __init__(self, value):
                    self.value = value

                def __reduce__(self):
                    return (Opaque, (self.value,))

            class Est:
                def __init__(self):
                    self.k = 2

            @c.cache
            def fit(est, x):
                est.coef_ = Opaque(x * est.k)
                return 1

            fit(Est(), 3)
            fit(Est(), 3)
            """
        ),
        encoding="utf-8",
    )
    out = run_python(script, tmp_path / ".cash", cwd=tmp_path, env={"CASH_DEBUG": "1"})
    lines = [line for line in out.stderr.splitlines() if line.startswith("cash.calls:")]
    assert len(lines) == 2, out.stderr
    assert "not stored: it changed its argument 'est' in place, and " + REASON in lines[0], lines[0]
    # The second miss names the refusal as its reason, once.
    assert lines[1].count(REASON) == 1, lines[1]
    assert "not stored last time: it changed its argument 'est' in place, and " + REASON in lines[1], lines[1]
    assert "not stored again, for the same reason" in lines[1], lines[1]
    assert "which a hit would not do" not in out.stderr
