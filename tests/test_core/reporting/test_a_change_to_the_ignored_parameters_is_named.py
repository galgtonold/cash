"""A call that misses because ``ignore=`` changed says so plainly.

``ignore=`` decides which arguments the key reads, so adding one re-keys
every call: the entries an earlier run stored no longer apply. The reason
read "it now uses ignored parameters workers".

Every step is a fresh process on one cache.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = textwrap.dedent("""
    import json, sys
    import cash

    @cash.cache({options})
    def nearest(n, workers=1):
        return n * 2

    explanation = nearest.explain(3, workers=4)
    nearest(3, workers=4)
    print(json.dumps({{"reasons": nearest.cache_info()["miss_reasons"], "details": explanation.details}}, default=str))
""")


def _run(proj, options=""):
    (proj / "job.py").write_text(JOB.format(options=options), encoding="utf-8")
    p = run_python("job.py", cwd=proj)
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    ("before", "after", "said"),
    [
        ("", "ignore=['workers']", "it now ignores parameter workers"),
        ("ignore=['workers']", "", "it no longer ignores parameter workers"),
        ("", "key=lambda n, workers=1: n", "it now keys its arguments with key="),
    ],
    ids=["added", "removed", "key"],
)
def test_the_miss_names_the_change(tmp_path, before, after, said):
    _run(tmp_path, before)
    out = _run(tmp_path, after)
    assert out["reasons"] == {"code or state changed": 1}, out
    text = json.dumps(out["details"])
    assert said in text, text
    assert "calls stored before the change run again" in text, text
    assert "uses ignored parameters" not in text, text
