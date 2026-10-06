"""A cached function reached through a value keys its whole state.

A method of an object passed in, a property of a module-level object, a
class passed in, a bound method or a helper passed as an argument, a lambda
in a module-level list: each reaches the cached ``price``, which reads
``RATE``. The value path keyed ``price``'s code alone, so editing ``RATE``
served the old result, where calling ``price`` by name recomputed. So did a
change to a static ``depends_on=`` source of the callee. Fresh process per
run, the constant edited in between.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.timeout(300)

JOB = """
    import os, sys
    import cash
    from cash import DataSource

    RATE = {rate}

    class Version(DataSource):
        def get_id(self):
            return "version"

        def state_token(self):
            return os.environ["VERSION"]

    @cash.cache
    def price(x):
        return x * RATE

    @cash.cache(depends_on=[Version()])
    def load(x):
        return x  # reads the versioned data in a way cash does not see

    class Pricer:
        def run(self, x):
            return price(x)

        @classmethod
        def crun(cls, x):
            return price(x)

        @property
        def ten(self):
            return price(10)

        def loaded(self, x):
            return load(x)

    PRICER = Pricer()
    STEPS = [lambda x: price(x)]

    def helper(x):
        return price(x)

    @cash.cache
    def obj_arg(p, x):
        return p.run(x)

    @cash.cache
    def global_property():
        return PRICER.ten

    @cash.cache
    def class_arg(c, x):
        return c.crun(x)

    @cash.cache
    def call(f, x):
        return f(x)

    @cash.cache
    def run_steps(x):
        return STEPS[0](x)

    @cash.cache
    def r_method(p, x):
        print("[R_METHOD]", file=sys.stderr)  # @cash:assume-safe
        return p.loaded(x)

    print(obj_arg(Pricer(), 10), global_property(), class_arg(Pricer, 10), call(PRICER.run, 10),
          call(helper, 10), run_steps(10), r_method(Pricer(), 1))
"""


def test_an_edit_the_callee_sees_recomputes_the_caller(tmp_path):
    job = tmp_path / "job.py"
    job.write_text(textwrap.dedent(JOB).format(rate=2), encoding="utf-8")
    first = run_python("job.py", cwd=tmp_path, env={"VERSION": "1"})
    assert first.stdout.split() == ["20"] * 6 + ["1"]
    again = run_python("job.py", cwd=tmp_path, env={"VERSION": "1"})
    assert "[R_METHOD]" not in again.stderr
    moved = run_python("job.py", cwd=tmp_path, env={"VERSION": "2"})
    assert "[R_METHOD]" in moved.stderr  # the callee's depends_on= source moved
    job.write_text(textwrap.dedent(JOB).format(rate=3), encoding="utf-8")
    assert run_python("job.py", cwd=tmp_path, env={"VERSION": "2"}).stdout.split() == ["30"] * 6 + ["1"]
