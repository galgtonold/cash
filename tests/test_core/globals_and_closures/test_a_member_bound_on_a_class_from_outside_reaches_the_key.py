"""A method or getter bound on a class from outside its body reaches the key.

A class a cached function calls is keyed by its source text, and
``Thing.prop = property(newp)`` (or a lambda getter, a method, a classmethod,
a staticmethod, a ``cached_property``'s ``.func``) leaves that text as it was:
the call kept serving the old getter's result, in this process and the next.
Each variant has its own class, so one change cannot make another's call miss.
The script runs each call several times (the memoised hit paths too) and must
print what plain Python prints; a second process with the classes patched
before any call must not be served what the unpatched one stored.
"""

from __future__ import annotations

import functools
import textwrap

import pytest

from cash.loaded_code import code_objects_of
from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

CLASS = """
class {name}:
    def __init__(self, x):
        self.x = x

    @property
    def prop(self):
        return self.x + 1

    def m(self):
        return self.x + 2

    @classmethod
    def cm(cls, x):
        return x + 3

    @staticmethod
    def sm(x):
        return x + 4

    @functools.cached_property
    def cp(self):
        return self.x + 5

    def other(self):
        return self.x + 6

"""

NAMES = ("PropDef", "PropLambda", "Method", "ClassMeth", "StaticMeth", "CachedFunc", "CachedOther", "PropOther")

SCRIPT = (
    "import functools\nimport sys\n\nimport cash\n\n"
    + "".join(CLASS.format(name=n) for n in NAMES)
    + textwrap.dedent("""
    def new_prop(self):
        return self.x + 100


    def new_m(self):
        return self.x + 300


    def new_cm(cls, x):
        return x + 400


    def patch(step):
        if step == 1:
            PropDef.prop = property(new_prop)
            PropLambda.prop = property(lambda self: self.x + 200)
            Method.m = new_m
        elif step == 2:
            ClassMeth.cm = classmethod(new_cm)
            StaticMeth.sm = staticmethod(lambda x: x + 500)
            CachedFunc.cp.func = lambda self: self.x + 600
        elif step == 3:
            CachedOther.cp.func = vars(CachedOther)["other"]
            PropOther.prop = property(vars(PropOther)["other"])


    @cash.cache
    def prop_def(x):
        return PropDef(x).prop


    @cash.cache
    def prop_lambda(x):
        return PropLambda(x).prop


    @cash.cache
    def method(x):
        return Method(x).m()


    @cash.cache
    def class_meth(x):
        return ClassMeth.cm(x)


    @cash.cache
    def static_meth(x):
        return StaticMeth.sm(x)


    @cash.cache
    def cached_func(x):
        return CachedFunc(x).cp


    @cash.cache
    def cached_other(x):
        return CachedOther(x).cp


    def helper(x):
        return PropOther(x).prop


    @cash.cache
    def prop_other(x):
        return helper(x)


    @cash.cache
    def as_argument(cls, x):
        return cls(x).cp


    def show():
        return [
            [
                prop_def(1), prop_lambda(1), method(1), class_meth(1), static_meth(1),
                cached_func(1), cached_other(1), prop_other(1), as_argument(CachedOther, 1),
            ]
            for _ in range(3)
        ]


    if sys.argv[1] == "all":
        for step in (1, 2, 3):
            patch(step)
        print(show())
    else:
        print(show())
        for step in (1, 2, 3):
            patch(step)
            print(show())
""")
)


def _run(tmp_path, mode, **env):
    return run_python("main.py", mode, cwd=tmp_path, env=env).stdout


def test_a_member_bound_from_outside_matches_plain_python(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    plain = _run(tmp_path, "steps", CASH_DISABLE="1")
    cached = _run(tmp_path, "steps")
    assert cached == plain, f"cash printed\n{cached}\nplain Python printed\n{plain}"
    # And a second process, from the entries the first stored.
    again = _run(tmp_path, "steps")
    assert again == plain, f"the second process printed\n{again}\nplain Python printed\n{plain}"


def test_a_process_patched_before_its_calls_is_not_served_the_unpatched_results(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    plain = _run(tmp_path, "all", CASH_DISABLE="1")
    _run(tmp_path, "none")
    patched = _run(tmp_path, "all")
    assert patched == plain, f"cash printed\n{patched}\nplain Python printed\n{plain}"


def test_the_code_a_class_runs_includes_its_cached_property_and_property_setter():
    """What a memo per class is checked against sees a getter set in place."""

    class Holder:
        @functools.cached_property
        def cp(self):
            return 1

        @property
        def prop(self):
            return 2

        @prop.setter
        def prop(self, value):
            pass

    def swapped(self):
        return 3

    before = code_objects_of(Holder)
    assert vars(Holder)["cp"].func.__code__ in before
    assert vars(Holder)["prop"].fset.__code__ in before
    Holder.cp.func = swapped
    assert swapped.__code__ in code_objects_of(Holder)
