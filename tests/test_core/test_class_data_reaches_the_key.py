"""The data behind a user class reaches the key, not only the class's code.

A class's source (and its bases' source) was keyed; what the class HOLDS and
what its code READS was not, so each shape below served the old result after
the value changed, with no warning:

* a module global read by an inherited method, a property, a mixin, a
  ``super()`` call into a base, an inherited ``__init__``, a
  ``cached_property``, ``__init_subclass__``, a metaclass ``__call__`` or a
  descriptor's ``__get__`` -- also for an instance passed as an argument;
* a class attribute set at run time (``Settings.RATE = int(sys.argv[1])``) or
  computed in a class body, read through ``self``, ``cls``, ``type(self)`` or
  ``getattr``;
* ``cfg.Cfg.RATE`` and ``cfg.Color.RED.value`` through ``import cfg``;
* a class with no source of its own: ``namedtuple(defaults=...)``,
  ``make_dataclass``, ``type(...)``, and a namedtuple whose fields were
  reordered (a hit rebuilt the stored tuple through the new class, swapping
  x and y);
* a callable instance held as a class attribute;
* a user transformer in a library pipeline whose ``transform`` reads a
  module constant.

Fresh process per run; the body prints RAN, so an unchanged second run proves
nothing over-invalidates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

MODELS = textwrap.dedent("""
    import collections, dataclasses, enum, functools

    RATE = {rate}

    class Base:
        def scale(self, x):
            return x * RATE

        @property
        def rate(self):
            return RATE

        def m(self):
            return RATE

    class Model(Base):
        pass

    class Mixin:
        def mixed(self):
            return RATE

    class Mixed(Mixin, Base):
        pass

    class Child(Base):
        def m(self):
            return super().m() + 1

    class Init:
        def __init__(self):
            self.v = RATE

    class InitChild(Init):
        pass

    class Cached:
        @functools.cached_property
        def v(self):
            return RATE

    class Registered:
        def __init_subclass__(cls):
            cls.k = RATE

    class Sub(Registered):
        pass

    class Meta(type):
        def __call__(cls, *a):
            obj = super().__call__(*a)
            obj.v = RATE
            return obj

    class Built(metaclass=Meta):
        pass

    class Desc:
        def __get__(self, obj, owner):
            return RATE

    class WithDesc:
        v = Desc()

    class Settings:
        RATE = 1

        def scaled(self, x):
            return x * self.RATE

        @classmethod
        def via_cls(cls, x):
            return x * cls.RATE

        def via_type(self, x):
            return x * type(self).RATE

    class Computed(Settings):
        RATE = RATE * 3

    class Cfg:
        RATE = {rate}
        LIST = [{rate}]

    class Color(enum.Enum):
        RED = {rate}

    P = collections.namedtuple("P", "a b", defaults=[{rate}])
    D = dataclasses.make_dataclass("D", [("a", int, dataclasses.field(default={rate}))])
    T = type("T", (), {{"k": {rate}}})
    Point = collections.namedtuple("Point", "{fields}")

    class CC:
        def __init__(self, k):
            self.k = k

        def __call__(self, x):
            return x * self.k

    class Svc:
        helper = CC({rate})

        def run(self, x):
            return self.helper(x)
""")

MAIN = textwrap.dedent("""
    import json, sys, time, warnings
    warnings.simplefilter("ignore")
    import cash
    import models
    import models as M
    from models import Model, Mixed, Child, InitChild, Cached, Sub, Built, WithDesc
    from models import Settings, Computed, P, D, T, Point, Svc

    Settings.RATE = int(sys.argv[1])

    def body():
        print("RAN", file=sys.stderr)  # @cash:assume-safe

    @cash.cache
    def base_method(x):
        body()
        m = Model()
        return (m.scale(x), m.rate)

    @cash.cache
    def mixin(x):
        body()
        return Mixed().mixed() * x

    @cash.cache
    def super_call(x):
        body()
        return Child().m() * x

    @cash.cache
    def inherited_init(x):
        body()
        return InitChild().v * x

    @cash.cache
    def cached_prop(x):
        body()
        return Cached().v * x

    @cash.cache
    def init_subclass(x):
        body()
        return Sub.k * x

    @cash.cache
    def metaclass_call(x):
        body()
        return Built().v * x

    @cash.cache
    def descriptor(x):
        body()
        return WithDesc().v * x

    @cash.cache
    def instance_arg(m, x):
        body()
        return m.scale(x)

    @cash.cache
    def runtime_attr(x):
        body()
        return Settings().scaled(x)

    @cash.cache
    def through_cls(x):
        body()
        return Settings.via_cls(x) + Settings().via_type(x)

    @cash.cache
    def via_getattr(x):
        body()
        return getattr(Settings, "RATE") * x

    @cash.cache
    def computed_in_body(x):
        body()
        return Computed().scaled(x)

    @cash.cache
    def module_class_const(x):
        body()
        return x * models.Cfg.RATE + models.Cfg.LIST[0] + M.Cfg().RATE

    @cash.cache
    def module_enum_value(x):
        body()
        return x * models.Color.RED.value

    @cash.cache
    def sourceless_classes(x):
        body()
        return x + P(0).b + D().a + T().k

    @cash.cache
    def namedtuple_fields(n):
        body()
        return Point(x=n, y=n * 10)

    @cash.cache
    def class_attr_callable(x):
        body()
        return Svc().run(x)

    print(json.dumps({
        "base_method": base_method(10),
        "mixin": mixin(10),
        "super_call": super_call(10),
        "inherited_init": inherited_init(10),
        "cached_prop": cached_prop(10),
        "init_subclass": init_subclass(10),
        "metaclass_call": metaclass_call(10),
        "descriptor": descriptor(10),
        "instance_arg": instance_arg(Model(), 10),
        "runtime_attr": runtime_attr(10),
        "through_cls": through_cls(10),
        "via_getattr": via_getattr(10),
        "computed_in_body": computed_in_body(10),
        "module_class_const": module_class_const(10),
        "module_enum_value": module_enum_value(10),
        "sourceless_classes": sourceless_classes(10),
        "namedtuple_fields": list(namedtuple_fields(1)._asdict().items()),
        "class_attr_callable": class_attr_callable(10),
    }))
""")


def _project(tmp_path, rate, fields="x y"):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "models.py").write_text(MODELS.format(rate=rate, fields=fields), encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    return proj


def _run(tmp_path, proj, argv, *, disable=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if disable:
        env["CASH_DISABLE"] = "1"
    out = subprocess.run([sys.executable, "main.py", str(argv)], cwd=str(proj), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def test_every_shape_sees_the_change(tmp_path):
    """THE BUG: each of these served the result computed under the old value."""
    proj = _project(tmp_path, rate=1)
    first, ran = _run(tmp_path, proj, 1)
    assert ran == len(first)

    _project(tmp_path, rate=5, fields="y x")
    second, ran = _run(tmp_path, proj, 3)
    expected, _ = _run(tmp_path, proj, 3, disable=True)

    stale = sorted(name for name in expected if second[name] != expected[name])
    assert not stale, f"served the old result: {stale}"
    assert ran == len(first)


def test_nothing_changed_still_hits(tmp_path):
    """The control: keying class data and what class code reads costs no hit."""
    proj = _project(tmp_path, rate=1)
    first, _ = _run(tmp_path, proj, 1)
    again, ran = _run(tmp_path, proj, 1)

    assert again == first
    assert ran == 0, f"{ran} call(s) recomputed with nothing changed"


def test_a_class_attribute_changed_in_process_is_seen():
    """`D.K = 100` between two calls of one process."""
    import cash

    c = cash.Cash(backend=cash.InMemoryBackend())

    class D:
        K = 1

        def scaled(self, x):
            return x * self.K

    @c.cache
    def f(x):
        return D().scaled(x)

    assert f(10) == 10
    D.K = 100
    assert f(10) == 1000


def test_a_counter_the_class_keeps_does_not_miss_every_call():
    """A class attribute its own code writes is state, not an input."""
    import cash

    c = cash.Cash(backend=cash.InMemoryBackend())
    runs = []

    class Counter:
        calls = 0
        FACTOR = 2

        def bump(self, x):
            type(self).calls += 1
            return x * self.FACTOR

    @c.cache
    def f(x):
        runs.append(x)  # @cash:assume-safe
        return Counter().bump(x)

    for _ in range(3):
        assert f(3) == 6
    assert len(runs) == 1


def test_an_unhashable_lock_on_a_class_is_not_a_warning(recwarn):
    """A lock is no result input: left out of the class's data quietly."""
    import threading

    import cash

    c = cash.Cash(backend=cash.InMemoryBackend())

    class Guarded:
        _lock = threading.Lock()
        K = 3

        def get(self, x):
            with self._lock:
                return x * self.K

    @c.cache
    def f(x):
        return Guarded().get(x)

    assert f(2) == 6
    assert f(2) == 6
    assert not [w for w in recwarn if "KEY-UNHASHABLE" in str(w.message)]
    Guarded.K = 4
    assert f(2) == 8


SKLEARN_LIB = textwrap.dedent("""
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import make_pipeline

    V = {v}

    class ScaleV(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            return X * V

    PIPE = make_pipeline(ScaleV()).fit([[0.0]])
""")

SKLEARN_MAIN = textwrap.dedent("""
    import json, sys, time, warnings
    warnings.simplefilter("ignore")
    import cash, numpy as np
    from lib import PIPE

    @cash.cache
    def predict(x):
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        return float(PIPE.transform(np.array([[x]]))[0, 0])

    print(json.dumps({"predict": predict(2.0)}))
""")


def test_a_pipeline_transformer_reading_a_constant_sees_the_change(tmp_path):
    """sklearn wraps `transform`: the constant read inside the user's own function."""
    pytest.importorskip("sklearn")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text(SKLEARN_MAIN, encoding="utf-8")
    (proj / "lib.py").write_text(SKLEARN_LIB.format(v=1), encoding="utf-8")
    first, _ = _run(tmp_path, proj, 0)
    assert first == {"predict": 2.0}
    again, ran = _run(tmp_path, proj, 0)
    assert again == first and ran == 0

    (proj / "lib.py").write_text(SKLEARN_LIB.format(v=5), encoding="utf-8")
    second, ran = _run(tmp_path, proj, 0)
    assert second == {"predict": 10.0}
    assert ran == 1
