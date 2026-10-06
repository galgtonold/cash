"""A class held in data -- a registry, a list, a partial, a closure -- is keyed by its code.

A class pickles by name, which an edit does not move. ``models.REG[name]()``
with ``REG = {"double": Double}``, a ``Base.registry`` filled by
``__init_subclass__``, ``partial(go, cls=Double)`` and a closure over
``{"d": Double}`` all served the old result after ``Double.run`` was edited.
"""

from __future__ import annotations

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

DOUBLE = """
    class Double:
        def run(self, x):
            return x * 2
"""

REGISTRY = """
    class Base:
        registry = {}

        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            Base.registry[cls.__name__.lower()] = cls

    class Double(Base):
        def run(self, x):
            return x * 2
"""

SHAPES = {
    "a dict of classes read as module.REG": (
        DOUBLE + "\n    REG = {'double': Double}\n",
        """
        import cash, models

        @cash.cache
        def f(x):
            return models.REG['double']().run(x)
        """,
    ),
    "a list of classes read as module.PLUGINS": (
        DOUBLE + "\n    PLUGINS = [Double]\n",
        """
        import cash, models

        @cash.cache
        def f(x):
            return models.PLUGINS[0]().run(x)
        """,
    ),
    "an __init_subclass__ registry on a class": (
        REGISTRY,
        """
        import cash
        from models import Base

        @cash.cache
        def f(x):
            return Base.registry['double']().run(x)
        """,
    ),
    "a partial binding the class": (
        DOUBLE,
        """
        import cash, functools
        from models import Double

        def go(x, cls):
            return cls().run(x)

        runner = functools.partial(go, cls=Double)

        @cash.cache
        def f(x):
            return runner(x)
        """,
    ),
    "a helper's closure over a dict of classes": (
        DOUBLE,
        """
        import cash
        from models import Double

        def make():
            reg = {'d': Double}
            def go(x):
                return reg['d']().run(x)
            return go

        runner = make()

        @cash.cache
        def f(x):
            return runner(x)
        """,
    ),
    "the cached function's closure over a dict of classes": (
        DOUBLE,
        """
        import cash
        from models import Double

        def make():
            reg = {'d': Double}
            @cash.cache
            def f(x):
                return reg['d']().run(x)
            return f

        f = make()
        """,
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_editing_the_class_recomputes(tmp_path, shape):
    models, main = SHAPES[shape]
    first, after, uncached = edited_runs(
        tmp_path,
        {"models.py": models, "main.py": main + "\n        print(f(3))\n"},
        [("models.py", "x * 2", "x * 3")],
    )
    assert first == "6"
    assert after == uncached == "9"
