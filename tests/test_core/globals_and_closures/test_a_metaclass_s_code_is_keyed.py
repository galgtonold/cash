"""A class's metaclass is part of its code.

``Model()`` runs the metaclass's ``__call__``; ``Model.factor`` can be a
property on the metaclass. Neither is in the class's own source, so editing
them served the old result, whether the class was named in the body or passed
as an argument.
"""

from __future__ import annotations

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

CALL = """
    class Configured(type):
        def __call__(cls, *args):
            obj = super().__call__(*args)
            obj.scale = 1
            return obj

    class Model(metaclass=Configured):
        def predict(self, x):
            return x * self.scale
"""

PROPERTY = """
    class Meta(type):
        @property
        def factor(cls):
            return 1

    class Model(metaclass=Meta):
        pass
"""

SHAPES = {
    "__call__, the class named in the body": (
        CALL,
        "@cash.cache\ndef f(x):\n    return Model().predict(x)\n\nprint(f(3))\n",
        ("obj.scale = 1", "obj.scale = 100"),
    ),
    "__call__, the class passed as an argument": (
        CALL,
        "@cash.cache\ndef f(cls, x):\n    return cls().predict(x)\n\nprint(f(Model, 3))\n",
        ("obj.scale = 1", "obj.scale = 100"),
    ),
    "a property on the metaclass": (
        PROPERTY,
        "@cash.cache\ndef f(x):\n    return x * Model.factor\n\nprint(f(3))\n",
        ("return 1", "return 100"),
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_editing_the_metaclass_recomputes(tmp_path, shape):
    models, body, (old, new) = SHAPES[shape]
    main = "import cash\nfrom models import Model\n\n" + body
    first, after, uncached = edited_runs(tmp_path, {"main.py": main, "models.py": models}, [("models.py", old, new)])
    assert first == "3"
    assert after == uncached == "300"


def test_a_catch_all_getattr_on_the_metaclass_is_keyed(tmp_path):
    settings = """
        class DefaultsMeta(type):
            def __getattr__(cls, name):
                return 1

        class Defaults(metaclass=DefaultsMeta):
            pass
    """
    main = "import cash\nfrom settings import Defaults\n\n@cash.cache\ndef f(x):\n    return x + Defaults.retries\n\nprint(f(3))\n"
    first, after, uncached = edited_runs(
        tmp_path, {"main.py": main, "settings.py": settings}, [("settings.py", "return 1", "return 100")]
    )
    assert first == "4"
    assert after == uncached == "103"
