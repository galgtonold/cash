"""A class a helper names only to test objects against is keyed by its code,
not by the data its methods read.

``def log(msg, classes=(Base,)): ... isinstance(obj, classes)`` and
``except Base:`` ask what an object is; no code of ``Base`` runs. Every
global its methods read was folded all the same, so a version string a
documentation property of the base class reads -- different on every commit
of a source checkout -- moved the key of every cached function that logs.
Once the class is constructed, or its test is answered by a metaclass, what
its methods read is an input again and stays keyed.
"""

from __future__ import annotations

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

VERSION = 'VERSION = "1.0.dev1"\n'

MODELS = """
    from version import VERSION

    class Base:
        @property
        def doc_link(self):
            return "https://docs/" + VERSION
"""

ABC_MODELS = """
    import abc
    from version import VERSION

    class Base(abc.ABC):
        @classmethod
        def __subclasshook__(cls, other):
            return VERSION.endswith("dev2") or NotImplemented
"""

CACHED = """
    import cash
    from helpers import describe

    @cash.cache
    def f(x):
        print("ran")
        return describe(x)

    print(f(3))
"""

TESTED_ONLY = {
    "a parameter defaulting to a tuple of it": """
        from models import Base

        def describe(obj, classes=(Base,)):
            return "base" if isinstance(obj, classes) else type(obj).__name__
    """,
    "isinstance and issubclass": """
        from models import Base

        def describe(obj):
            if isinstance(obj, (Base, bytes)) or issubclass(type(obj), Base):
                return "base"
            return type(obj).__name__
    """,
    "an except clause": """
        from models import Base

        def describe(obj):
            try:
                return type(obj).__name__
            except Base:
                return "base"
    """,
}

USED = {
    "constructed": (
        MODELS,
        """
        from models import Base

        def describe(obj):
            return Base().doc_link
        """,
    ),
    "a defaulted parameter it is constructed from": (
        MODELS,
        """
        from models import Base

        def describe(obj, classes=(Base,)):
            return classes[0]().doc_link
        """,
    ),
    "a metaclass answering the test": (
        ABC_MODELS,
        """
        from models import Base

        def describe(obj):
            return "base" if isinstance(obj, Base) else type(obj).__name__
        """,
    ),
}


def _files(models: str, helpers: str) -> dict[str, str]:
    return {"main.py": CACHED, "helpers.py": helpers, "models.py": models, "version.py": VERSION}


@pytest.mark.parametrize("shape", sorted(TESTED_ONLY))
def test_what_its_methods_read_does_not_move_the_key(tmp_path, shape):
    first, after, uncached = edited_runs(tmp_path, _files(MODELS, TESTED_ONLY[shape]), [("version.py", "dev1", "dev2")])
    assert first == "ran\nint"
    assert uncached == "ran\nint"
    assert after == "int", "a class only tested against moved the key through what its methods read"


@pytest.mark.parametrize("shape", sorted(USED))
def test_a_class_whose_code_can_run_keeps_what_it_reads(tmp_path, shape):
    models, helpers = USED[shape]
    first, after, uncached = edited_runs(tmp_path, _files(models, helpers), [("version.py", "dev1", "dev2")])
    assert after == uncached
    assert after.startswith("ran"), "an edit to what the class's code reads was served stale"
    assert first != after
