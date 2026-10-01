"""``Cash(backend="sqlite")`` builds that backend.

``Cash(...)`` takes any setting by name, and ``backend`` is a setting whose
values are names ("tiered", "file", "sqlite", ...). But the constructor's own
``backend`` parameter, a backend instance, took the string: construction
succeeded, and every cached call then failed with ``AttributeError: 'str'
object has no attribute 'get'``, plus a traceback at exit.
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.backends import SQLiteBackend


def test_a_backend_name_is_the_setting(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), backend="sqlite", register_magic=False)
    try:

        @c.cache
        def f(n):
            return n * 2

        assert f(1) == 2
        assert f(1) == 2
        assert f.cache_info()["hits"] == 1
        assert isinstance(c.backend, SQLiteBackend)
        assert c.config.backend == "sqlite"
    finally:
        c.shutdown()


def test_anything_else_is_refused_at_construction():
    with pytest.raises(TypeError, match="backend instance or a backend type name"):
        Cash(backend=42, register_magic=False)
