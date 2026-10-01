"""The import hook wraps only the modules a handler is registered for.

Handlers are looked up by the exact module name, so wrapping the loader of
any other module patches nothing. The hook used to match on the top-level
package as well, which put every submodule of a registered package (every
``pandas.*`` a pandas import pulls in) through its wrapping loader for no
effect.
"""

import sys

import pytest

from cash.tracking import reader_patches
from cash.tracking.reader_patches import FileDependencyRegistry, PostImportHook, _PatchingLoader


@pytest.fixture
def package(tmp_path, monkeypatch):
    pkg = tmp_path / "hookpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = FileDependencyRegistry()
    registry.register("hookpkg", "read_*", FileDependencyRegistry._create_path_arg_handler)
    monkeypatch.setattr(reader_patches, "_registry", registry)
    yield
    for name in ("hookpkg", "hookpkg.helpers"):
        sys.modules.pop(name, None)


def test_a_registered_module_is_wrapped(package):
    spec = PostImportHook().find_spec("hookpkg", None)
    assert spec is not None and isinstance(spec.loader, _PatchingLoader)


def test_a_submodule_of_a_registered_package_is_left_alone(package):
    import hookpkg

    assert PostImportHook().find_spec("hookpkg.helpers", hookpkg.__path__) is None
