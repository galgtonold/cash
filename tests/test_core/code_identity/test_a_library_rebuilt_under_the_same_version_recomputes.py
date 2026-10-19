"""A library rebuilt under the same version number recomputes.

Only the version number of an installed library went into the key, so a team
package rebuilt and reinstalled without a version bump (``pip install .``, a
``git+`` install of a branch that moved) was served the old code's results,
with no warning. A digest of the files its ``RECORD`` lists, by the hash pip
wrote for each, now goes with the version: new code recomputes, and the same
files reinstalled (a new ``RECORD`` with the same hashes, another
``INSTALLER``) hit as before.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import textwrap

import pytest

from cash.decorator.library_versions import _record_digest
from tests.test_core.code_identity.test_an_upgraded_library_recomputes import _make_proj, _run

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


@pytest.fixture
def proj(tmp_path):
    return _make_proj(tmp_path)


def _hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _install(folder, factor, installer="pip", hashed=True):
    """fakelib 1.0 as pip lays it out, its code multiplying by *factor*."""
    shutil.rmtree(folder, ignore_errors=True)
    package = folder / "fakelib"
    package.mkdir(parents=True)
    code = textwrap.dedent(
        f"""
        class Model:
            def fit(self, x):
                return x * {factor}

        def scale(x):
            return x * {factor}
        """
    ).encode()
    (package / "__init__.py").write_bytes(code)
    info = folder / "fakelib-1.0.dist-info"
    info.mkdir()
    metadata = b"Metadata-Version: 2.1\nName: fakelib\nVersion: 1.0\n"
    (info / "METADATA").write_bytes(metadata)
    installed_by = installer.encode() + b"\n"
    (info / "INSTALLER").write_bytes(installed_by)
    (info / "top_level.txt").write_text("fakelib\n", encoding="utf-8")
    rows = [
        f"fakelib/__init__.py,{_hash(code) if hashed else ''},{len(code) if hashed else ''}",
        "fakelib/__pycache__/__init__.cpython-314.pyc,,",
        f"fakelib-1.0.dist-info/METADATA,{_hash(metadata)},{len(metadata)}",
        f"fakelib-1.0.dist-info/INSTALLER,{_hash(installed_by)},{len(installed_by)}",
        "fakelib-1.0.dist-info/RECORD,,",
    ]
    (info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_code_rebuilt_under_the_same_version_recomputes(proj, tmp_path):
    lib = tmp_path / "site"
    _install(lib, 2)
    assert _run(proj, lib, "by_module") == ("6", True)
    assert _run(proj, lib, "by_module") == ("6", False), "an unchanged library recomputed"
    _install(lib, 5)
    assert _run(proj, lib, "by_module") == ("15", True), "the rebuilt library served the old code's result"
    assert _run(proj, lib, "by_module") == ("15", False)


def test_the_same_files_reinstalled_still_hit(proj, tmp_path):
    lib = tmp_path / "site"
    _install(lib, 2)
    assert _run(proj, lib, "by_name") == ("6", True)
    _install(lib, 2, installer="uv")
    assert _run(proj, lib, "by_name") == ("6", False), "the same code reinstalled recomputed"


def test_a_record_without_hashes_keys_by_the_version_alone(proj, tmp_path):
    """An installer that writes no hashes leaves nothing to compare: the
    version alone, as before."""
    lib = tmp_path / "site"
    _install(lib, 2, hashed=False)
    assert _run(proj, lib, "by_name") == ("6", True)
    _install(lib, 5, hashed=False)
    assert _run(proj, lib, "by_name") == ("6", False)


def test_the_digest_reads_only_the_hashes_of_the_code():
    text = "pkg/a.py,sha256=AAA,3\npkg/__pycache__/a.pyc,,\npkg-1.dist-info/INSTALLER,sha256=X,4\npkg-1.dist-info/RECORD,,\n"
    other_installer = text.replace("sha256=X", "sha256=Y")
    other_code = text.replace("sha256=AAA", "sha256=BBB")
    assert _record_digest(text) == _record_digest(other_installer)
    assert _record_digest(text) != _record_digest(other_code)
    assert _record_digest("pkg-1.dist-info/RECORD,,\n") is None
