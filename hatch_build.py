"""Build hook: compiles the optional C extension ``cash._cwatch``.

hatchling builds the package as before; this hook only adds the compiled
module to the wheel. The extension is optional everywhere:

* It is built only on CPython 3.12 or newer (the dict, function and type
  watchers it uses arrived in 3.12). Anywhere else the wheel is pure Python.
* ``CASH_PURE_PYTHON=1`` skips it on purpose: that is how the universal
  ``py3-none-any`` wheel is built.
* When compiling fails (no compiler, no Python headers), the build prints a
  warning and goes on without it, so installing from the sdist never fails
  over the extension. cash then runs its pure-Python path.
  ``CASH_REQUIRE_EXTENSION=1`` turns that failure into a build error; the
  release workflow sets it, so a platform wheel always has the extension.

The compiler is driven by setuptools' ``build_ext``, which finds it on every
OS (gcc/clang, MSVC) and honours ``CC``, ``CFLAGS``, ``ARCHFLAGS`` and the
rest. setuptools is asked for only when the extension is to be built.

For a checkout, ``python hatch_build.py`` compiles the module in place
(``src/cash/_cwatch.<abi>.so``, gitignored), as ``pip install -e .`` does.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import sysconfig
import tempfile
import traceback
from pathlib import Path

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # run as a script, outside a build
    BuildHookInterface = object

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "src" / "cash"
C_DIR = PACKAGE_DIR / "_c"
MODULE = "cash._cwatch"

#: The build backend gets setuptools only when it compiles.
COMPILER_REQUIREMENT = "setuptools>=64"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def skip_reason() -> str | None:
    """Why this build leaves the extension out, or None to build it."""
    if _flag("CASH_PURE_PYTHON"):
        return "CASH_PURE_PYTHON is set"
    if platform.python_implementation() != "CPython":
        return f"{platform.python_implementation()} is not CPython"
    if sys.version_info < (3, 12):
        return "it needs Python 3.12 or newer"
    if not sorted(C_DIR.glob("*.c")):
        return f"no C sources in {C_DIR}"
    return None


def compile_extension(build_lib: Path, build_temp: Path) -> Path:
    """Compile ``cash._cwatch`` under *build_lib*; returns the module file."""
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    sources = sorted(C_DIR.glob("*.c"))
    depends = sorted(C_DIR.glob("*.h"))
    extension = Extension(
        MODULE,
        sources=[os.path.relpath(s, ROOT) for s in sources],
        depends=[os.path.relpath(d, ROOT) for d in depends],
        include_dirs=[os.path.relpath(C_DIR, ROOT)],
    )

    class _BuildExt(build_ext):
        def build_extensions(self):
            if self.compiler.compiler_type != "msvc":  # gcc and clang; warnings shown, never fatal
                for ext in self.extensions:
                    ext.extra_compile_args = ["-Wall", *ext.extra_compile_args]
            super().build_extensions()

    dist = Distribution({"name": "cash-lib", "ext_modules": [extension]})
    dist.script_name = "hatch_build.py"
    cmd = _BuildExt(dist)
    cmd.build_lib = str(build_lib)
    cmd.build_temp = str(build_temp)
    cmd.inplace = False
    cmd.ensure_finalized()
    cwd = os.getcwd()
    os.chdir(ROOT)  # the source paths are relative, which keeps them out of the object files
    try:
        cmd.run()
    finally:
        os.chdir(cwd)
    built = Path(cmd.get_ext_fullpath(MODULE))
    if not built.is_absolute():
        built = ROOT / built
    if not built.is_file():
        raise RuntimeError(f"build_ext reported success but {built} is missing")
    return built


def module_filename() -> str:
    return "_cwatch" + sysconfig.get_config_var("EXT_SUFFIX")


def _warn(message: str) -> None:
    print(f"cash-lib build: {message}", file=sys.stderr, flush=True)


class CustomBuildHook(BuildHookInterface):
    """hatchling's hook for the wheel target (``standard`` and ``editable``)."""

    PLUGIN_NAME = "custom"

    def dependencies(self) -> list[str]:
        return [] if skip_reason() else [COMPILER_REQUIREMENT]

    def initialize(self, version: str, build_data: dict) -> None:
        self._tmp = None
        reason = skip_reason()
        if reason:
            _warn(f"building without the C extension ({reason}); cash will use its pure-Python path")
            return
        self._tmp = Path(tempfile.mkdtemp(prefix="cash-ext-"))
        try:
            built = compile_extension(self._tmp / "lib", self._tmp / "temp")
        except Exception as exc:  # any compile failure means "no extension", not a failed install
            if _flag("CASH_REQUIRE_EXTENSION"):
                raise
            traceback.print_exc()
            _warn(
                f"could not compile the C extension ({type(exc).__name__}: {exc}); "
                "installing without it, cash will use its pure-Python path"
            )
            return
        if version == "editable":
            # An editable install imports cash from src/, so the module goes there.
            shutil.copy2(built, PACKAGE_DIR / module_filename())
        else:
            build_data["force_include"][str(built)] = f"cash/{module_filename()}"
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)


if __name__ == "__main__":
    reason = skip_reason()
    if reason:
        sys.exit(f"not building the extension: {reason}")
    with tempfile.TemporaryDirectory(prefix="cash-ext-") as tmp:
        built = compile_extension(Path(tmp) / "lib", Path(tmp) / "temp")
        target = PACKAGE_DIR / module_filename()
        shutil.copy2(built, target)
    print(f"built {target}")
