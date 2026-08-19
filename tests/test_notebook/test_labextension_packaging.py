"""The JupyterLab extension half of the live-cell read, guarded from Python.

`cash.notebook.live_cells` receives what ``labextension/src/index.ts`` pushes.
Two things about that arrangement can break silently, and neither has any Python
symbol to hang a normal test off:

**1. ``comm.commsOverSubshells = 'disabled'`` disappearing.** JupyterLab 4.6
defaults ``commsOverSubshells: perCommTarget``, which delivers comms on a
SUBSHELL THREAD. Without that one assignment the design fails twice over: the
push-before-``execute_request`` ordering stops being FIFO and becomes a race
(measured a 0.4-7.4ms lead, won 130/130 -- the kind of evidence that reads as a
guarantee until a slower machine disagrees; forcing the main shell gave 56/56),
and the deliberately lock-free store in ``live_cells.py`` becomes cross-thread
mutable state. ``live_cells.py`` can only say so in a comment. This file is the
enforcement, and it checks the SHIPPED bundle as well as the source, because
the bundle is committed and is what a user actually runs.

**2. Packaging drift.** A prebuilt labextension is discovered by *name* under
``share/jupyter/labextensions/``. The npm package name, the ``shared-data``
destination in ``pyproject.toml``, and ``_jupyter_labextension_paths()`` must
all agree, or the wheel builds happily and registers nothing -- which is the
failure mode this whole packaging step exists to avoid.

None of this needs Node: the built output is committed precisely so that
building the wheel, and running this suite, never require a JavaScript
toolchain.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEXT_SRC = REPO_ROOT / "labextension"
INDEX_TS = LABEXT_SRC / "src" / "index.ts"
BUILT = REPO_ROOT / "src" / "cash" / "labextension"

# The extension's identity, asserted from four independent places below.
EXT_NAME = "cash-live-cells"

# Loose about whitespace, quote style and receiver name, so a rename or a
# reformat does not read as a removal. Minifiers preserve property names and
# string literals, so the same pattern matches the built bundle.
FORCES_MAIN_SHELL = re.compile(r"""commsOverSubshells\s*[=:]\s*["']disabled["']""")

_WHY = (
    "\n\n`comm.commsOverSubshells = 'disabled'` is LOAD-BEARING, not a tuning knob.\n"
    "Without it JupyterLab 4.6 delivers the comm on a subshell thread, which\n"
    "(a) turns the push-before-execute_request ordering from a FIFO guarantee\n"
    "into a race, and (b) makes the lock-free store in\n"
    "src/cash/notebook/live_cells.py cross-thread mutable state.\n"
    "If you are changing this on purpose, change labextension/src/index.ts,\n"
    "the thread-safety note in live_cells.py, this test, and\n"
    "labextension/scripts/check-comms-over-subshells.js together."
)


def _skip_without_checkout():
    if not LABEXT_SRC.exists():
        pytest.skip("not running from the repo checkout")


def _built_js() -> list[Path]:
    return sorted(BUILT.rglob("*.js"))


def _pyproject() -> dict:
    """Parsed ``pyproject.toml``.

    Deliberately FAILS rather than skips when no TOML parser is available.
    Python 3.10 -- the floor -- has no ``tomllib``, but ``tomli`` is in this
    project's ``[dev]`` install closure there, and ``[dev]`` is required to run
    this suite at all. A packaging test that quietly disappears on one
    interpreter of a five-version CI matrix is worse than one that says what to
    install.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            pytest.fail(
                "no TOML parser: this needs tomllib (Python 3.11+) or tomli. "
                "`pip install -e '.[dev]'` provides one on every supported version."
            )
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


# --- The one line ------------------------------------------------------------


def test_the_source_forces_the_comm_onto_the_main_shell():
    _skip_without_checkout()
    assert FORCES_MAIN_SHELL.search(INDEX_TS.read_text(encoding="utf-8")), (
        f"{INDEX_TS} no longer forces the comm onto the main shell." + _WHY
    )


def test_the_shipped_bundle_forces_the_comm_onto_the_main_shell():
    """The source is not what runs -- the committed bundle is.

    A correct source plus a stale or wrongly-built bundle ships the bug, and
    that combination is exactly what a committed build artifact makes possible.
    """
    _skip_without_checkout()
    files = _built_js()
    assert files, (
        f"no built JavaScript under {BUILT}; run `cd labextension && npm run build` "
        f"and commit the result (the wheel ships this directory)"
    )
    hit = [f for f in files if FORCES_MAIN_SHELL.search(f.read_text(encoding="utf-8", errors="replace"))]
    assert hit, (
        f"the SHIPPED bundle under {BUILT} does not force the comm onto the main "
        f"shell ({len(files)} .js files scanned)." + _WHY
    )


# --- Wiring: the frontend and the kernel must be talking about the same comm --


def test_the_extension_targets_the_comm_the_kernel_registers():
    """A typo on either side is a silent no-op, not an error."""
    _skip_without_checkout()
    from cash.notebook.live_cells import TARGET

    m = re.search(r"""const\s+TARGET\s*=\s*['"]([^'"]+)['"]""", INDEX_TS.read_text(encoding="utf-8"))
    assert m, f"could not find the comm target constant in {INDEX_TS}"
    assert m.group(1) == TARGET, (
        f"the extension opens comm target {m.group(1)!r} but the kernel registers "
        f"{TARGET!r}; the push would be dropped by the comm manager"
    )


# --- Packaging: builds-but-does-not-register is the failure mode --------------


def test_the_built_output_is_complete_and_loadable():
    _skip_without_checkout()
    pkg = BUILT / "package.json"
    assert pkg.exists(), f"{pkg} missing; run `cd labextension && npm run build`"
    data = json.loads(pkg.read_text(encoding="utf-8"))
    load = data.get("jupyterlab", {}).get("_build", {}).get("load")
    assert load, "built package.json has no jupyterlab._build.load; JupyterLab cannot load it"
    assert "\\" not in load, (
        f"jupyterlab._build.load is {load!r} -- a Windows path separator. JupyterLab "
        f"serves this string to the browser as a URL, where a backslash is not a "
        f"separator, so a bundle built on Windows would fail to load anywhere. "
        f"`npm run build` normalises it (labextension/scripts/normalize-load-path.js); "
        f"a rebuild that skipped that step is the likely cause."
    )
    assert (BUILT / load).exists(), f"jupyterlab._build.load points at {load}, which does not exist"


def test_the_built_output_matches_its_source_package():
    """Catches a bundle rebuilt from a different package.json than the one in git."""
    _skip_without_checkout()
    src = json.loads((LABEXT_SRC / "package.json").read_text(encoding="utf-8"))
    built = json.loads((BUILT / "package.json").read_text(encoding="utf-8"))
    assert src["name"] == built["name"] == EXT_NAME
    assert src["version"] == built["version"], (
        "labextension/package.json and the committed build disagree on the version; "
        "the bundle is stale -- rebuild and commit it"
    )


def test_every_place_that_names_the_extension_agrees():
    """A prebuilt labextension is resolved by NAME.

    The npm package name, the directory the wheel drops it in, and the develop-
    install hook must be the same string, or JupyterLab finds a directory whose
    federated module it cannot resolve.
    """
    _skip_without_checkout()
    import cash

    npm_name = json.loads((LABEXT_SRC / "package.json").read_text(encoding="utf-8"))["name"]
    shared = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["shared-data"]

    assert npm_name == EXT_NAME
    assert shared["src/cash/labextension"] == f"share/jupyter/labextensions/{EXT_NAME}"
    assert shared["labextension/install.json"] == f"share/jupyter/labextensions/{EXT_NAME}/install.json"

    paths = cash._jupyter_labextension_paths()
    assert paths == [{"src": "labextension", "dest": EXT_NAME}]
    assert (REPO_ROOT / "src" / "cash" / paths[0]["src"]).is_dir(), (
        "_jupyter_labextension_paths() points at a directory that does not exist, so "
        "`jupyter labextension develop` would install nothing"
    )


def test_install_json_hands_uninstallation_back_to_pip():
    _skip_without_checkout()
    data = json.loads((LABEXT_SRC / "install.json").read_text(encoding="utf-8"))
    assert data["packageManager"] == "python"
    assert data["packageName"] == _pyproject()["project"]["name"], (
        "install.json names a different distribution than pyproject.toml, so "
        "`jupyter labextension uninstall` would tell the user to uninstall "
        "something that does not exist"
    )


def test_the_sdist_can_rebuild_the_extension():
    """The sdist is a whitelist; a missing entry is invisible until someone
    tries to build a wheel from it."""
    _skip_without_checkout()
    include = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    for needed in (
        "/src/cash",                    # carries the committed build output
        "/labextension/install.json",   # named by path in the shared-data mapping
        "/labextension/package.json",
        "/labextension/src",
    ):
        assert needed in include, f"sdist include list is missing {needed}"
