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

# The C1 fix (CAS-274 review): a comm the kernel refused must not be latched on
# to forever. Both patterns are written to survive minification -- property
# names and the `catch` keyword both do -- so one regex serves the TypeScript
# source and the shipped bundle.
REBUILDS_A_CLOSED_COMM = re.compile(r"\.onClose\s*=")
CATCH_DROPS_THE_COMM = re.compile(r"catch\s*\([^)]*\)\s*\{[^}]*\.delete\(")

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


_WHY_REOPEN = """

A fresh kernel REFUSES the first comm_open: the flush-before-execute ordering
guarantees it arrives before the cell that runs `import cash` registers the
target. ipykernel replies comm_close and JupyterLab disposes the handler, so an
extension that keeps holding it is MUTE for the life of that kernel -- silently,
because every later push just throws into the catch.

Recovery is to drop the comm on close, and on a throw, so the next execution
flush opens a fresh one -- by which time the import has run."""


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


# --- A refused comm must not be latched on to (review C1) --------------------
#
# The fix is frontend-only, so these source+bundle assertions are its regression
# test: there is no JS harness here, and the kernel side cannot observe an
# extension that has gone quiet. Same two-layer pattern as the line above, and
# for the same reason -- the bundle is committed, so a correct source proves
# nothing about what ships.


def test_the_source_rebuilds_a_comm_the_kernel_closed():
    _skip_without_checkout()
    text = INDEX_TS.read_text(encoding="utf-8")
    assert REBUILDS_A_CLOSED_COMM.search(text), (
        f"{INDEX_TS} installs no onClose handler." + _WHY_REOPEN
    )
    assert CATCH_DROPS_THE_COMM.search(text), (
        f"{INDEX_TS}'s catch swallows the error without dropping the comm, so a "
        f"disposed handler ('Cannot send') is latched on to forever." + _WHY_REOPEN
    )


def test_the_shipped_bundle_rebuilds_a_comm_the_kernel_closed():
    _skip_without_checkout()
    files = _built_js()
    assert files, f"no built JavaScript under {BUILT}"
    blobs = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    assert any(REBUILDS_A_CLOSED_COMM.search(b) for b in blobs), (
        f"the SHIPPED bundle under {BUILT} installs no onClose handler." + _WHY_REOPEN
    )
    assert any(CATCH_DROPS_THE_COMM.search(b) for b in blobs), (
        f"the SHIPPED bundle under {BUILT} does not drop the comm on a throw." + _WHY_REOPEN
    )


def test_only_the_execution_flush_may_open_a_comm():
    """Source-only, and deliberately so.

    Gating the open is a noise control rather than a correctness fix -- opening
    from the debounce path would still recover, just after spraying "No such
    comm target registered" through the kernel log of every user who never
    imports cash. Worth pinning, not worth a bundle-side regex that would be
    guessing at minified argument positions.
    """
    _skip_without_checkout()
    text = INDEX_TS.read_text(encoding="utf-8")
    assert re.search(r"if\s*\(\s*!allowOpen\s*\)", text), (
        "the open is no longer gated on allowOpen; every debounced keystroke "
        "would now try to open a comm"
    )
    assert re.search(r"executionScheduled[\s\S]{0,400}?send\(\s*panel\s*,\s*true\s*\)", text), (
        "the execution flush must pass allowOpen=true -- it is the only route "
        "that may open a comm, and the only one that runs after `import cash` "
        "has had a chance to register the target"
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


_WHY_PLUGIN_ID = """

JupyterLab resolves a PACKAGE-level disable -- `jupyter labextension disable
cash-live-cells`, the `--disable-extension` server flag, the 4.1+ per-plugin UI
-- against the part of the plugin id BEFORE the colon. It does NOT look at the
installed directory name.

While the id read `cash:live-cells` and the package was `cash-live-cells`, the
documented disable command wrote {"cash-live-cells": true} into page_config.json,
matched nothing, and the extension went on pushing -- while `jupyter labextension
list` reported it `disabled ok`. The kill switch lied in both directions
(measured end to end, CAS-274 Task 4 Finding A). A feature whose safety story is
"degrade, never break" has to be switchable off.

Keep the id `<npm package name>:<plugin name>`."""

PLUGIN_ID = re.compile(r"""id\s*:\s*["']([^"']+)["']""")


def test_the_plugin_id_is_namespaced_by_the_package_name():
    """The plugin id's prefix is what `labextension disable` matches on.

    Checked in the SHIPPED bundle as well as the source: the bundle is
    committed, so a stale one would keep an old id in front of users no matter
    what the TypeScript says.
    """
    _skip_without_checkout()

    source_ids = PLUGIN_ID.findall(INDEX_TS.read_text(encoding="utf-8"))
    assert f"{EXT_NAME}:plugin" in source_ids, (
        f"labextension/src/index.ts declares no plugin id starting with "
        f"{EXT_NAME!r}; found {source_ids!r}.{_WHY_PLUGIN_ID}"
    )

    shipped = [text for path in _built_js()
               for text in [path.read_text(encoding="utf-8", errors="replace")]]
    assert any(f"{EXT_NAME}:plugin" in text for text in shipped), (
        f"no bundle under {BUILT} carries the plugin id {EXT_NAME}:plugin -- "
        f"the committed build is stale, rebuild it with "
        f"`cd labextension && npm run build`.{_WHY_PLUGIN_ID}"
    )
    orphan = f"{EXT_NAME.split('-')[0]}:"      # the old, unmatchable `cash:` prefix
    assert not any(f"'{orphan}live-cells'" in text or f'"{orphan}live-cells"' in text
                   for text in shipped), (
        f"a bundle still registers the old {orphan}live-cells id."
        f"{_WHY_PLUGIN_ID}"
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
