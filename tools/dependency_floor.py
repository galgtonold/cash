"""Which of cash's dependencies no longer release for this Python?

    python tools/dependency_floor.py [distribution]

Run in an environment where cash was installed from wheels, on the oldest
Python cash supports (the nightly workflow does). pip installs the newest
version of a dependency that still supports the running Python, without a
word when that is no longer the newest there is: ``blake3`` stopped building
wheels for Python 3.10 while cash still supported 3.10, and cash's oldest
Python then hung on an old release of a dependency with nobody told.

For each dependency of the distribution (default ``cash-lib``) that applies
to this interpreter, the installed version is compared with the newest on
PyPI. One that is behind is printed as a GitHub Actions warning and listed
in the job summary. The exit code stays 0: this reports, the decision
(drop the Python, or stay on the old release) is a person's.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from importlib import metadata

__all__ = ["applying_dependencies", "behind", "main"]


def applying_dependencies(requirements: list[str], environment: dict[str, str] | None = None) -> list[str]:
    """The names in *requirements* (``Requires-Dist`` lines) that apply to
    this interpreter and belong to no extra."""
    from packaging.requirements import Requirement

    names = []
    for line in requirements:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker is not None and "extra" in str(marker):
            continue
        if marker is None or marker.evaluate(environment):
            names.append(requirement.name)
    return names


def behind(installed: dict[str, str], newest: dict[str, str]) -> list[tuple[str, str, str]]:
    """``(name, installed, newest)`` for each dependency whose installed
    version is not the newest release."""
    from packaging.version import Version

    return [
        (name, version, newest[name])
        for name, version in sorted(installed.items())
        if name in newest and Version(version) < Version(newest[name])
    ]


def _newest_on_pypi(name: str) -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as response:
        return json.load(response)["info"]["version"]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    distribution = args[0] if args else "cash-lib"
    names = applying_dependencies(metadata.requires(distribution) or [])
    installed = {name: metadata.version(name) for name in names}
    newest = {name: _newest_on_pypi(name) for name in names}
    python = ".".join(map(str, sys.version_info[:2]))
    late = behind(installed, newest)
    lines = [f"{name} {installed[name]} (newest {newest[name]})" for name in sorted(installed)]
    print(f"Dependencies of {distribution} on Python {python}: " + ", ".join(lines))
    for name, have, latest in late:
        print(
            f"::warning title=A dependency left Python {python} behind::{name} {latest} does not install on "
            f"Python {python}; pip installed {name} {have}. Drop Python {python}, or accept the older {name}."
        )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and late:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Dependencies whose newest release does not install on Python {python}\n\n")
            for name, have, latest in late:
                fh.write(f"- `{re.sub(r'[^A-Za-z0-9_.-]', '', name)}`: installed {have}, newest {latest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
