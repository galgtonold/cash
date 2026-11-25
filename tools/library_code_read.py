"""Is the library code cash's rules were read in still what is installed?

A few notebook rules rest on how scikit-learn and numpy pickle their
objects (``cash.notebook.code_read.RULES``). Each holds for the texts of
the functions that were read, by digest. A release that edits one of them
turns the rule off for its users, safely and silently: the statement that
needed the rule runs every time. This script is how a maintainer hears of
it. The nightly workflow runs it on the newest releases.

    python tools/library_code_read.py

Prints one line per rule. Exits 1 when a rule is off, with the digest to
add to ``RULES`` once the functions it names have been read again and
still do what the rule's text in ``cash/notebook/anchors.py`` says.
"""

from __future__ import annotations

import importlib
import sys

#: What to import so each rule's library is loaded: a rule looks only at
#: modules the process has already imported.
IMPORTS = {"estimator_pickling": "sklearn.base", "masked_array_pickling": "numpy.ma"}


def main() -> int:
    from cash.notebook import code_read

    for name, module in IMPORTS.items():
        try:
            importlib.import_module(module)
        except ImportError as error:
            print(f"::error::{name}: could not import {module}: {error}")
            return 1
    missing = sorted(set(code_read.RULES) - set(IMPORTS))
    if missing:
        print(f"::error::no import listed here for the rules {missing}")
        return 1
    for name, line in code_read.status().items():
        print(f"{name}: {line}")
    off = [name for name in code_read.RULES if not code_read.is_read(name)]
    for name in off:
        rule = code_read.RULES[name]
        functions = rule.functions() or []
        names = ", ".join(getattr(f, "__qualname__", repr(f)) for f in functions)
        print(
            f"::error::{name} is off: read {names} in the installed {rule.library} again, and if the "
            f"rule still holds add the digest {code_read.source_digest(functions)} to "
            "cash.notebook.code_read.RULES"
        )
    return 1 if off else 0


if __name__ == "__main__":
    sys.exit(main())
