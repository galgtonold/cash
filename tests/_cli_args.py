"""The arguments a ``cash`` subcommand receives, for tests that call one directly."""

from __future__ import annotations

import argparse
from typing import Any

from cash.__main__ import build_parser


def cli_args(command: str, **overrides: Any) -> argparse.Namespace:
    """What ``cash <command>`` parses to, with *overrides* set on top: every
    option the command has, at its default unless overridden."""
    args = build_parser().parse_args([command])
    for name, value in overrides.items():
        if not hasattr(args, name):
            raise AttributeError(f"`cash {command}` has no option {name!r}")
        setattr(args, name, value)
    return args
