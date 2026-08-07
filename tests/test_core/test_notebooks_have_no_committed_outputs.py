"""A tracked notebook must not carry committed cell outputs.

The release checklist has asked for this for a while and nothing enforced it,
so it was missed: `examples/cash_testing.ipynb` shipped 3 cells of outputs and
sat at 45,244 bytes, against 2,126 once stripped. Twenty times its own size,
all of it one run's residue.

Why it matters beyond tidiness:

* **The sdist reads the WORKING tree.** A notebook someone ran locally leaks
  that run's outputs — and whatever was in the DataFrame it rendered — into a
  published artifact.
* **Outputs are the demo's first impression.** A stale output showing an old
  API, an old badge, or a timing from a different machine reads as current.
* Notebook diffs become unreviewable, so a real content change hides inside a
  wall of re-serialised base64.

Checked against `git ls-files` rather than a directory walk, deliberately:
`.ipynb_checkpoints/` and `.claude/worktrees/` are full of outputs and are
correctly git-ignored. Walking the tree would fail on files nobody committed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _tracked_notebooks() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return [line for line in out.stdout.split() if line]


def test_the_repo_has_tracked_notebooks_to_check():
    """Non-vacuity: if `git ls-files` ever returns nothing, the test below
    passes for the wrong reason and this file stops protecting anything."""
    assert _tracked_notebooks(), "no tracked notebooks found — is git available?"


@pytest.mark.parametrize("name", _tracked_notebooks())
def test_a_tracked_notebook_carries_no_outputs(name):
    nb = json.loads((REPO / name).read_text(encoding="utf-8"))
    dirty = [
        i for i, cell in enumerate(nb.get("cells", []))
        if cell.get("cell_type") == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert not dirty, (
        f"{name} has committed outputs in cell(s) {dirty}. Clear them before "
        f"committing — the sdist is built from the working tree, so a local "
        f"run's outputs would ship. In Jupyter: Kernel > Restart & Clear Output."
    )
