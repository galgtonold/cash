"""The explorer shows sizes as the CLI does: powers of 1024, labelled KiB/MiB/GiB.

It used to divide by 1024 and label the result "KB"/"MB"/"GB" -- in the page
and in the widget -- so a 2 GiB entry read as "2 GB" beside a ``cash info``
that said "2.0 GiB", and a "2GB" cap read as mis-parsed.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess

import pytest

pytest.importorskip("IPython")

from cash import Cash  # noqa: E402
from cash.config import human_bytes  # noqa: E402

SIZES = [0, 500, 1023, 1024, 2048, 1536, 5 * 1024**2, 2 * 1024**3, 3 * 1024**4]


def _page(tmp_path) -> str:
    app = Cash(cache_dir=str(tmp_path / "c"))
    frame = app.explorer()._widget_html()
    return base64.b64decode(frame.src.split(",", 1)[1]).decode("utf-8")


def _format_bytes_js(page: str) -> str:
    match = re.search(r"function formatBytes\(.*?\n                }\n", page, re.S)
    assert match, "the page defines formatBytes"
    return match.group(0)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the page's script")
def test_the_page_labels_sizes_as_human_bytes_does(tmp_path):
    script = _format_bytes_js(_page(tmp_path)) + f"console.log(JSON.stringify({json.dumps(SIZES)}.map(formatBytes)));"
    shown = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert shown == [human_bytes(n) for n in SIZES]


def test_the_page_never_labels_powers_of_1024_as_decimal_units(tmp_path):
    script = _format_bytes_js(_page(tmp_path))
    for decimal_unit in ("'KB'", "'MB'", "'GB'", "'TB'", "'Bytes'", "' Bytes'"):
        assert decimal_unit not in script
