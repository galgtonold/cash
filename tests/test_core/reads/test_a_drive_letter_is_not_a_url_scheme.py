"""A Windows drive letter followed by ``//`` is a local path, not a URL.

``f"{ROOT}/x.csv"`` with ``ROOT = "D:/"`` gives ``D://x.csv``, which Windows
opens as the local file ``D:\\x.csv``. Read as a URL with the one-letter
scheme ``d``, the read was put on the remote channel: without fsspec the
cached call raised ("tracking d:// objects requires fsspec"), and with it the
local file was fingerprinted as a remote object. The rule is pure string
logic, so it is pinned on every platform.
"""

from __future__ import annotations

import pytest

from cash._paths import is_remote_url
from cash.tracking.file_tracker import FileAccessTracker


@pytest.mark.parametrize("path", ["C://Temp//x.csv", "d://data/x.csv", "Z://"])
def test_a_drive_letter_is_not_a_scheme(path):
    assert not is_remote_url(path)


@pytest.mark.parametrize("url", ["s3://bucket/key", "gs://b/k", "az://c/b", "https://example.org/x.csv"])
def test_real_schemes_are_still_urls(url):
    assert is_remote_url(url)


def test_a_drive_letter_read_is_tracked_as_a_file():
    with FileAccessTracker() as tracker:
        tracker.track_path("C://Temp//x.csv")
    assert not tracker.get_accessed_remote_urls()
