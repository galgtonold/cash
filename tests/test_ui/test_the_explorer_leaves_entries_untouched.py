"""Browsing the cache with the explorer changes nothing in it, and what it
shows cannot break its own page.

The RAM tier hands out the metadata dicts it stores; the explorer's display
fields (the function's whole source, a formatted time) added to those would
come back with every later hit and be written to disk with the entry. And
names and source go into an HTML page: a ``<locals>`` qualname or a
``</script>`` inside cached source must show as text.
"""

from __future__ import annotations

import base64

import pytest

from cash import Cash
from cash.backends.memory_backend import InMemoryBackend


def test_listing_entries_adds_nothing_to_the_stored_metadata():
    c = Cash(backend=InMemoryBackend())

    @c.cache
    def f(x):
        return x + 1

    f(1)
    (key,) = [e["key"] for e in c.backend.list_entries()]
    listed = c.explorer().list_entries()
    assert "source_code" in listed[0]

    stored, _ = c.backend.get(key)
    assert "source_code" not in stored
    assert "timestamp_human" not in stored


def test_the_page_embeds_source_and_names_as_text():
    pytest.importorskip("IPython")
    c = Cash(backend=InMemoryBackend())

    def outer():
        @c.cache
        def inner(x):
            return "</script><b>" + str(x)

        return inner

    outer()(1)
    frame = c.explorer()._widget_html()
    page = base64.b64decode(frame.src.split(",", 1)[1]).decode("utf-8")
    script = page[page.index("<script>") :]
    # Only the page's own closing tag; the one in the source is escaped.
    assert script.count("</script>") == 1
    assert "escapeHtml(func.name)" in page
