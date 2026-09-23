"""The explorer's preview shows the stored value.

The backend hands back the value itself, and the preview used to unpickle it
a second time, so every preview failed with "a bytes-like object is
required".
"""

from __future__ import annotations

import pytest

from cash import Cash


@pytest.fixture
def app(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"))


def _key_of(explorer, func):
    [entry] = [e for e in explorer.list_entries() if e.get("func_name") == Cash._get_func_key(func)]
    return entry["key"]


def test_the_preview_shows_the_cached_value(app):
    @app.cache
    def settings():
        return {"alpha": 1, "beta": [2, 3]}

    settings()
    explorer = app.explorer()
    preview = explorer.get_preview(_key_of(explorer, settings))
    assert preview == str({"alpha": 1, "beta": [2, 3]})


def test_a_cached_none_is_previewed_not_reported_missing(app):
    @app.cache
    def nothing():
        return None

    nothing()
    explorer = app.explorer()
    assert explorer.get_preview(_key_of(explorer, nothing)) == "None"
    assert explorer.get_preview("no-such-key") == "Value not found in cache."


def test_a_frame_is_previewed_by_its_head(app):
    pd = pytest.importorskip("pandas")

    @app.cache
    def frame():
        return pd.DataFrame({"n": range(100)})

    frame()
    explorer = app.explorer()
    preview = explorer.get_preview(_key_of(explorer, frame))
    assert preview == str(pd.DataFrame({"n": range(5)}))


def test_selecting_an_entry_in_the_widget_previews_it(app):
    pytest.importorskip("ipywidgets")

    @app.cache
    def answer():
        return 42

    answer()
    explorer = app.explorer()
    layout = explorer.widget()
    tabs = layout.children[1].children[0]
    assert [tabs.get_title(i) for i in range(len(tabs.children))] == ["Overview", "Entries"]

    shown = []
    explorer.get_preview = lambda key: shown.append(key) or "42"
    entries_selector = tabs.children[1].children[0]
    # The only function is selected from the start, with its entries listed.
    [(_label, key)] = entries_selector.options
    entries_selector.value = key
    assert shown == [_key_of(explorer, answer)]


def test_deleting_an_entry_in_the_widget_updates_the_lists(app):
    pytest.importorskip("ipywidgets")

    @app.cache
    def square(x):
        return x * x

    square(1)
    square(2)
    explorer = app.explorer()
    layout = explorer.widget()
    func_selector = layout.children[0].children[0]
    entries_tab = layout.children[1].children[0].children[1]
    entries_selector, delete_button = entries_tab.children[0], entries_tab.children[1]
    assert len(entries_selector.options) == 2

    entries_selector.value = entries_selector.options[0][1]
    delete_button.click()
    assert len(entries_selector.options) == 1

    entries_selector.value = entries_selector.options[0][1]
    delete_button.click()
    assert func_selector.options == ()
