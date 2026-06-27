"""Batch 223 – Observer/event pattern interaction tests.

Tests editing cells with event handling, callback patterns,
and verifying that edits propagate correctly.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestObserverPatternEdits:
    """Editing observer/event patterns."""

    def test_edit_event_handler(self, nb_runner):
        """Edit cell 2 to add a second handler — cash should re-execute."""
        nb_runner.create_notebook([
            "class EventBus:\n    def __init__(self):\n        self.handlers = []\n    def on(self, fn):\n        self.handlers.append(fn)\n    def emit(self, data):\n        return [h(data) for h in self.handlers]",
            "bus = EventBus()\nbus.on(lambda x: x.upper())\nresults = bus.emit('hello')\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        assert "results = ['HELLO']" in nb_runner.get_output(2)

        # Edit cell 2: add a second handler
        nb_runner.set_cell_source(2, "bus = EventBus()\nbus.on(lambda x: x.upper())\nbus.on(lambda x: x[::-1])\nresults = bus.emit('hello')\nprint(f'results = {results}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        raw2 = nb_runner.get_raw_output(2)
        print(f"DEBUG filtered output cell 2: {repr(out2)}")
        print(f"DEBUG raw output cell 2: {repr(raw2)}")
        assert "HELLO" in out2
        assert "olleh" in out2

    def test_edit_callback_list(self, nb_runner):
        """Edit a list of callbacks."""
        nb_runner.create_notebook([
            "transforms = [str.upper, str.strip]",
            "text = '  hello  '\nfor fn in transforms:\n    text = fn(text)\nprint(f'text = {text}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "text = HELLO" in nb_runner.get_output(2)

        # Change transform order
        nb_runner.set_cell_source(1, "transforms = [str.strip, str.title]")
        nb_runner.run_all()
        assert "text = Hello" in nb_runner.get_output(2)

    def test_edit_dispatch_map(self, nb_runner):
        """Edit a dispatch map (command pattern)."""
        nb_runner.create_notebook([
            "dispatch = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b}",
            "result = dispatch['add'](10, 3)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(2)

        # Use sub command
        nb_runner.set_cell_source(2, "result = dispatch['sub'](10, 3)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)
