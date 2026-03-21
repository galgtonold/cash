"""Batch 237 – Collections module interaction tests.

Tests editing cells using collections types like defaultdict,
Counter, OrderedDict, deque.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestCollectionsEdits:
    """Editing patterns using collections module types."""

    def test_edit_counter_input(self, nb_runner):
        """Edit input to a Counter."""
        nb_runner.create_notebook([
            "from collections import Counter\nwords = 'the cat sat on the mat'.split()",
            "counts = Counter(words)\nmost = counts.most_common(2)\nprint(f'most = {most}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('the', 2)" in nb_runner.get_output(2)

        # Change input text
        nb_runner.set_cell_source(1, "from collections import Counter\nwords = 'a a a b b c'.split()")
        nb_runner.run_all()
        assert "('a', 3)" in nb_runner.get_output(2)

    def test_edit_defaultdict_factory(self, nb_runner):
        """Edit the default factory of a defaultdict."""
        nb_runner.create_notebook([
            "from collections import defaultdict\ndd = defaultdict(int)",
            "dd['x'] += 1\ndd['y'] += 2\nresult = dict(dd)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 1" in out
        assert "'y': 2" in out

        # Change to list factory
        nb_runner.set_cell_source(1, "from collections import defaultdict\ndd = defaultdict(list)")
        nb_runner.set_cell_source(2, "dd['x'].append(1)\ndd['y'].append(2)\ndd['y'].append(3)\nresult = dict(dd)\nprint(f'result = {result}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'x': [1]" in out2
        assert "'y': [2, 3]" in out2

    def test_edit_deque_maxlen(self, nb_runner):
        """Edit deque maxlen parameter."""
        nb_runner.create_notebook([
            "from collections import deque\nd = deque(maxlen=3)",
            "for i in range(5):\n    d.append(i)\nresult = list(d)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        # Change maxlen
        nb_runner.set_cell_source(1, "from collections import deque\nd = deque(maxlen=5)")
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)
