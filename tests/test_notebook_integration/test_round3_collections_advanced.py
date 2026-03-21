"""Batch 85 – collections advanced: defaultdict, ChainMap, deque, Counter."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDefaultDict:
    """defaultdict patterns."""

    def test_defaultdict_grouping(self, nb_runner):
        """Group items using defaultdict(list)."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import defaultdict
                words = ['apple', 'banana', 'avocado', 'blueberry', 'cherry', 'apricot']
                by_letter = defaultdict(list)
                for w in words:
                    by_letter[w[0]].append(w)
                grouped = dict(by_letter)
            """),
            "print(f'grouped={grouped}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple" in out
        assert "banana" in out
        assert "cherry" in out

    def test_defaultdict_counter(self, nb_runner):
        """defaultdict(int) as a manual counter."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import defaultdict
                text = "hello world hello python world"
                counts = defaultdict(int)
                for word in text.split():
                    counts[word] += 1
                result = dict(counts)
            """),
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "hello" in out
        assert "2" in out


class TestChainMap:
    """ChainMap patterns."""

    def test_chainmap_layered_config(self, nb_runner):
        """ChainMap for layered configuration."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import ChainMap
                defaults = {'color': 'red', 'size': 10, 'font': 'Arial'}
                user_prefs = {'color': 'blue', 'size': 14}
                session = {'color': 'green'}
                config = ChainMap(session, user_prefs, defaults)
                final_color = config['color']
                final_size = config['size']
                final_font = config['font']
            """),
            "print(f'color={final_color} size={final_size} font={final_font}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "color=green" in out
        assert "size=14" in out
        assert "font=Arial" in out


class TestDeque:
    """deque patterns."""

    def test_deque_sliding_window(self, nb_runner):
        """Sliding window using deque with maxlen."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import deque
                data = [1, 5, 3, 8, 2, 9, 4, 7]
                window_size = 3
                window = deque(maxlen=window_size)
                averages = []
                for val in data:
                    window.append(val)
                    if len(window) == window_size:
                        averages.append(round(sum(window) / window_size, 2))
            """),
            "print(f'averages={averages}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "3.0" in out  # (1+5+3)/3
        assert "averages=" in out

    def test_deque_rotation(self, nb_runner):
        """deque rotate operations."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import deque
                d = deque([1, 2, 3, 4, 5])
                d.rotate(2)
                right = list(d)
                d.rotate(-3)
                left = list(d)
            """),
            "print(f'right={right}')\nprint(f'left={left}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "right=[4, 5, 1, 2, 3]" in out
        assert "left=[2, 3, 4, 5, 1]" in out


class TestCounterAdvanced:
    """Counter advanced usage."""

    def test_counter_operations(self, nb_runner):
        """Counter arithmetic operations."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import Counter
                a = Counter('aabbbcccc')
                b = Counter('abcdddd')
                combined = a + b
                diff = a - b
                intersection = a & b
                union = a | b
                results = {
                    'combined_a': combined['a'],
                    'diff_c': diff['c'],
                    'inter_b': intersection['b'],
                    'union_d': union['d'],
                }
            """),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "combined_a" in out
        assert "results=" in out
