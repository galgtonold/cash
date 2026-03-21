"""Batch 82 – match/case (structural pattern matching, Python 3.10+)."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestMatchCase:
    """Tests for match/case statement caching."""

    def test_basic_match(self, nb_runner):
        """match/case with literal patterns."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def classify(code):
                    match code:
                        case 200:
                            return 'ok'
                        case 404:
                            return 'not found'
                        case 500:
                            return 'server error'
                        case _:
                            return 'unknown'
                results = [classify(c) for c in [200, 404, 500, 301]]
            """),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "ok" in out
        assert "not found" in out
        assert "server error" in out
        assert "unknown" in out

    def test_match_sequence_pattern(self, nb_runner):
        """match/case with sequence unpacking patterns."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def parse_command(cmd):
                    match cmd.split():
                        case ["quit"]:
                            return "exit"
                        case ["go", direction]:
                            return f"moving {direction}"
                        case ["get", item, "from", location]:
                            return f"getting {item} from {location}"
                        case _:
                            return "unknown"
                outputs = [
                    parse_command("quit"),
                    parse_command("go north"),
                    parse_command("get key from chest"),
                    parse_command("dance"),
                ]
            """),
            "print(f'outputs={outputs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "exit" in out
        assert "moving north" in out
        assert "getting key from chest" in out
        assert "unknown" in out

    def test_match_class_pattern(self, nb_runner):
        """match/case with class patterns."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass
                class Point:
                    x: float
                    y: float

                @dataclass
                class Circle:
                    center: Point
                    radius: float

                def describe(shape):
                    match shape:
                        case Point(x=0, y=0):
                            return "origin"
                        case Point(x=x, y=y) if x == y:
                            return f"diagonal at {x}"
                        case Circle(center=Point(x=0, y=0), radius=r):
                            return f"centered circle r={r}"
                        case _:
                            return "other"

                labels = [
                    describe(Point(0, 0)),
                    describe(Point(3, 3)),
                    describe(Circle(Point(0, 0), 5)),
                    describe(Point(1, 2)),
                ]
            """),
            "print(f'labels={labels}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "origin" in out
        assert "diagonal at 3" in out
        assert "centered circle r=5" in out
        assert "other" in out

    def test_match_guard_propagation(self, nb_runner):
        """match/case with guard conditions, upstream change propagation."""
        nb_runner.create_notebook([
            "threshold = 50",
            textwrap.dedent("""\
                def categorize(val, thresh):
                    match val:
                        case x if x > thresh:
                            return 'high'
                        case x if x > thresh // 2:
                            return 'medium'
                        case _:
                            return 'low'
                cats = [categorize(v, threshold) for v in [10, 30, 70]]
            """),
            "print(f'cats={cats}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "high" in nb_runner.get_output(3)
        assert "low" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "threshold = 20")
        nb_runner.run_cells([1, 2, 3])
        out = nb_runner.get_output(3)
        # With threshold=20: 10 is low, 30 is high, 70 is high
        assert "high" in out

    def test_match_mapping_pattern(self, nb_runner):
        """match/case with mapping (dict) patterns."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def process_event(event):
                    match event:
                        case {"type": "click", "x": x, "y": y}:
                            return f"click at ({x},{y})"
                        case {"type": "key", "key": k, "mod": "ctrl"}:
                            return f"ctrl+{k}"
                        case {"type": "key", "key": k}:
                            return f"key {k}"
                        case _:
                            return "unknown event"
                results = [
                    process_event({"type": "click", "x": 10, "y": 20}),
                    process_event({"type": "key", "key": "s", "mod": "ctrl"}),
                    process_event({"type": "key", "key": "a"}),
                    process_event({"type": "scroll"}),
                ]
            """),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "click at (10,20)" in out
        assert "ctrl+s" in out
        assert "key a" in out
        assert "unknown event" in out
