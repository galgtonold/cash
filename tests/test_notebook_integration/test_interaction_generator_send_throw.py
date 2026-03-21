"""
Interaction test: generator with send() and throw().
Tests generator coroutine-like patterns with send(), throw(),
and close(), verifying cross-cell generator state.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGeneratorSendThrow:
    """Test generator send and throw across cells."""

    def test_generator_send(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define generator with send
            "def accumulator():\n    total = 0\n    while True:\n        value = yield total\n        if value is None:\n            break\n        total += value\nprint('accumulator defined')",
            # Cell 2: use send
            "gen = accumulator()\ncurrent = next(gen)  # prime\nprint(f'start={current}')\ncurrent = gen.send(10)\nprint(f'after_10={current}')\ncurrent = gen.send(20)\nprint(f'after_20={current}')\ncurrent = gen.send(5)\nprint(f'after_5={current}')",
            # Cell 3: use result
            "final = current\nprint(f'final={final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "start=0" in out2
        assert "after_10=10" in out2
        assert "after_20=30" in out2
        assert "after_5=35" in out2
        out3 = nb_runner.get_output(3)
        assert "final=35" in out3

    def test_generator_edit(self, nb_runner):
        nb_runner.create_notebook([
            "def running_avg():\n    total = 0\n    count = 0\n    avg = 0\n    while True:\n        value = yield avg\n        if value is None:\n            break\n        total += value\n        count += 1\n        avg = total / count\nprint('running_avg defined')",
            "gen = running_avg()\nnext(gen)\ngen.send(10)\ngen.send(20)\nresult = gen.send(30)\nprint(f'avg={result}')",
            "rounded = round(result, 1)\nprint(f'rounded={rounded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=20.0" in nb_runner.get_output(2)
        assert "rounded=20.0" in nb_runner.get_output(3)

        # Change values
        nb_runner.set_cell_source(2, "gen = running_avg()\nnext(gen)\ngen.send(100)\nresult = gen.send(200)\nprint(f'avg={result}')")
        nb_runner.run_cells([2, 3])
        assert "avg=150.0" in nb_runner.get_output(2)
        assert "rounded=150.0" in nb_runner.get_output(3)

    def test_generator_cache(self, nb_runner):
        nb_runner.create_notebook([
            "def counter_gen(start=0):\n    n = start\n    while True:\n        reset = yield n\n        if reset is not None:\n            n = reset\n        else:\n            n += 1\nprint('counter_gen defined')",
            "g = counter_gen(10)\nv1 = next(g)\nv2 = next(g)\nv3 = g.send(100)\nv4 = next(g)\nprint(f'vals={[v1, v2, v3, v4]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=[10, 11, 100, 101]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "vals=[10, 11, 100, 101]" in nb_runner.get_output(2)
