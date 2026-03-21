"""Batch 484: typing protocol and runtime checkable."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypingProtocolRuntime:
    def test_protocol_duck_typing(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import Protocol, runtime_checkable",
            "@runtime_checkable\nclass Drawable(Protocol):\n    def draw(self) -> str: ...\nclass Circle:\n    def draw(self) -> str: return 'O'\nclass Square:\n    def draw(self) -> str: return '[]'\nc = Circle()\ns = Square()\nprint(f'c_drawable={isinstance(c, Drawable)} s_drawable={isinstance(s, Drawable)}')\nprint(f'c={c.draw()} s={s.draw()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c_drawable=True" in out
        assert "s_drawable=True" in out
        assert "c=O" in out
        assert "s=[]" in out

    def test_non_conforming(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import Protocol, runtime_checkable",
            "@runtime_checkable\nclass Serializable(Protocol):\n    def to_json(self) -> str: ...\nclass Good:\n    def to_json(self) -> str: return '{}'\nclass Bad:\n    pass\nprint(f'good={isinstance(Good(), Serializable)} bad={isinstance(Bad(), Serializable)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "good=True" in out
        assert "bad=False" in out

    def test_protocol_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import Protocol, runtime_checkable",
            "@runtime_checkable\nclass HasLen(Protocol):\n    def __len__(self) -> int: ...\nprint(f'list={isinstance([], HasLen)} int={isinstance(5, HasLen)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "list=True" in out
        assert "int=False" in out
        nb_runner.set_cell_source(2, "@runtime_checkable\nclass HasLen(Protocol):\n    def __len__(self) -> int: ...\nprint(f'str={isinstance(\"hi\", HasLen)} dict={isinstance({}, HasLen)}')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "str=True" in out2
        assert "dict=True" in out2
