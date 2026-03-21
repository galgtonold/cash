"""Batch 354: class inheritance with super() and MRO edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestInheritanceSuper:
    def test_super_basic(self, nb_runner):
        nb_runner.create_notebook([
            "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        return f'{self.name} makes a sound'\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} barks'",
            "d = Dog('Rex')\nresult = d.speak()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Rex barks" in nb_runner.get_output(2)

    def test_super_chain_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Base:\n    def value(self):\n        return 10\nclass Mid(Base):\n    def value(self):\n        return super().value() + 5\nclass Top(Mid):\n    def value(self):\n        return super().value() * 2",
            "t = Top()\nresult = t.value()\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2)
        # Edit base
        nb_runner.set_cell_source(1, "class Base:\n    def value(self):\n        return 100\nclass Mid(Base):\n    def value(self):\n        return super().value() + 5\nclass Top(Mid):\n    def value(self):\n        return super().value() * 2")
        nb_runner.run_all()
        assert "result=210" in nb_runner.get_output(2)

    def test_mixin_pattern(self, nb_runner):
        nb_runner.create_notebook([
            "class JsonMixin:\n    def to_dict(self):\n        return self.__dict__\nclass Person(JsonMixin):\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age",
            "p = Person('Alice', 30)\nd = p.to_dict()\nprint(f'd={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'name': 'Alice'" in nb_runner.get_output(2)
        assert "'age': 30" in nb_runner.get_output(2)
