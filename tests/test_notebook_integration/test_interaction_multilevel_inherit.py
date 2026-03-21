"""
Batch 304: Multi-level inheritance and mixin interaction tests.
Tests base/derived class method changes with 3-level inheritance and mixins.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiLevelInheritanceInteraction:
    """Test multi-level inheritance and mixin patterns with cache invalidation."""

    def test_base_greeting_edit(self, nb_runner):
        """Editing base class greeting should propagate to derived."""
        nb_runner.create_notebook([
            (
                "class Animal:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "    def greeting(self):\n"
                "        return f'I am {self.name}'"
            ),
            (
                "class Dog(Animal):\n"
                "    def speak(self):\n"
                "        return f'{self.greeting()} and I bark'"
            ),
            "d = Dog('Rex')\nmsg = d.speak()",
            "print(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=I am Rex and I bark" in out

        nb_runner.set_cell_source(1, (
            "class Animal:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "    def greeting(self):\n"
            "        return f'Call me {self.name}'"
        ))
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Call me Rex and I bark" in out

    def test_three_level_square_edit(self, nb_runner):
        """Three-level inheritance: Shape > Rectangle > Square."""
        nb_runner.create_notebook([
            "class Shape:\n    def area(self):\n        return 0",
            "class Rectangle(Shape):\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    def area(self):\n        return self.w * self.h",
            "class Square(Rectangle):\n    def __init__(self, s):\n        super().__init__(s, s)",
            "sq = Square(5)\na = sq.area()",
            "print(f'area={a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "area=25" in out

        nb_runner.set_cell_source(4, "sq = Square(10)\na = sq.area()")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "area=100" in out

    def test_mixin_json_edit(self, nb_runner):
        """Editing a mixin method should propagate to mixed-in class."""
        nb_runner.create_notebook([
            "class JsonMixin:\n    def to_json(self):\n        import json\n        return json.dumps(self.__dict__)",
            "class User(JsonMixin):\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age",
            "u = User('Alice', 30)\nj = u.to_json()",
            "print(f'json={j}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Alice" in out

        nb_runner.set_cell_source(1, "class JsonMixin:\n    def to_json(self):\n        import json\n        d = {'_type': self.__class__.__name__}\n        d.update(self.__dict__)\n        return json.dumps(d)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "_type" in out
        assert "User" in out
