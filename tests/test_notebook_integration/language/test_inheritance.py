"""Inheritance, super() and the MRO across cells."""

import textwrap

import pytest


# Class hierarchy interaction tests.
#
# Tests editing base classes, overriding methods, adding/removing
# inheritance, and multiple inheritance scenarios.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestInheritanceEdits:
    """Editing class hierarchies."""

    def test_edit_base_class_method(self, nb_runner):
        """Edit a method in the base class, verify subclass picks it up."""
        nb_runner.create_notebook(
            [
                "class Animal:\n    def speak(self):\n        return 'generic sound'",
                "class Dog(Animal):\n    pass",
                "d = Dog()\nresult = d.speak()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = generic sound" in nb_runner.get_output(3)

        # Edit base class
        nb_runner.set_cell_source(1, "class Animal:\n    def speak(self):\n        return 'LOUD sound'")
        nb_runner.run_all()
        assert "result = LOUD sound" in nb_runner.get_output(3)

    def test_add_method_override(self, nb_runner):
        """Add a method override to a subclass."""
        nb_runner.create_notebook(
            [
                "class Shape:\n    def area(self):\n        return 0",
                "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r",
                "c = Circle(5)\nresult = c.area()\nprint(f'area = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 0" in nb_runner.get_output(3)

        # Override area in Circle
        nb_runner.set_cell_source(
            2,
            "import math as _math_mod\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return _math_mod.pi * self.r ** 2",
        )
        nb_runner.set_cell_source(3, "c = Circle(5)\nresult = c.area()\nprint(f'area = {result:.2f}')")
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(3)

    def test_change_parent_class(self, nb_runner):
        """Change which class a subclass inherits from."""
        nb_runner.create_notebook(
            [
                "class Base1:\n    val = 10",
                "class Base2:\n    val = 20",
                "class Child(Base1):\n    pass",
                "result = Child.val\nprint(f'val = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(4)

        # Change parent
        nb_runner.set_cell_source(3, "class Child(Base2):\n    pass")
        nb_runner.run_all()
        assert "val = 20" in nb_runner.get_output(4)


# Class inheritance chain edit tests.
#
# Tests editing base/parent classes and verifying that changes
# propagate through inheritance hierarchies.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestInheritanceChainEdits:
    """Editing classes in an inheritance hierarchy."""

    def test_edit_derived_class_override(self, nb_runner):
        """Edit a derived class to override a base method."""
        nb_runner.create_notebook(
            [
                "class Shape:\n    def describe(self):\n        return 'shape'",
                "class Circle(Shape):\n    pass",
                "c = Circle()\nresult = c.describe()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = shape" in nb_runner.get_output(3)

        # Add override
        nb_runner.set_cell_source(2, "class Circle(Shape):\n    def describe(self):\n        return 'circle'")
        nb_runner.run_all()
        assert "result = circle" in nb_runner.get_output(3)

    def test_edit_super_call(self, nb_runner):
        """Edit a class that uses super()."""
        nb_runner.create_notebook(
            [
                "class Base:\n    def value(self):\n        return 10",
                "class Child(Base):\n    def value(self):\n        return super().value() + 5",
                "obj = Child()\nresult = obj.value()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

        # Edit base value
        nb_runner.set_cell_source(1, "class Base:\n    def value(self):\n        return 100")
        nb_runner.run_all()
        assert "result = 105" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestInheritanceSuper:
    """class inheritance with super() and MRO edits."""

    def test_super_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        return f'{self.name} makes a sound'\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} barks'",
                "d = Dog('Rex')\nresult = d.speak()\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Rex barks" in nb_runner.get_output(2)

    def test_super_chain_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Base:\n    def value(self):\n        return 10\nclass Mid(Base):\n    def value(self):\n        return super().value() + 5\nclass Top(Mid):\n    def value(self):\n        return super().value() * 2",
                "t = Top()\nresult = t.value()\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2)
        # Edit base
        nb_runner.set_cell_source(
            1,
            "class Base:\n    def value(self):\n        return 100\nclass Mid(Base):\n    def value(self):\n        return super().value() + 5\nclass Top(Mid):\n    def value(self):\n        return super().value() * 2",
        )
        nb_runner.run_all()
        assert "result=210" in nb_runner.get_output(2)

    def test_mixin_pattern(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class JsonMixin:\n    def to_dict(self):\n        return self.__dict__\nclass Person(JsonMixin):\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age",
                "p = Person('Alice', 30)\nd = p.to_dict()\nprint(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'name': 'Alice'" in nb_runner.get_output(2)
        assert "'age': 30" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClassInheritanceMRO:
    """class inheritance and method resolution order."""

    def test_single_inheritance(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Animal:\n    def speak(self): return 'generic'\nclass Dog(Animal):\n    def speak(self): return 'woof'\nclass Cat(Animal):\n    def speak(self): return 'meow'",
                "d = Dog()\nc = Cat()\nprint(f'd={d.speak()} c={c.speak()} is_animal={isinstance(d, Animal)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=woof" in out
        assert "c=meow" in out
        assert "is_animal=True" in out

    def test_mro(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class A:\n    val = 'A'\nclass B(A):\n    val = 'B'\nclass C(A):\n    val = 'C'\nclass D(B, C):\n    pass",
                "mro = [cls.__name__ for cls in D.__mro__]\nval = D.val\nprint(f'mro={mro} val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mro=['D', 'B', 'C', 'A', 'object']" in nb_runner.get_output(2)
        assert "val=B" in nb_runner.get_output(2)

    def test_inheritance_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Shape:\n    def area(self): return 0\nclass Square(Shape):\n    def __init__(self, s): self.s = s\n    def area(self): return self.s ** 2",
                "sq = Square(5)\nprint(f'area={sq.area()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "sq = Square(10)\nprint(f'area={sq.area()}')")
        nb_runner.run_all()
        assert "area=100" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultipleInheritance:
    """Multiple inheritance edits."""

    def test_mixin_edit(self, nb_runner):
        """Edit a mixin class in a multiple inheritance chain."""
        nb_runner.create_notebook(
            [
                "class LogMixin:\n    def log(self):\n        return 'LOG:base'",
                "class Service:\n    name = 'svc'",
                "class App(LogMixin, Service):\n    pass",
                "a = App()\nprint(f'log={a.log()} name={a.name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=LOG:base" in nb_runner.get_output(4)
        assert "name=svc" in nb_runner.get_output(4)

        # Edit mixin
        nb_runner.set_cell_source(1, "class LogMixin:\n    def log(self):\n        return 'LOG:v2'")
        nb_runner.run_all()
        assert "log=LOG:v2" in nb_runner.get_output(4)

    def test_add_class_attribute(self, nb_runner):
        """Add a class attribute to a parent, use in child."""
        nb_runner.create_notebook(
            [
                "class Config:\n    debug = False",
                "class App(Config):\n    name = 'myapp'",
                "a = App()\nprint(f'debug={a.debug} name={a.name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "debug=False" in nb_runner.get_output(3)

        # Enable debug in parent
        nb_runner.set_cell_source(1, "class Config:\n    debug = True")
        nb_runner.run_all()
        assert "debug=True" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMultipleInheritanceMRO:
    """multiple inheritance and MRO resolution."""

    def test_diamond_mro(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class A:\n    def who(self): return 'A'\nclass B(A):\n    def who(self): return 'B'\nclass C(A):\n    def who(self): return 'C'\nclass D(B, C):\n    pass\nd = D()\nmro = [c.__name__ for c in D.__mro__]\nprint(f'who={d.who()} mro={mro}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "who=B" in out
        assert "mro=['D', 'B', 'C', 'A', 'object']" in out

    def test_super_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class Base:\n    def greet(self): return 'Base'\nclass Left(Base):\n    def greet(self): return 'Left+' + super().greet()\nclass Right(Base):\n    def greet(self): return 'Right+' + super().greet()\nclass Child(Left, Right):\n    def greet(self): return 'Child+' + super().greet()\nresult = Child().greet()\nprint(f'chain={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chain=Child+Left+Right+Base" in nb_runner.get_output(2)

    def test_mro_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "class X:\n    val = 10\nclass Y(X):\n    val = 20\nclass Z(Y):\n    pass\nprint(f'val={Z.val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=20" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "class X:\n    val = 10\nclass Y(X):\n    pass\nclass Z(Y):\n    pass\nprint(f'val={Z.val}')"
        )
        nb_runner.run_all()
        assert "val=10" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDiamondMRO:
    """multiple inheritance diamond pattern and MRO."""

    def test_diamond_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class A:\n    def who(self):\n        return 'A'\nclass B(A):\n    def who(self):\n        return 'B'\nclass C(A):\n    def who(self):\n        return 'C'\nclass D(B, C):\n    pass",
                "d = D()\nresult = d.who()\nmro = [c.__name__ for c in D.__mro__]\nprint(f'result={result} mro={mro}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=B" in nb_runner.get_output(2)
        assert "mro=['D', 'B', 'C', 'A', 'object']" in nb_runner.get_output(2)

    def test_diamond_edit_order(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class A:\n    val = 1\nclass B(A):\n    val = 2\nclass C(A):\n    val = 3\nclass D(B, C):\n    pass",
                "result = D.val\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=2" in nb_runner.get_output(2)
        # Switch MRO order
        nb_runner.set_cell_source(
            1, "class A:\n    val = 1\nclass B(A):\n    val = 2\nclass C(A):\n    val = 3\nclass D(C, B):\n    pass"
        )
        nb_runner.run_all()
        assert "result=3" in nb_runner.get_output(2)

    def test_super_cooperative(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Base:\n    def __init__(self):\n        self.log = ['Base']\nclass Left(Base):\n    def __init__(self):\n        super().__init__()\n        self.log.append('Left')\nclass Right(Base):\n    def __init__(self):\n        super().__init__()\n        self.log.append('Right')\nclass Child(Left, Right):\n    def __init__(self):\n        super().__init__()\n        self.log.append('Child')",
                "c = Child()\nprint(f'log={c.log}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=['Base', 'Right', 'Left', 'Child']" in nb_runner.get_output(2)


# Complex inheritance & MRO patterns — diamond, mixin, super() chains.
class TestDiamondInheritance:
    """Test diamond inheritance and MRO."""

    @pytest.mark.stress
    def test_diamond_change_base(self, nb_runner):
        """Changing base class propagates through diamond."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Base:
                    value = 10
                class Left(Base): pass
                class Right(Base): pass
                class Diamond(Left, Right): pass
            """),
                textwrap.dedent("""\
                result = Diamond.value
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)

        # Change base
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Base:
                value = 99
            class Left(Base): pass
            class Right(Base): pass
            class Diamond(Left, Right): pass
        """),
        )
        nb_runner.run_all()
        assert "result=99" in nb_runner.get_output(2)

    # complex inheritance: diamonds, MRO, super() chains.
    @pytest.mark.stress
    @pytest.mark.integration
    def test_diamond_propagation(self, nb_runner):
        """Change in base class propagates through diamond."""
        nb_runner.create_notebook(
            [
                "base_label = 'v1'",
                textwrap.dedent("""\
                class Base:
                    label = base_label
                class Left(Base): pass
                class Right(Base): pass
                class Diamond(Left, Right): pass
                d = Diamond()
            """),
                "print(f'label={d.label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=v1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "base_label = 'v2'")
        nb_runner.run_cells([1, 2, 3])
        assert "label=v2" in nb_runner.get_output(3)


# Interaction test: class method resolution order (MRO) with cooperative super().
# Tests diamond inheritance, MRO resolution, and super() chain
# across cells with method override behavior.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMROCooperativeSuper:
    """Test MRO and cooperative super() across cells."""

    def test_diamond_mro(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define diamond hierarchy
                "class Base:\n    def greet(self):\n        return 'Base'\nclass Left(Base):\n    def greet(self):\n        return 'Left+' + super().greet()\nclass Right(Base):\n    def greet(self):\n        return 'Right+' + super().greet()\nclass Diamond(Left, Right):\n    def greet(self):\n        return 'Diamond+' + super().greet()\nprint('Diamond hierarchy defined')",
                # Cell 2: test MRO
                "d = Diamond()\nresult = d.greet()\nmro = [c.__name__ for c in Diamond.__mro__]\nprint(f'result={result}')\nprint(f'mro={mro}')",
                # Cell 3: verify chain
                "parts = result.split('+')\nprint(f'chain_len={len(parts)}')\nprint(f'starts_with_diamond={parts[0] == \"Diamond\"}')\nprint(f'ends_with_base={parts[-1] == \"Base\"}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "result=Diamond+Left+Right+Base" in out2
        assert "mro=['Diamond', 'Left', 'Right', 'Base', 'object']" in out2
        out3 = nb_runner.get_output(3)
        assert "chain_len=4" in out3
        assert "starts_with_diamond=True" in out3
        assert "ends_with_base=True" in out3

    def test_mro_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class A:\n    def who(self):\n        return 'A'\nclass B(A):\n    def who(self):\n        return 'B+' + super().who()\nclass C(A):\n    def who(self):\n        return 'C+' + super().who()\nclass D(B, C):\n    def who(self):\n        return 'D+' + super().who()\nprint('ABCD defined')",
                "d = D()\nprint(f'who={d.who()}')",
                "length = len(d.who().split('+'))\nprint(f'depth={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "who=D+B+C+A" in nb_runner.get_output(2)
        assert "depth=4" in nb_runner.get_output(3)

        # Edit C to not call super
        nb_runner.set_cell_source(
            1,
            "class A:\n    def who(self):\n        return 'A'\nclass B(A):\n    def who(self):\n        return 'B+' + super().who()\nclass C(A):\n    def who(self):\n        return 'C_STOP'\nclass D(B, C):\n    def who(self):\n        return 'D+' + super().who()\nprint('ABCD redefined')",
        )
        nb_runner.run_cells([1, 2, 3])
        assert "who=D+B+C_STOP" in nb_runner.get_output(2)
        assert "depth=3" in nb_runner.get_output(3)

    def test_mro_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class X:\n    val = 'X'\nclass Y(X):\n    val = 'Y'\nclass Z(Y):\n    pass  # inherits from Y\nprint('XYZ defined')",
                "z = Z()\nprint(f'val={z.val}')\nprint(f'mro={[c.__name__ for c in Z.__mro__]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=Y" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "val=Y" in nb_runner.get_output(2)


# Multi-level inheritance and mixin interaction tests.
# Tests base/derived class method changes with 3-level inheritance and mixins.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMultiLevelInheritanceInteraction:
    """Test multi-level inheritance and mixin patterns with cache invalidation."""

    def test_base_greeting_edit(self, nb_runner):
        """Editing base class greeting should propagate to derived."""
        nb_runner.create_notebook(
            [
                (
                    "class Animal:\n"
                    "    def __init__(self, name):\n"
                    "        self.name = name\n"
                    "    def greeting(self):\n"
                    "        return f'I am {self.name}'"
                ),
                ("class Dog(Animal):\n    def speak(self):\n        return f'{self.greeting()} and I bark'"),
                "d = Dog('Rex')\nmsg = d.speak()",
                "print(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=I am Rex and I bark" in out

        nb_runner.set_cell_source(
            1,
            (
                "class Animal:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "    def greeting(self):\n"
                "        return f'Call me {self.name}'"
            ),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Call me Rex and I bark" in out

    def test_three_level_square_edit(self, nb_runner):
        """Three-level inheritance: Shape > Rectangle > Square."""
        nb_runner.create_notebook(
            [
                "class Shape:\n    def area(self):\n        return 0",
                "class Rectangle(Shape):\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    def area(self):\n        return self.w * self.h",
                "class Square(Rectangle):\n    def __init__(self, s):\n        super().__init__(s, s)",
                "sq = Square(5)\na = sq.area()",
                "print(f'area={a}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class JsonMixin:\n    def to_json(self):\n        import json\n        return json.dumps(self.__dict__)",
                "class User(JsonMixin):\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age",
                "u = User('Alice', 30)\nj = u.to_json()",
                "print(f'json={j}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Alice" in out

        nb_runner.set_cell_source(
            1,
            "class JsonMixin:\n    def to_json(self):\n        import json\n        d = {'_type': self.__class__.__name__}\n        d.update(self.__dict__)\n        return json.dumps(d)",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "_type" in out
        assert "User" in out


@pytest.mark.stress
class TestMixinPatterns:
    """Test mixin class patterns."""

    def test_multiple_mixins(self, nb_runner):
        """Multiple mixins providing different features."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class JsonMixin:
                    def to_json(self):
                        import json
                        d = {k: v for k, v in self.__dict__.items() if not k.startswith('_cash')}
                        return json.dumps(d, sort_keys=True)

                class LogMixin:
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self._log = []
                    def log(self, msg):
                        self._log.append(msg)

                class ValidateMixin:
                    def validate(self):
                        for k, v in self.__dict__.items():
                            if k.startswith('_'):
                                continue
                            if v is None:
                                return False
                        return True
            """),
                textwrap.dedent("""\
                class User(JsonMixin, LogMixin, ValidateMixin):
                    def __init__(self, name, email):
                        super().__init__()
                        self.name = name
                        self.email = email

                u = User("Alice", "alice@example.com")
                u.log("created")
                json_out = u.to_json()
                valid = u.validate()
                print(f"json={json_out}")
                print(f"valid={valid} log_count={len(u._log)}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert '"name": "Alice"' in out
        assert "valid=True" in out

    def test_mixin_evolution(self, nb_runner):
        """Evolving mixin adds new method."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class PrintMixin:
                    def describe(self):
                        return f"Object with {len(self.__dict__)} attrs"
            """),
                textwrap.dedent("""\
                class Item(PrintMixin):
                    def __init__(self, name, price):
                        self.name = name
                        self.price = price

                item = Item("Widget", 9.99)
                desc = item.describe()
                print(f"desc={desc}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Object with" in out

        # Evolve mixin
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class PrintMixin:
                def describe(self):
                    attrs = ', '.join(f'{k}={v}' for k, v in sorted(self.__dict__.items()) if not k.startswith('_cash'))
                    return f"Object({attrs})"
        """),
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "name=Widget" in out2
        assert "price=9.99" in out2


# Class composition patterns.
#
# Tests composition (has-a) relationships with edits.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestClassComposition:
    """Composition-based class patterns with edit propagation."""

    def test_engine_in_car(self, nb_runner):
        """Edit composed engine class, car reflects change."""
        nb_runner.create_notebook(
            [
                "class Engine:\n    def __init__(self, hp):\n        self.hp = hp\n    def describe(self):\n        return f'{self.hp}hp'",
                "class Car:\n    def __init__(self, name, engine):\n        self.name = name\n        self.engine = engine\n    def spec(self):\n        return f'{self.name}: {self.engine.describe()}'",
                "e = Engine(200)\nc = Car('Sedan', e)\nprint(f'spec = {c.spec()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "spec = Sedan: 200hp" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "class Engine:\n    def __init__(self, hp):\n        self.hp = hp\n    def describe(self):\n        return f'{self.hp}HP turbo'",
        )
        nb_runner.run_all()
        assert "spec = Sedan: 200HP turbo" in nb_runner.get_output(3)

    def test_strategy_pattern(self, nb_runner):
        """Edit strategy object, context reflects new behavior."""
        nb_runner.create_notebook(
            [
                "class AddStrategy:\n    def execute(self, a, b):\n        return a + b",
                "strategy = AddStrategy()\nresult = strategy.execute(10, 20)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class AddStrategy:\n    def execute(self, a, b):\n        return a * b",
        )
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)

    def test_nested_composition(self, nb_runner):
        """Three-level composition: department -> team -> member."""
        nb_runner.create_notebook(
            [
                "class Member:\n    def __init__(self, name):\n        self.name = name",
                "class Team:\n    def __init__(self, members):\n        self.members = members\n    def names(self):\n        return [m.name for m in self.members]",
                "t = Team([Member('Alice'), Member('Bob')])\nprint(f'names = {t.names()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            3,
            "t = Team([Member('Charlie'), Member('Diana'), Member('Eve')])\nprint(f'names = {t.names()}')",
        )
        nb_runner.run_all()
        assert "names = ['Charlie', 'Diana', 'Eve']" in nb_runner.get_output(3)


# Multi-cell class composition (has-a) interaction tests.
#
# Tests where one class has another class as a member,
# and edits propagate through the composition.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestCompositionEdits:
    """Editing composed class structures."""

    def test_edit_component_class(self, nb_runner):
        """Edit the component class in a composition."""
        nb_runner.create_notebook(
            [
                "class Engine:\n    def __init__(self, hp):\n        self.hp = hp\n    def describe(self):\n        return f'{self.hp}hp'",
                "class Car:\n    def __init__(self, name, engine):\n        self.name = name\n        self.engine = engine\n    def info(self):\n        return f'{self.name}: {self.engine.describe()}'",
                "e = Engine(200)\nc = Car('Tesla', e)\nprint(f'info = {c.info()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info = Tesla: 200hp" in nb_runner.get_output(3)

        # Edit Engine to include type
        nb_runner.set_cell_source(
            1,
            "class Engine:\n    def __init__(self, hp, typ='gas'):\n        self.hp = hp\n        self.typ = typ\n    def describe(self):\n        return f'{self.hp}hp {self.typ}'",
        )
        nb_runner.set_cell_source(3, "e = Engine(300, 'electric')\nc = Car('Tesla', e)\nprint(f'info = {c.info()}')")
        nb_runner.run_all()
        assert "info = Tesla: 300hp electric" in nb_runner.get_output(3)

    def test_edit_container_class(self, nb_runner):
        """Edit the container class in a composition."""
        nb_runner.create_notebook(
            [
                "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price",
                "class Cart:\n    def __init__(self):\n        self.items = []\n    def add(self, item):\n        self.items.append(item)\n    def total(self):\n        return sum(i.price for i in self.items)",
                "cart = Cart()\ncart.add(Item('A', 10))\ncart.add(Item('B', 20))\nprint(f'total = {cart.total()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        # Edit Cart to add tax
        nb_runner.set_cell_source(
            2,
            "class Cart:\n    def __init__(self, tax=0.1):\n        self.items = []\n        self.tax = tax\n    def add(self, item):\n        self.items.append(item)\n    def total(self):\n        subtotal = sum(i.price for i in self.items)\n        return subtotal * (1 + self.tax)",
        )
        nb_runner.run_all()
        assert "total = 33.0" in nb_runner.get_output(3)

    def test_edit_both_classes(self, nb_runner):
        """Edit both component and container."""
        nb_runner.create_notebook(
            [
                "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
                "class Line:\n    def __init__(self, p1, p2):\n        self.p1 = p1\n        self.p2 = p2\n    def length(self):\n        return ((self.p2.x - self.p1.x)**2 + (self.p2.y - self.p1.y)**2) ** 0.5",
                "a = Point(0, 0)\nb = Point(3, 4)\nline = Line(a, b)\nprint(f'length = {line.length()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length = 5.0" in nb_runner.get_output(3)

        # Change points
        nb_runner.set_cell_source(
            3,
            "a = Point(1, 1)\nb = Point(4, 5)\nline = Line(a, b)\nprint(f'length = {line.length()}')",
        )
        nb_runner.run_all()
        assert "length = 5.0" in nb_runner.get_output(3)
