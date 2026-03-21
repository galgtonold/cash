"""Batch 83 – complex inheritance: diamonds, MRO, super() chains."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDiamondInheritance:
    """Diamond inheritance patterns and MRO."""

    def test_basic_diamond(self, nb_runner):
        """Classic diamond: A → B,C → D with super() chains."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    def who(self):
                        return ['Base']

                class Left(Base):
                    def who(self):
                        return ['Left'] + super().who()

                class Right(Base):
                    def who(self):
                        return ['Right'] + super().who()

                class Diamond(Left, Right):
                    def who(self):
                        return ['Diamond'] + super().who()

                d = Diamond()
                chain = d.who()
                mro = [c.__name__ for c in Diamond.__mro__]
            """),
            "print(f'chain={chain}')\nprint(f'mro={mro}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Diamond" in out
        assert "Left" in out
        assert "Right" in out
        assert "Base" in out

    def test_mixin_diamond(self, nb_runner):
        """Mixins creating a diamond with cooperative super()."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Serializable:
                    def serialize(self):
                        return {'type': type(self).__name__}

                class JsonMixin(Serializable):
                    def serialize(self):
                        data = super().serialize()
                        data['format'] = 'json'
                        return data

                class LogMixin(Serializable):
                    def serialize(self):
                        data = super().serialize()
                        data['logged'] = True
                        return data

                class ApiModel(JsonMixin, LogMixin):
                    def serialize(self):
                        data = super().serialize()
                        data['api'] = True
                        return data

                m = ApiModel()
                result = m.serialize()
            """),
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "api" in out
        assert "format" in out
        assert "json" in out

    def test_diamond_propagation(self, nb_runner):
        """Change in base class propagates through diamond."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label=v1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "base_label = 'v2'")
        nb_runner.run_cells([1, 2, 3])
        assert "label=v2" in nb_runner.get_output(3)

    def test_multi_level_inheritance(self, nb_runner):
        """3+ levels of inheritance with method overriding."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Animal:
                    def speak(self): return "..."
                    def kind(self): return "animal"

                class Mammal(Animal):
                    def kind(self): return "mammal"

                class Dog(Mammal):
                    def speak(self): return "woof"

                class Puppy(Dog):
                    def speak(self): return "yip"

                animals = [Animal(), Mammal(), Dog(), Puppy()]
                info = [(a.kind(), a.speak()) for a in animals]
            """),
            "print(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "animal" in out
        assert "mammal" in out
        assert "woof" in out
        assert "yip" in out


class TestSuperChains:
    """Advanced super() usage patterns."""

    def test_super_init_chain(self, nb_runner):
        """__init__ chain through multiple inheritance with super()."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class A:
                    def __init__(self):
                        self.log = ['A']

                class B(A):
                    def __init__(self):
                        super().__init__()
                        self.log.append('B')

                class C(A):
                    def __init__(self):
                        super().__init__()
                        self.log.append('C')

                class D(B, C):
                    def __init__(self):
                        super().__init__()
                        self.log.append('D')

                d = D()
                init_order = d.log
            """),
            "print(f'init_order={init_order}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # MRO: D → B → C → A; __init__ runs A first, then C, then B, then D
        assert "A" in out
        assert "B" in out
        assert "C" in out
        assert "D" in out
