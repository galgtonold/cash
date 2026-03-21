"""Batch 52: Dataclass & NamedTuple advanced patterns — cash caching with typed data."""
import textwrap
import pytest


@pytest.mark.stress
class TestDataclassAdvanced:
    """Test advanced dataclass patterns."""

    def test_dataclass_inheritance(self, nb_runner):
        """Dataclass with inheritance across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass, field

                @dataclass
                class Animal:
                    name: str
                    legs: int = 4

                @dataclass
                class Dog(Animal):
                    breed: str = "mixed"
                    tricks: list = field(default_factory=list)
            """),
            textwrap.dedent("""\
                d = Dog("Rex", breed="Labrador")
                d.tricks.append("sit")
                d.tricks.append("shake")
                print(f"name={d.name} breed={d.breed} tricks={d.tricks} legs={d.legs}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=Rex breed=Labrador" in nb_runner.get_output(2)
        assert "tricks=['sit', 'shake']" in nb_runner.get_output(2)

    def test_frozen_dataclass(self, nb_runner):
        """Frozen (immutable) dataclass."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass(frozen=True)
                class Point:
                    x: float
                    y: float

                    def distance(self):
                        return (self.x ** 2 + self.y ** 2) ** 0.5
            """),
            textwrap.dedent("""\
                p1 = Point(3.0, 4.0)
                p2 = Point(6.0, 8.0)
                d1 = p1.distance()
                d2 = p2.distance()
                print(f"d1={d1} d2={d2}")
                # Frozen allows hashing
                point_set = {p1, p2, Point(3.0, 4.0)}
                print(f"unique={len(point_set)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d1=5.0 d2=10.0" in out
        assert "unique=2" in out

    def test_dataclass_post_init(self, nb_runner):
        """Dataclass with __post_init__ validation."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass
                class Temperature:
                    celsius: float
                    fahrenheit: float = 0.0

                    def __post_init__(self):
                        self.fahrenheit = self.celsius * 9/5 + 32
            """),
            textwrap.dedent("""\
                t1 = Temperature(0)
                t2 = Temperature(100)
                print(f"t1_f={t1.fahrenheit} t2_f={t2.fahrenheit}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "t1_f=32.0 t2_f=212.0" in nb_runner.get_output(2)

    def test_dataclass_change_propagates(self, nb_runner):
        """Changing dataclass definition propagates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from dataclasses import dataclass

                @dataclass
                class Config:
                    name: str
                    value: int = 0
            """),
            textwrap.dedent("""\
                c = Config("test", 10)
                print(f"c={c}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name='test'" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            from dataclasses import dataclass

            @dataclass
            class Config:
                name: str
                value: int = 0
                active: bool = True
        """))
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name='test'" in out
        assert "active=True" in out


@pytest.mark.stress
class TestNamedTupleAdvanced:
    """Test NamedTuple patterns."""

    def test_namedtuple_methods(self, nb_runner):
        """NamedTuple with custom methods."""
        nb_runner.create_notebook([
            "from typing import NamedTuple",
            textwrap.dedent("""\
                class Vector(NamedTuple):
                    x: float
                    y: float
                    z: float = 0.0

                    def magnitude(self):
                        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

                    def __add__(self, other):
                        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
            """),
            textwrap.dedent("""\
                v1 = Vector(1, 2, 3)
                v2 = Vector(4, 5, 6)
                v3 = v1 + v2
                print(f"v3={v3} mag={v3.magnitude():.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "v3=Vector(x=5, y=7, z=9)" in nb_runner.get_output(3)

    def test_namedtuple_as_dict(self, nb_runner):
        """NamedTuple _asdict and _replace."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import namedtuple
                Record = namedtuple('Record', ['id', 'name', 'score'])
                r1 = Record(1, 'Alice', 95.5)
            """),
            textwrap.dedent("""\
                d = r1._asdict()
                r2 = r1._replace(score=98.0)
                print(f"dict={dict(d)}")
                print(f"replaced={r2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'name': 'Alice'" in out
        assert "score=98.0" in out

    def test_multiple_namedtuples(self, nb_runner):
        """Multiple NamedTuples interacting."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from typing import NamedTuple, List

                class Student(NamedTuple):
                    name: str
                    grade: float

                class Classroom(NamedTuple):
                    name: str
                    students: tuple  # NamedTuples are immutable

                students = (Student("Alice", 90), Student("Bob", 85), Student("Charlie", 95))
                room = Classroom("Math101", students)
            """),
            textwrap.dedent("""\
                avg = sum(s.grade for s in room.students) / len(room.students)
                top = max(room.students, key=lambda s: s.grade)
                print(f"room={room.name} avg={avg:.1f} top={top.name}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "room=Math101 avg=90.0 top=Charlie" in nb_runner.get_output(2)
