"""Batch 45: Weakref & memory patterns — cash caching with weak references and GC."""
import textwrap
import pytest


@pytest.mark.stress
class TestWeakrefBasics:
    """Test weak references and garbage collection."""

    def test_weakref_to_class_instance(self, nb_runner):
        """Weak reference to class instance across cells."""
        nb_runner.create_notebook([
            "import weakref",
            textwrap.dedent("""\
                class Node:
                    def __init__(self, value):
                        self.value = value
                    def __repr__(self):
                        return f"Node({self.value})"

                obj = Node(42)
                ref = weakref.ref(obj)
                print(f"alive={ref() is not None} val={ref().value}")
            """),
            textwrap.dedent("""\
                # Access through weakref
                result = ref().value * 2
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "alive=True val=42" in nb_runner.get_output(2)
        assert "result=84" in nb_runner.get_output(3)

    def test_weakref_set(self, nb_runner):
        """WeakSet caching across cells."""
        nb_runner.create_notebook([
            "import weakref",
            textwrap.dedent("""\
                class Item:
                    def __init__(self, name):
                        self.name = name

                items = [Item("a"), Item("b"), Item("c")]
                ws = weakref.WeakSet(items)
                print(f"count={len(ws)}")
            """),
            textwrap.dedent("""\
                names = sorted([i.name for i in ws])
                print(f"names={names}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(2)
        assert "names=['a', 'b', 'c']" in nb_runner.get_output(3)

    def test_weakvalue_dict(self, nb_runner):
        """WeakValueDictionary pattern."""
        nb_runner.create_notebook([
            "import weakref",
            textwrap.dedent("""\
                class CacheEntry:
                    def __init__(self, data):
                        self.data = data

                cache = weakref.WeakValueDictionary()
                entries = []
                for i in range(3):
                    e = CacheEntry(f"data_{i}")
                    cache[f"key_{i}"] = e
                    entries.append(e)  # keep strong refs
                print(f"cache_size={len(cache)}")
            """),
            textwrap.dedent("""\
                values = [cache[k].data for k in sorted(cache.keys())]
                print(f"values={values}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cache_size=3" in nb_runner.get_output(2)
        assert "values=['data_0', 'data_1', 'data_2']" in nb_runner.get_output(3)


@pytest.mark.stress
class TestSlotPatterns:
    """Test __slots__ and memory-optimized classes."""

    def test_slots_class(self, nb_runner):
        """Class with __slots__ caching."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Point:
                    __slots__ = ('x', 'y')
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                    def __repr__(self):
                        return f"Point({self.x}, {self.y})"

                p = Point(3, 4)
                print(f"p={p}")
            """),
            textwrap.dedent("""\
                distance = (p.x ** 2 + p.y ** 2) ** 0.5
                print(f"distance={distance}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p=Point(3, 4)" in nb_runner.get_output(1)
        assert "distance=5.0" in nb_runner.get_output(2)

    def test_slots_inheritance(self, nb_runner):
        """Slots with inheritance."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    __slots__ = ('x',)
                    def __init__(self, x):
                        self.x = x

                class Derived(Base):
                    __slots__ = ('y',)
                    def __init__(self, x, y):
                        super().__init__(x)
                        self.y = y

                d = Derived(10, 20)
                print(f"x={d.x} y={d.y}")
            """),
            textwrap.dedent("""\
                total = d.x + d.y
                print(f"total={total}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=10 y=20" in nb_runner.get_output(1)
        assert "total=30" in nb_runner.get_output(2)


@pytest.mark.stress
class TestFinalizationPatterns:
    """Test finalizers and cleanup patterns."""

    def test_weakref_finalize(self, nb_runner):
        """weakref.finalize callback registration."""
        nb_runner.create_notebook([
            "import weakref",
            textwrap.dedent("""\
                cleanup_log = []

                class Resource:
                    def __init__(self, name):
                        self.name = name
                        weakref.finalize(self, lambda n: cleanup_log.append(f"cleaned:{n}"), name)

                r1 = Resource("res1")
                r2 = Resource("res2")
                print(f"resources_created=2 log={cleanup_log}")
            """),
            textwrap.dedent("""\
                # Resources still alive
                print(f"r1={r1.name} r2={r2.name} log={cleanup_log}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "resources_created=2" in nb_runner.get_output(2)
        assert "r1=res1 r2=res2" in nb_runner.get_output(3)

    def test_singleton_via_weakref(self, nb_runner):
        """Singleton-like pattern using weakref."""
        nb_runner.create_notebook([
            "import weakref",
            textwrap.dedent("""\
                class Singleton:
                    _instances = weakref.WeakValueDictionary()

                    def __new__(cls, name):
                        if name in cls._instances:
                            return cls._instances[name]
                        instance = super().__new__(cls)
                        instance.name = name
                        cls._instances[name] = instance
                        return instance

                a = Singleton("shared")
                b = Singleton("shared")
                print(f"same={a is b} name={a.name}")
            """),
            textwrap.dedent("""\
                c = Singleton("other")
                print(f"diff={a is not c} names={a.name},{c.name}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "same=True name=shared" in nb_runner.get_output(2)
        assert "diff=True names=shared,other" in nb_runner.get_output(3)
