"""Batch 72: Advanced enum patterns — cash caching with Enum, Flag, IntEnum."""
import textwrap
import pytest


@pytest.mark.stress
class TestEnumAdvanced:
    """Test advanced enum patterns across cells."""

    def test_intflag_bitwise(self, nb_runner):
        """IntFlag with bitwise operations across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import IntFlag

                class Permission(IntFlag):
                    READ = 1
                    WRITE = 2
                    EXECUTE = 4
                    ADMIN = READ | WRITE | EXECUTE

                user_perms = Permission.READ | Permission.WRITE
                admin_perms = Permission.ADMIN
                print(f"user={user_perms.value} admin={admin_perms.value}")
            """),
            textwrap.dedent("""\
                can_execute = bool(user_perms & Permission.EXECUTE)
                can_read = bool(user_perms & Permission.READ)
                print(f"exec={can_execute} read={can_read}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "user=3 admin=7" in nb_runner.get_output(1)
        assert "exec=False" in nb_runner.get_output(2)
        assert "read=True" in nb_runner.get_output(2)

    def test_enum_with_methods(self, nb_runner):
        """Enum with custom methods across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum

                class Season(Enum):
                    SPRING = 1
                    SUMMER = 2
                    AUTUMN = 3
                    WINTER = 4

                    def next(self):
                        members = list(Season)
                        idx = (members.index(self) + 1) % len(members)
                        return members[idx]

                    @property
                    def is_warm(self):
                        return self in (Season.SPRING, Season.SUMMER)

                current = Season.AUTUMN
                next_season = current.next()
                print(f"current={current.name} next={next_season.name}")
            """),
            textwrap.dedent("""\
                warm_seasons = [s for s in Season if s.is_warm]
                names = [s.name for s in warm_seasons]
                print(f"warm={names}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "current=AUTUMN next=WINTER" in nb_runner.get_output(1)
        assert "warm=['SPRING', 'SUMMER']" in nb_runner.get_output(2)

    def test_enum_change_propagation(self, nb_runner):
        """Enum value change propagation."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum

                class Color(Enum):
                    RED = '#FF0000'
                    GREEN = '#00FF00'
                    BLUE = '#0000FF'

                selected = Color.RED
            """),
            textwrap.dedent("""\
                print(f"color={selected.name} hex={selected.value}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color=RED hex=#FF0000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            from enum import Enum

            class Color(Enum):
                RED = '#FF0000'
                GREEN = '#00FF00'
                BLUE = '#0000FF'

            selected = Color.BLUE
        """))
        nb_runner.run_cells([1, 2])
        assert "color=BLUE hex=#0000FF" in nb_runner.get_output(2)


@pytest.mark.stress
class TestEnumAutoAndFunctional:
    """Test auto() and functional Enum creation."""

    def test_auto_enum(self, nb_runner):
        """Enum with auto() values across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum, auto

                class Priority(Enum):
                    LOW = auto()
                    MEDIUM = auto()
                    HIGH = auto()
                    CRITICAL = auto()

                tasks = [
                    ('Deploy', Priority.HIGH),
                    ('Refactor', Priority.LOW),
                    ('Fix bug', Priority.CRITICAL),
                    ('Test', Priority.MEDIUM),
                ]
            """),
            textwrap.dedent("""\
                sorted_tasks = sorted(tasks, key=lambda t: t[1].value, reverse=True)
                for name, prio in sorted_tasks:
                    print(f"  {prio.name}: {name}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "CRITICAL: Fix bug" in out
        # CRITICAL (4) should come first
        assert out.index("CRITICAL") < out.index("LOW")

    def test_functional_enum(self, nb_runner):
        """Functional Enum creation across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from enum import Enum

                Status = Enum('Status', ['PENDING', 'ACTIVE', 'COMPLETED', 'ARCHIVED'])
                items = {
                    'task1': Status.PENDING,
                    'task2': Status.ACTIVE,
                    'task3': Status.COMPLETED,
                }
                print(f"count={len(items)}")
            """),
            textwrap.dedent("""\
                active = [k for k, v in items.items() if v == Status.ACTIVE]
                print(f"active={active}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(1)
        assert "active=['task2']" in nb_runner.get_output(2)
