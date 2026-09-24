"""enum.Enum across cells."""

import textwrap

import pytest


# Enum and constant pattern interaction tests.
#
# Tests editing enum definitions, constant values, and
# patterns that use them across cells.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestEnumEdits:
    """Editing enum definitions."""

    def test_edit_enum_member(self, nb_runner):
        """Edit an enum member value."""
        nb_runner.create_notebook(
            [
                "from enum import Enum",
                "class Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
                "c = Color.RED\nprint(f'color = {c.name}, value = {c.value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color = RED, value = 1" in nb_runner.get_output(3)

        # Change value
        nb_runner.set_cell_source(
            2,
            "class Color(Enum):\n    RED = 10\n    GREEN = 20\n    BLUE = 30",
        )
        nb_runner.set_cell_source(3, "c = Color.RED\nprint(f'color = {c.name}, value = {c.value}')")
        nb_runner.run_all()
        assert "color = RED, value = 10" in nb_runner.get_output(3)

    def test_add_enum_member(self, nb_runner):
        """Add a new member to an enum."""
        nb_runner.create_notebook(
            [
                "from enum import Enum",
                "class Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'",
                "statuses = [s.value for s in Status]\nprint(f'statuses = {statuses}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "active" in out
        assert "inactive" in out

        # Add PENDING
        nb_runner.set_cell_source(
            2,
            "class Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'\n    PENDING = 'pending'",
        )
        nb_runner.set_cell_source(3, "statuses = [s.value for s in Status]\nprint(f'statuses = {statuses}')")
        nb_runner.run_all()
        assert "pending" in nb_runner.get_output(3)


# Enum definition and constant map edit tests.
#
# Tests editing enum definitions, constant maps, and named values
# used in downstream calculations.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestEnumConstEdits:
    """Editing enums and constant mappings."""

    def test_edit_enum_add_member(self, nb_runner):
        """Edit an enum to add a new member."""
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
                "colors = [c.name for c in Color]\nprint(f'colors = {colors}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "RED" in nb_runner.get_output(2)
        assert "BLUE" in nb_runner.get_output(2)

        # Add YELLOW
        nb_runner.set_cell_source(
            1, "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    YELLOW = 4"
        )
        nb_runner.run_all()
        assert "YELLOW" in nb_runner.get_output(2)

    def test_edit_status_code_lookup(self, nb_runner):
        """Edit a constant mapping lookup."""
        nb_runner.create_notebook(
            [
                "STATUS_CODES = {200: 'OK', 404: 'Not Found', 500: 'Server Error'}",
                "msg = STATUS_CODES.get(200, 'Unknown')\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = OK" in nb_runner.get_output(2)

        # Change lookup key
        nb_runner.set_cell_source(2, "msg = STATUS_CODES.get(404, 'Unknown')\nprint(f'msg = {msg}')")
        nb_runner.run_all()
        assert "msg = Not Found" in nb_runner.get_output(2)

    def test_edit_tax_rate_constant(self, nb_runner):
        """Edit a named constant used in calculations."""
        nb_runner.create_notebook(
            [
                "TAX_RATE = 0.08",
                "price = 100\ntax = price * TAX_RATE\ntotal = price + tax\nprint(f'total = {total:.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 108.00" in nb_runner.get_output(2)

        # Change tax rate
        nb_runner.set_cell_source(1, "TAX_RATE = 0.10")
        nb_runner.run_all()
        assert "total = 110.00" in nb_runner.get_output(2)


# enum patterns with caching.
# Tests Enum, IntEnum, Flag operations, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestEnumPatterns:
    """Test Enum operation caching."""

    def test_enum_basic(self, nb_runner):
        """Basic Enum creation and comparison with caching."""
        nb_runner.create_notebook(
            [
                "from enum import Enum",
                "class Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
                "c = Color.GREEN\nresult = c.name",
                "print(f'result={result} value={c.value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=GREEN" in out
        assert "value=2" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=GREEN" in out2

    def test_enum_edit_selection(self, nb_runner):
        """Edit enum member selection, verify propagation."""
        nb_runner.create_notebook(
            [
                "from enum import Enum",
                "class Status(Enum):\n    PENDING = 'pending'\n    ACTIVE = 'active'\n    CLOSED = 'closed'",
                "current = Status.PENDING",
                "is_open = current != Status.CLOSED\nprint(f'status={current.value} is_open={is_open}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "status=pending" in out
        assert "is_open=True" in out

        nb_runner.set_cell_source(3, "current = Status.CLOSED")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "status=closed" in out2
        assert "is_open=False" in out2

    def test_int_enum_arithmetic(self, nb_runner):
        """IntEnum used in arithmetic."""
        nb_runner.create_notebook(
            [
                "from enum import IntEnum",
                "class Priority(IntEnum):\n    LOW = 1\n    MEDIUM = 2\n    HIGH = 3\n    CRITICAL = 4",
                "tasks = [Priority.HIGH, Priority.LOW, Priority.MEDIUM]\navg = sum(tasks) / len(tasks)",
                "print(f'avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "avg=2.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "avg=2.0" in out2


# Collections patterns, enum usage, protocol/structural typing,
# __slots__, and complex comprehension patterns.
#
# Tests how cash handles specialized collection types, enums across cells,
# Protocol-based structural subtyping, __slots__ classes, and deeply nested
# comprehensions.
@pytest.mark.stress
@pytest.mark.integration
class TestEnumChangeInvalidation:
    """Test enum usage across cells."""

    def test_enum_change_invalidation(self, nb_runner):
        """Changing enum definition should invalidate downstream."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from enum import Enum
                class Status(Enum):
                    ACTIVE = 'active'
                    INACTIVE = 'inactive'
            """),
                textwrap.dedent("""\
                s = Status.ACTIVE
                print(s.value)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "active" in nb_runner.get_output(2)

        # Change enum
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from enum import Enum
            class Status(Enum):
                ACTIVE = 'enabled'
                INACTIVE = 'disabled'
        """),
        )
        nb_runner.run_all()
        assert "enabled" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestEnumClassUsage:
    """enum class definitions and usage."""

    def test_enum_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
                "c = Color.RED\nname = c.name\nval = c.value\nprint(f'name={name} val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=RED" in nb_runner.get_output(2)
        assert "val=1" in nb_runner.get_output(2)

    def test_enum_iteration(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Status(Enum):\n    OPEN = 'open'\n    CLOSED = 'closed'\n    PENDING = 'pending'",
                "names = [s.name for s in Status]\nvalues = [s.value for s in Status]\nprint(f'names={names} values={values}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "OPEN" in out
        assert "'open'" in out

    def test_enum_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Dir(Enum):\n    NORTH = 0\n    SOUTH = 1",
                "count = len(Dir)\nfirst = Dir(0).name\nprint(f'count={count} first={first}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1, "from enum import Enum\nclass Dir(Enum):\n    NORTH = 0\n    SOUTH = 1\n    EAST = 2\n    WEST = 3"
        )
        nb_runner.run_all()
        assert "count=4" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestEnumAutoCustom:
    """enum auto and custom value methods."""

    def test_auto_values(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum, auto",
                "class Color(Enum):\n    RED = auto()\n    GREEN = auto()\n    BLUE = auto()\nprint(f'red={Color.RED.value} green={Color.GREEN.value} blue={Color.BLUE.value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "red=1" in out
        assert "green=2" in out
        assert "blue=3" in out

    def test_custom_enum_method(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum",
                "class Planet(Enum):\n    EARTH = (5.97e24, 6.37e6)\n    MARS = (6.42e23, 3.39e6)\n    def __init__(self, mass, radius):\n        self.mass = mass\n        self.radius = radius\n    @property\n    def surface_gravity(self):\n        G = 6.674e-11\n        return G * self.mass / self.radius**2\nearth_g = round(Planet.EARTH.surface_gravity, 2)\nmars_g = round(Planet.MARS.surface_gravity, 2)\nprint(f'earth={earth_g} mars={mars_g}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "earth=9.82" in out
        assert "mars=3.73" in out

    def test_enum_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum, auto",
                "class Dir(Enum):\n    N = auto()\n    S = auto()\nprint(f'count={len(Dir)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class Dir(Enum):\n    N = auto()\n    S = auto()\n    E = auto()\n    W = auto()\nprint(f'count={len(Dir)}')",
        )
        nb_runner.run_all()
        assert "count=4" in nb_runner.get_output(2)


@pytest.mark.stress
class TestEnumAutoAndFunctional:
    """Test auto() and functional Enum creation."""

    def test_auto_enum(self, nb_runner):
        """Enum with auto() values across cells."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "CRITICAL: Fix bug" in out
        # CRITICAL (4) should come first
        assert out.index("CRITICAL") < out.index("LOW")


# Interaction test: enum with custom methods and class attributes.
# Tests Enum with custom methods, classmethods, properties,
# and cross-cell enum-based dispatch.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestEnumCustomMethods:
    """Test Enum with custom methods across cells."""

    def test_enum_methods(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define enum with methods
                "from enum import Enum\nclass HttpStatus(Enum):\n    OK = 200\n    NOT_FOUND = 404\n    SERVER_ERROR = 500\n    @property\n    def is_error(self):\n        return self.value >= 400\n    @classmethod\n    def from_code(cls, code):\n        for member in cls:\n            if member.value == code:\n                return member\n        return None\n    def describe(self):\n        descriptions = {200: 'Success', 404: 'Not Found', 500: 'Internal Error'}\n        return descriptions.get(self.value, 'Unknown')\nprint('HttpStatus defined')",
                # Cell 2: use enum methods
                "s1 = HttpStatus.OK\ns2 = HttpStatus.from_code(404)\nprint(f's1_err={s1.is_error}')\nprint(f's2_err={s2.is_error}')\nprint(f's1_desc={s1.describe()}')\nprint(f's2_desc={s2.describe()}')",
                # Cell 3: iterate and filter
                "errors = [s for s in HttpStatus if s.is_error]\nprint(f'errors={[e.name for e in errors]}')\ncodes = [s.value for s in HttpStatus]\nprint(f'codes={codes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "s1_err=False" in out2
        assert "s2_err=True" in out2
        assert "s1_desc=Success" in out2
        assert "s2_desc=Not Found" in out2
        out3 = nb_runner.get_output(3)
        assert "errors=['NOT_FOUND', 'SERVER_ERROR']" in out3
        assert "codes=[200, 404, 500]" in out3

    def test_enum_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Color(Enum):\n    RED = 'red'\n    GREEN = 'green'\n    BLUE = 'blue'\n    def hex_code(self):\n        codes = {'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF'}\n        return codes[self.value]\nprint('Color defined')",
                "colors = [Color.RED, Color.BLUE]\nhexes = [c.hex_code() for c in colors]\nprint(f'hexes={hexes}')",
                "all_hexes = [c.hex_code() for c in Color]\nprint(f'all={all_hexes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hexes=['#FF0000', '#0000FF']" in nb_runner.get_output(2)

        # Edit selection
        nb_runner.set_cell_source(
            2, "colors = [Color.GREEN, Color.RED]\nhexes = [c.hex_code() for c in colors]\nprint(f'hexes={hexes}')"
        )
        nb_runner.run_cells([2, 3])
        assert "hexes=['#00FF00', '#FF0000']" in nb_runner.get_output(2)

    def test_enum_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Direction(Enum):\n    NORTH = (0, 1)\n    SOUTH = (0, -1)\n    EAST = (1, 0)\n    WEST = (-1, 0)\n    @property\n    def dx(self): return self.value[0]\n    @property\n    def dy(self): return self.value[1]\nprint('Direction defined')",
                "path = [Direction.NORTH, Direction.NORTH, Direction.EAST, Direction.EAST, Direction.SOUTH]\nfinal_x = sum(d.dx for d in path)\nfinal_y = sum(d.dy for d in path)\nprint(f'pos=({final_x},{final_y})')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=(2,1)" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "pos=(2,1)" in nb_runner.get_output(2)


# Advanced enum patterns — cash caching with Enum, Flag, IntEnum.
@pytest.mark.stress
class TestEnumAdvanced:
    """Test advanced enum patterns across cells."""

    def test_intflag_bitwise(self, nb_runner):
        """IntFlag with bitwise operations across cells."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "user=3 admin=7" in nb_runner.get_output(1)
        assert "exec=False" in nb_runner.get_output(2)
        assert "read=True" in nb_runner.get_output(2)

    def test_enum_with_methods(self, nb_runner):
        """Enum with custom methods across cells."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "current=AUTUMN next=WINTER" in nb_runner.get_output(1)
        assert "warm=['SPRING', 'SUMMER']" in nb_runner.get_output(2)

    def test_enum_change_propagation(self, nb_runner):
        """Enum value change propagation."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color=RED hex=#FF0000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from enum import Enum

            class Color(Enum):
                RED = '#FF0000'
                GREEN = '#00FF00'
                BLUE = '#0000FF'

            selected = Color.BLUE
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "color=BLUE hex=#0000FF" in nb_runner.get_output(2)


# Enum member editing and dispatch patterns.
#
# Tests enum-based dispatch/mapping with edits.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestEnumDispatchEdits:
    """Enum-based dispatch patterns with edits."""

    def test_enum_dispatch_edit(self, nb_runner):
        """Edit enum dispatch mapping."""
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
                "dispatch = {Color.RED: 'stop', Color.GREEN: 'go', Color.BLUE: 'info'}",
                "result = dispatch[Color.RED]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = stop" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "dispatch = {Color.RED: 'danger', Color.GREEN: 'safe', Color.BLUE: 'neutral'}",
        )
        nb_runner.run_all()
        assert "result = danger" in nb_runner.get_output(3)

    def test_enum_value_function(self, nb_runner):
        """Edit function using enum values."""
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Priority(Enum):\n    LOW = 1\n    MED = 5\n    HIGH = 10",
                "def score(p):\n    return p.value * 2",
                "s = score(Priority.HIGH)\nprint(f's = {s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "def score(p):\n    return p.value ** 2",
        )
        nb_runner.run_all()
        assert "s = 100" in nb_runner.get_output(3)

    def test_enum_class_edit(self, nb_runner):
        """Edit enum class itself, downstream dispatch updates."""
        nb_runner.create_notebook(
            [
                "from enum import Enum\nclass Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'",
                "labels = {s: s.value.upper() for s in Status}\nprint(f'labels = {labels}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "ACTIVE" in out
        assert "INACTIVE" in out

        nb_runner.set_cell_source(
            1,
            "from enum import Enum\nclass Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'\n    PENDING = 'pending'",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "PENDING" in out2
