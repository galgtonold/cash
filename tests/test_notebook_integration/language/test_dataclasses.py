"""dataclasses across cells: defaults, factories, ordering, frozen and slots."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassFieldDefaults:
    """dataclass with field defaults and post_init."""

    def test_dataclass_defaults(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str\n    debug: bool = False\n    tags: list = field(default_factory=list)",
                "c = Config('test')\nc.tags.append('v1')\nprint(f'name={c.name} debug={c.debug} tags={c.tags}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=test" in out
        assert "debug=False" in out
        assert "tags=['v1']" in out

    def test_dataclass_post_init(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Rect:\n    width: float\n    height: float\n    area: float = 0\n    def __post_init__(self):\n        self.area = self.width * self.height",
                "r = Rect(3.0, 4.0)\nprint(f'area={r.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12.0" in nb_runner.get_output(2)

    def test_dataclass_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: int\n    y: int",
                "p = Point(1, 2)\ndist_sq = p.x**2 + p.y**2\nprint(f'dist_sq={dist_sq}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dist_sq=5" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "p = Point(3, 4)\ndist_sq = p.x**2 + p.y**2\nprint(f'dist_sq={dist_sq}')")
        nb_runner.run_all()
        assert "dist_sq=25" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassFieldEdits:
    """dataclass field defaults, post_init, and frozen edits."""

    def test_dataclass_post_init(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Rectangle:\n    width: float\n    height: float\n    area: float = field(init=False)\n    def __post_init__(self):\n        self.area = self.width * self.height",
                "r = Rectangle(3.0, 4.0)\nprint(f'area={r.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12.0" in nb_runner.get_output(2)

    def test_dataclass_frozen_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int",
                "p = Point(1, 2)\nresult = f'{p.x},{p.y}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=1,2" in nb_runner.get_output(2)
        # Edit class
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int\n    z: int = 0",
        )
        nb_runner.set_cell_source(2, "p = Point(10, 20, 30)\nresult = f'{p.x},{p.y},{p.z}'\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=10,20,30" in nb_runner.get_output(2)

    def test_dataclass_default_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'default'\n    tags: list = field(default_factory=list)",
                "c1 = Config('test', ['a', 'b'])\nc2 = Config()\nprint(f'c1={c1.name},{c1.tags}')\nprint(f'c2={c2.name},{c2.tags}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c1=test,['a', 'b']" in out
        assert "c2=default,[]" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassDefaultFactory:
    """Test dataclass default_factory and field metadata across cells."""

    def test_dataclass_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: dataclass with default_factory
                "from dataclasses import dataclass, field\n@dataclass\nclass ShoppingCart:\n    owner: str\n    items: list = field(default_factory=list)\n    discounts: dict = field(default_factory=dict)\n\ncart = ShoppingCart('Alice')\ncart.items.append('apple')\ncart.items.append('banana')\ncart.discounts['apple'] = 0.1\nprint(f'owner={cart.owner}')\nprint(f'items={cart.items}')",
                # Cell 2: use cart state
                "total_items = len(cart.items)\nhas_discount = len(cart.discounts) > 0\nprint(f'total_items={total_items}')\nprint(f'has_discount={has_discount}')",
                # Cell 3: create second cart (no sharing)
                "cart2 = ShoppingCart('Bob')\ncart2.items.append('orange')\nprint(f'cart1_items={len(cart.items)}')\nprint(f'cart2_items={len(cart2.items)}')\nprint(f'independent={cart.items is not cart2.items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "owner=Alice" in out1
        assert "items=['apple', 'banana']" in out1
        out2 = nb_runner.get_output(2)
        assert "total_items=2" in out2
        assert "has_discount=True" in out2
        out3 = nb_runner.get_output(3)
        assert "cart1_items=2" in out3
        assert "cart2_items=1" in out3
        assert "independent=True" in out3

    def test_dataclass_factory_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str\n    tags: list = field(default_factory=lambda: ['default'])\n\nc = Config('test')\nprint(f'tags={c.tags}')",
                "tag_str = ','.join(c.tags)\nprint(f'tag_str={tag_str}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "tag_str=default" in nb_runner.get_output(2)

        # Edit to add more default tags
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str\n    tags: list = field(default_factory=lambda: ['default', 'v2'])\n\nc = Config('test')\nprint(f'tags={c.tags}')",
        )
        nb_runner.run_cells([1, 2])
        assert "tag_str=default,v2" in nb_runner.get_output(2)

    def test_dataclass_factory_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Stats:\n    values: list = field(default_factory=lambda: [10, 20, 30])\n\ns = Stats()\navg = sum(s.values) / len(s.values)\nprint(f'avg={avg:.1f}')",
                "spread = max(s.values) - min(s.values)\nprint(f'spread={spread}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=20.0" in nb_runner.get_output(1)
        assert "spread=20" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "spread=20" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassPostInit:
    """dataclass post init and field defaults."""

    def test_field_default_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field",
                "@dataclass\nclass Config:\n    name: str = 'default'\n    tags: list = field(default_factory=list)\nc1 = Config()\nc2 = Config('custom', ['a', 'b'])\nprint(f'c1={c1.name}:{c1.tags} c2={c2.name}:{c2.tags}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c1=default:[]" in out
        assert "c2=custom:['a', 'b']" in out

    def test_dataclass_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field",
                "@dataclass\nclass Point:\n    x: float = 0\n    y: float = 0\np = Point(1, 2)\nprint(f'p={p.x},{p.y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p=1,2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Point:\n    x: float = 0\n    y: float = 0\np = Point(100, 200)\nprint(f'p={p.x},{p.y}')",
        )
        nb_runner.run_all()
        assert "p=100,200" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassPostInitFactory:
    """Test dataclass __post_init__ and default_factory across cells."""

    def test_dataclass_post_init_validation(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define dataclass with post_init
                "from dataclasses import dataclass, field\n@dataclass\nclass Product:\n    name: str\n    price: float\n    tags: list = field(default_factory=list)\n    discount_price: float = field(init=False)\n    def __post_init__(self):\n        self.discount_price = round(self.price * 0.9, 2)\n        self.name = self.name.strip().title()\nprint('Product defined')",
                # Cell 2: create instances
                "p1 = Product('  laptop  ', 999.99)\np2 = Product('mouse pad', 19.50, ['accessories', 'office'])\nprint(f'p1_name={p1.name}')\nprint(f'p1_disc={p1.discount_price}')\nprint(f'p2_tags={p2.tags}')",
                # Cell 3: aggregate
                "products = [p1, p2]\ntotal = sum(p.discount_price for p in products)\navg = round(total / len(products), 2)\nprint(f'total={total}')\nprint(f'avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "p1_name=Laptop" in out2
        assert "p1_disc=899.99" in out2
        assert "p2_tags=['accessories', 'office']" in out2
        out3 = nb_runner.get_output(3)
        assert "total=" in out3
        assert "avg=" in out3

    def test_dataclass_edit_price(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Product:\n    name: str\n    price: float\n    discount_price: float = field(init=False)\n    def __post_init__(self):\n        self.discount_price = round(self.price * 0.9, 2)\nprint('defined')",
                "p = Product('widget', 100.0)\nprint(f'disc={p.discount_price}')",
                "savings = round(p.price - p.discount_price, 2)\nprint(f'savings={savings}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "disc=90.0" in nb_runner.get_output(2)
        assert "savings=10.0" in nb_runner.get_output(3)

        # Change discount rate
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass, field\n@dataclass\nclass Product:\n    name: str\n    price: float\n    discount_price: float = field(init=False)\n    def __post_init__(self):\n        self.discount_price = round(self.price * 0.8, 2)\nprint('defined')",
        )
        nb_runner.run_cells([1, 2, 3])
        assert "disc=80.0" in nb_runner.get_output(2)
        assert "savings=20.0" in nb_runner.get_output(3)

    def test_dataclass_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Item:\n    name: str\n    qty: int = 1\n    history: list = field(default_factory=list)\nprint('defined')",
                "item = Item('bolt', 50, ['warehouse'])\nprint(f'name={item.name} qty={item.qty} hist={item.history}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=bolt qty=50 hist=['warehouse']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "name=bolt qty=50 hist=['warehouse']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassOrdering:
    """dataclass ordering and comparison methods."""

    def test_order_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(order=True)\nclass Priority:\n    level: int\n    name: str",
                "items = [Priority(3, 'low'), Priority(1, 'high'), Priority(2, 'mid')]\nsorted_items = sorted(items)\nresult = [(p.level, p.name) for p in sorted_items]\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'high'), (2, 'mid'), (3, 'low')]" in nb_runner.get_output(2)

    def test_order_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass(order=True)\nclass Student:\n    gpa: float\n    name: str = field(compare=False)",
                "students = [Student(3.5, 'A'), Student(3.9, 'B'), Student(3.2, 'C')]\nbest = max(students)\nprint(f'best={best.name}:{best.gpa}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=B:3.9" in nb_runner.get_output(2)
        # Edit students
        nb_runner.set_cell_source(
            2,
            "students = [Student(4.0, 'X'), Student(3.8, 'Y')]\nbest = max(students)\nprint(f'best={best.name}:{best.gpa}')",
        )
        nb_runner.run_all()
        assert "best=X:4.0" in nb_runner.get_output(2)

    def test_order_reverse(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(order=True)\nclass Score:\n    value: int",
                "scores = [Score(80), Score(95), Score(70), Score(88)]\ntop2 = sorted(scores, reverse=True)[:2]\nresult = [s.value for s in top2]\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[95, 88]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassFieldOrdering:
    """Test dataclass field-based ordering across cells."""

    def test_ordered_dataclass(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create ordered dataclass
                "from dataclasses import dataclass, field\n@dataclass(order=True)\nclass Student:\n    gpa: float\n    name: str = field(compare=False)\n\ns1 = Student(3.5, 'Alice')\ns2 = Student(3.9, 'Bob')\ns3 = Student(3.2, 'Charlie')\nstudents = [s1, s2, s3]\nsorted_names = [s.name for s in sorted(students)]\nprint(f'sorted={sorted_names}')",
                # Cell 2: comparisons
                "is_less = s1 < s2\nis_greater = s1 > s3\nprint(f'alice_lt_bob={is_less}')\nprint(f'alice_gt_charlie={is_greater}')",
                # Cell 3: min/max
                "best = max(students)\nworst = min(students)\nprint(f'best={best.name}')\nprint(f'worst={worst.name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "sorted=['Charlie', 'Alice', 'Bob']" in out1
        out2 = nb_runner.get_output(2)
        assert "alice_lt_bob=True" in out2
        assert "alice_gt_charlie=True" in out2
        out3 = nb_runner.get_output(3)
        assert "best=Bob" in out3
        assert "worst=Charlie" in out3

    def test_ordered_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(order=True)\nclass Score:\n    value: int\n\nscores = [Score(80), Score(95), Score(70)]\nbest = max(scores)\nprint(f'best={best.value}')",
                "spread = max(scores).value - min(scores).value\nprint(f'spread={spread}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=95" in nb_runner.get_output(1)
        assert "spread=25" in nb_runner.get_output(2)

        # Edit scores
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass(order=True)\nclass Score:\n    value: int\n\nscores = [Score(50), Score(100), Score(75)]\nbest = max(scores)\nprint(f'best={best.value}')",
        )
        nb_runner.run_cells([1, 2])
        assert "best=100" in nb_runner.get_output(1)
        assert "spread=50" in nb_runner.get_output(2)

    def test_ordered_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(order=True)\nclass Temp:\n    celsius: float\n\ntemps = [Temp(20.0), Temp(35.5), Temp(10.2)]\nhot = max(temps)\nprint(f'hot={hot.celsius}')",
                "is_hot = hot.celsius > 30\nprint(f'is_hot={is_hot}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hot=35.5" in nb_runner.get_output(1)
        assert "is_hot=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_hot=True" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassEqOrderHash:
    """Test dataclass eq/order/hash across cells."""

    def test_dataclass_ordering(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define ordered dataclass
                "from dataclasses import dataclass\n@dataclass(order=True, frozen=True)\nclass Priority:\n    level: int\n    name: str\nprint('Priority defined')",
                # Cell 2: create and sort
                "tasks = [\n    Priority(3, 'low'),\n    Priority(1, 'critical'),\n    Priority(2, 'medium'),\n    Priority(1, 'urgent'),\n]\nsorted_tasks = sorted(tasks)\nprint(f'sorted={sorted_tasks}')",
                # Cell 3: set operations (frozen=True makes hashable)
                "unique = set(tasks)\nprint(f'unique_count={len(unique)}')\nmin_task = min(tasks)\nmax_task = max(tasks)\nprint(f'min={min_task}')\nprint(f'max={max_task}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        # Sorted by (level, name)
        assert "Priority(level=1, name='critical')" in out2
        out3 = nb_runner.get_output(3)
        assert "unique_count=4" in out3
        assert "min=Priority(level=1, name='critical')" in out3
        assert "max=Priority(level=3, name='low')" in out3

    def test_dataclass_order_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(order=True, frozen=True)\nclass Score:\n    value: int\n    name: str\nprint('Score defined')",
                "scores = [Score(90, 'A'), Score(80, 'B'), Score(95, 'A+')]\nbest = max(scores)\nprint(f'best={best}')",
                "ranking = [s.name for s in sorted(scores, reverse=True)]\nprint(f'ranking={ranking}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=Score(value=95, name='A+')" in nb_runner.get_output(2)
        assert "ranking=['A+', 'A', 'B']" in nb_runner.get_output(3)

        # Add more scores
        nb_runner.set_cell_source(
            2,
            "scores = [Score(90, 'A'), Score(80, 'B'), Score(95, 'A+'), Score(100, 'S')]\nbest = max(scores)\nprint(f'best={best}')",
        )
        nb_runner.run_cells([2, 3])
        assert "best=Score(value=100, name='S')" in nb_runner.get_output(2)
        assert "ranking=['S', 'A+', 'A', 'B']" in nb_runner.get_output(3)

    def test_dataclass_order_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Coord:\n    x: int\n    y: int\nprint('Coord defined')",
                "points = {Coord(1, 2), Coord(3, 4), Coord(1, 2)}\nprint(f'unique={len(points)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "unique=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassInheritanceFrozen:
    """dataclass inheritance and frozen."""

    def test_dataclass_inheritance(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Animal:\n    name: str\n    sound: str\n@dataclass\nclass Pet(Animal):\n    owner: str\np = Pet('Rex', 'Woof', 'Alice')\nprint(f'name={p.name} sound={p.sound} owner={p.owner}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=Rex" in out
        assert "sound=Woof" in out
        assert "owner=Alice" in out

    def test_frozen_dataclass(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\np = Point(1.0, 2.0)\ntry:\n    p.x = 5.0\n    msg = 'mutable'\nexcept AttributeError:\n    msg = 'frozen'\nprint(f'msg={msg} x={p.x} y={p.y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "msg=frozen" in out  # FrozenInstanceError is subclass of AttributeError
        assert "x=1.0" in out

    def test_dataclass_inh_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Base:\n    val: int = 10\nclass Child(Base):\n    pass\nc = Child()\nprint(f'val={c.val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Base:\n    val: int = 99\nclass Child(Base):\n    pass\nc = Child()\nprint(f'val={c.val}')",
        )
        nb_runner.run_all()
        assert "val=99" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestDataclassSlotsFrozen:
    """Test dataclass with slots and frozen across cells."""

    def test_dataclass_frozen(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: frozen dataclass
                "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\n\np1 = Point(1.0, 2.0)\np2 = Point(3.0, 4.0)\nprint(f'p1={p1}')\nprint(f'p2={p2}')\nprint(f'hash_p1={hash(p1) == hash(Point(1.0, 2.0))}')",
                # Cell 2: use as dict key (hashable)
                "distances = {p1: 'near', p2: 'far'}\nprint(f'p1_dist={distances[p1]}')\nprint(f'p2_dist={distances[p2]}')",
                # Cell 3: verify immutability
                "try:\n    p1.x = 99\n    print('error=none')\nexcept Exception as e:\n    print(f'error={type(e).__name__}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "p1=Point(x=1.0, y=2.0)" in out1
        assert "hash_p1=True" in out1
        out2 = nb_runner.get_output(2)
        assert "p1_dist=near" in out2
        assert "p2_dist=far" in out2
        out3 = nb_runner.get_output(3)
        assert "FrozenInstanceError" in out3

    def test_dataclass_frozen_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Config:\n    host: str\n    port: int\n\ncfg = Config('localhost', 8080)\nprint(f'cfg={cfg}')",
                "addr = f'{cfg.host}:{cfg.port}'\nprint(f'addr={addr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr=localhost:8080" in nb_runner.get_output(2)

        # Edit config
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Config:\n    host: str\n    port: int\n\ncfg = Config('0.0.0.0', 9090)\nprint(f'cfg={cfg}')",
        )
        nb_runner.run_cells([1, 2])
        assert "addr=0.0.0.0:9090" in nb_runner.get_output(2)

    def test_dataclass_frozen_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Vec2:\n    x: int\n    y: int\n\nv = Vec2(3, 4)\nmag_sq = v.x**2 + v.y**2\nprint(f'mag_sq={mag_sq}')",
                "is_unit = mag_sq == 1\nprint(f'is_unit={is_unit}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag_sq=25" in nb_runner.get_output(1)
        assert "is_unit=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_unit=False" in nb_runner.get_output(2)


@pytest.mark.stress
class TestDataclassAdvanced:
    """Test advanced dataclass patterns."""

    def test_dataclass_change_propagates(self, nb_runner):
        """Changing dataclass definition propagates."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name='test'" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from dataclasses import dataclass

            @dataclass
            class Config:
                name: str
                value: int = 0
                active: bool = True
        """),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name='test'" in out
        assert "active=True" in out
