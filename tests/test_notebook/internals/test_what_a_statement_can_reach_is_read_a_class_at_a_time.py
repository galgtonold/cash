"""What a statement can run through the notebook's own objects is read a
class at a time, and still found wherever it is held.

``market.step()`` calls ``a.act(price)`` on each of a simulation's agents,
and ``act`` calls the strategy function each agent holds. Each agent's
``act`` was read by itself, as if each agent's class could differ: 20,000
agents, 20,000 readings of one method, each opening what the agent holds.
The method is now read once for all objects of a class, as the class holds
it when asked, and only what each object holds itself (its own ``act``, its
strategy) is looked at one object at a time. A function held anywhere in
the objects -- an attribute, a list, a dict, a closure, a partial, a bound
method of another object, one object's own replacement of a method, a
method patched on the class after the objects were made -- is still found.
"""

from __future__ import annotations

import ast
import gc
import linecache

import pytest

import cash.analysis.handed_callables as hc
from cash.analysis.handed_callables import handed_callable_values, handed_callables, new_analysis_round
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

N_AGENTS = 300

SETUP = f"""
import functools

def cautious(agent, price):
    return 1 if price < agent.belief else -1

class Agent:
    def __init__(self, aid, strategy):
        self.aid = aid
        self.belief = 10.0
        self.strategy = strategy
        self.rules = []
        self.neighbours = []
    def act(self, price):
        return self.strategy(self, price)
    def vote(self, price):
        return sum(rule(price) for rule in self.rules)

class Market:
    def __init__(self, n):
        self.price = 10.0
        self.agents = [Agent(i, cautious) for i in range(n)]
        for i, a in enumerate(self.agents):
            a.neighbours = [self.agents[(i + 1) % n]]
    def step(self):
        demand = 0
        for a in self.agents:
            demand += a.act(self.price)
        self.price += demand / len(self.agents)
    def poll(self):
        return [a.vote(self.price) for a in self.agents]

def run(market):
    market.step()

def poll(market):
    return market.poll()

class Advisor:
    def decide(self, agent, price):
        return special(agent, price)

def special(agent, price):
    return 0

def special_act(self, price):
    return 0

def make(f):
    return lambda agent, price: f(agent, price)

market = Market({N_AGENTS})
"""


def _namespace() -> dict:
    filename = f"<market-{abs(hash(SETUP))}>"
    linecache.cache[filename] = (len(SETUP), None, SETUP.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(SETUP, filename, "exec"), namespace)
    return namespace


#: Where the special function is put, as a statement run on the namespace,
#: and the name of the function whose source must be found.
PLACES = {
    "an agent's attribute": ("market.agents[-1].strategy = special", "special"),
    "an agent's own act": ("market.agents[-1].act = special", "special"),
    "the class's act, patched": ("Agent.act = special_act", "special_act"),
    "a closure an agent holds": ("market.agents[-1].strategy = make(special)", "special"),
    "a partial an agent holds": ("market.agents[-1].strategy = functools.partial(special)", "special"),
    "a bound method an agent holds": ("market.agents[-1].strategy = Advisor().decide", "decide"),
    "a neighbour's attribute": ("market.agents[0].neighbours[0].strategy = special", "special"),
}
#: Held in a container an agent holds: ``vote`` calls each of its rules.
HELD = {
    "a list an agent holds": "market.agents[-1].rules.append(special)",
    "a dict in a list an agent holds": "market.agents[-1].rules.append({'f': special})",
    "a neighbour's list": "market.agents[0].neighbours[0].rules.append(special)",
}


def _sources(namespace: dict, code: str) -> list[str]:
    new_analysis_round()
    return list(handed_callables(ast.parse(code), namespace).sources)


def _values(namespace: dict, code: str) -> list:
    new_analysis_round()
    call = ast.parse(code).body[0].value
    return handed_callable_values(call, namespace)


def _names(values: list) -> set[str]:
    out = set()
    for value in values:
        for part in (value, getattr(value, "__func__", None), getattr(value, "func", None)):
            if part is not None and hasattr(part, "__code__"):
                out.add(part.__code__.co_name)
    return out


@pytest.mark.parametrize("place", list(PLACES))
def test_a_function_held_by_one_agent_is_found_by_the_call_handed_the_market(place):
    statement, name = PLACES[place]
    namespace = _namespace()
    exec(statement, namespace)
    assert any(f"def {name}" in source for source in _sources(namespace, "out = run(market)")), place


@pytest.mark.parametrize("place", list(PLACES))
def test_a_function_held_by_one_agent_is_found_for_the_markets_own_step(place):
    statement, name = PLACES[place]
    namespace = _namespace()
    exec(statement, namespace)
    values = _values(namespace, "market.step()")
    if place in ("a closure an agent holds", "a bound method an agent holds"):
        # The closure, or the bound method, is what is found: what it calls
        # is followed from its body by the analyses that read it.
        assert any(
            getattr(v, "__closure__", None) or getattr(v, "__self__", None)
            for v in values
            if _names([v]) & {"<lambda>", "decide"}
        ), place
    else:
        assert name in _names(values), place


@pytest.mark.parametrize("place", list(HELD))
def test_a_function_in_a_container_an_agent_holds_is_found(place):
    namespace = _namespace()
    exec(HELD[place], namespace)
    assert any("def special" in source for source in _sources(namespace, "out = poll(market)")), place
    assert "special" in _names(_values(namespace, "market.poll()")), place


def test_each_agent_is_not_read_by_itself(monkeypatch):
    """The work is counted in readings of a method: once for the class's
    ``act``, not once for each of the agents."""
    namespace = _namespace()
    reads = [0]
    original = hc._call_facts

    def counting(fn):
        reads[0] += 1
        return original(fn)

    monkeypatch.setattr(hc, "_call_facts", counting)
    values = _values(namespace, "market.step()")
    assert "cautious" in _names(values)
    # Before: one reading of ``act`` for each agent (300 and more).
    assert reads[0] < 20, reads[0]


def test_frozen_objects_are_still_walked():
    """The objects ``gc.freeze`` moved out of the collector's sight are still
    looked through: what is found is read from the objects, never from the
    collector's view of who holds a function."""
    namespace = _namespace()
    exec("market.agents[-1].strategy = special", namespace)
    gc.freeze()
    try:
        assert gc.get_freeze_count()
        assert any("def special" in source for source in _sources(namespace, "out = run(market)"))
        assert "special" in _names(_values(namespace, "market.step()"))
    finally:
        gc.unfreeze()


SIM = f"""
import functools
import os
import time

BIAS = 0

def cautious(price, belief):
    return 1 if price + BIAS < belief else -1

def greedy(price, belief):
    return 1

TILT = 1

def tilted(sign, price, belief):
    return sign * TILT

class Agent:
    def __init__(self, strategy):
        self.belief = 10.0
        self.strategy = strategy
    def act(self, price):
        decide = self.strategy
        return decide(price, self.belief)

class Market:
    def __init__(self, n):
        self.price = 10.0
        self.agents = [Agent(cautious) for _ in range(n)]
    def demand(self):
        return sum(a.act(self.price) for a in self.agents)

def measure(market):
    fd = os.open(RUNS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return market.demand()

market = Market({N_AGENTS})
"""


def _runs(path) -> int:
    return len(path.read_bytes()) if path.exists() else 0


def test_the_agents_strategies_key_their_readers(cash_magics, mock_shell, tmp_path):
    mock_shell.user_ns["__name__"] = "__main__"
    runs = tmp_path / "runs.log"
    run_cash_cell(cash_magics, f"RUNS = {str(runs)!r}\n" + SIM)
    run_cash_cell(cash_magics, "out = measure(market)")
    assert mock_shell.user_ns["out"] == -N_AGENTS
    assert _runs(runs) == 1
    run_cash_cell(cash_magics, "out = measure(market)")
    assert _runs(runs) == 1  # nothing changed: served
    # A global only the strategy reads.
    run_cash_cell(cash_magics, "BIAS = -5")
    run_cash_cell(cash_magics, "out = measure(market)")
    assert mock_shell.user_ns["out"] == N_AGENTS
    assert _runs(runs) == 2
    # One agent's strategy replaced.
    run_cash_cell(cash_magics, "market.agents[7].strategy = greedy")
    run_cash_cell(cash_magics, "BIAS = 0")
    run_cash_cell(cash_magics, "out = measure(market)")
    assert mock_shell.user_ns["out"] == -N_AGENTS + 2
    assert _runs(runs) == 3
    # A partial one agent holds, of a function reading a global.
    run_cash_cell(cash_magics, "market.agents[8].strategy = functools.partial(tilted, 1)")
    run_cash_cell(cash_magics, "out = measure(market)")
    assert mock_shell.user_ns["out"] == -N_AGENTS + 4
    assert _runs(runs) == 4
    run_cash_cell(cash_magics, "out = measure(market)")
    assert _runs(runs) == 4
    run_cash_cell(cash_magics, "TILT = 3")
    run_cash_cell(cash_magics, "out = measure(market)")
    assert mock_shell.user_ns["out"] == -N_AGENTS + 6
    assert _runs(runs) == 5
