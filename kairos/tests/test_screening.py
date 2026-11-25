"""Steps for screening.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.nodes import BayesianFitting, Node, PairScreening

scenarios("screening.feature")


@given(parsers.parse("{count:d} people where y = sin(x1) + 0.5·x2 + {strength:g}·(x1 − 2)·(x2 − 2) plus noise"))
def interaction_people(context, count, strength):
    random = np.random.default_rng(3)
    x1, x2 = random.uniform(0, 4, count), random.uniform(0, 4, count)
    y = np.sin(x1) + 0.5 * x2 + strength * (x1 - 2) * (x2 - 2) + random.normal(0, 0.3, count)
    context["data"] = {"x1": x1, "x2": x2, "y": y}


@given(parsers.parse("{count:d} people where y = flag · 2x plus noise, with a yes/no flag"))
def gate_people(context, count):
    random = np.random.default_rng(0)
    flag, x = (random.uniform(size=count) < 0.4).astype(float), random.uniform(-1, 1, count)
    context["data"] = {"flag": flag, "x": x, "y": 2 * flag * x + random.normal(0, 0.1, count)}


@when("a node for y screens its pairs of parents")
def screen(context):
    parents = [name for name in context["data"] if name != "y"]
    node = Node("y", parents, intervals=12, screening=PairScreening(), kinds={"flag": "binary"})
    node.fit(context["data"])
    context["node"], context["pairs"] = node, node.pairs


@when(parsers.parse("I compute the evidence gain of the surface with ridge {first:g} and with ridge {second:g}"))
def gains_for_two_ridges(context, first, second):
    gains = []
    for ridge in (first, second):
        curves = Node("y", ["x1", "x2"], pairs=[], intervals=12, fitting=BayesianFitting(ridge=ridge))
        surface = Node("y", ["x1", "x2"], pairs=[("x1", "x2")], intervals=12, fitting=BayesianFitting(ridge=ridge))
        curves.fit(context["data"])
        surface.fit(context["data"])
        assert surface.evidence is not None and curves.evidence is not None  # set by fit
        gains.append(surface.evidence - curves.evidence)
    context["gains"] = gains


@then(parsers.re(r"the surface (?P<first>\w+)×(?P<second>\w+) is (?P<decision>kept|rejected)"))
def surface_decision(context, first, second, decision):
    kept = (first, second) in context["pairs"]
    assert kept == (decision == "kept")


@then("with the flag off, x moves y by less than a tenth of what it does with the flag on")
def gate_closes(context):
    def effect(flag: float) -> float:
        low, high = context["node"].predict({"flag": np.full(2, flag), "x": np.array([-0.9, 0.9])})
        return abs(high - low)
    assert effect(0.0) < 0.1 * effect(1.0)


@then(parsers.parse("the two gains differ by less than {tolerance:g}"))
def gains_close(context, tolerance):
    first, second = context["gains"]
    assert abs(first - second) < tolerance
