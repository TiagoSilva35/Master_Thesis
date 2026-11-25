"""Steps for right_graph.feature."""

from itertools import product

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import Background, EvidenceScore, Graph, NodeMaker
from src.causal_model.nodes import SCALES

scenarios("right_graph.feature")

TRUTH = {("x", "z"), ("y", "z")}


class PlainMaker(NodeMaker):
    """Sums of curves only: the plain scale and no surfaces."""

    def scales(self, name, values):
        return [SCALES[0]]

    def make(self, name, parents, scale=None):
        node = super().make(name, parents, scale)
        node.screening = None
        return node


@given(parsers.parse("{count:d} people where {form}"))
def people(context, count, form):
    random = np.random.default_rng(0)
    x, y, e = random.uniform(-1, 1, count), random.uniform(-1, 1, count), random.normal(0, 0.1, count)
    flag = (random.uniform(size=count) < 0.4).astype(float)
    kinds = {}
    if form == "z is a sum of curves":
        z = np.sin(2 * x) + y ** 2 + 3 * e
    elif form == "z is e to a sum":
        z = np.exp(x + y + e)
    elif form == "z is a logistic of a sum":
        z = 1 / (1 + np.exp(-2 * (x + y + e)))
    elif form == "z is a product that can be negative":
        z = 2 * x * y + e
    elif form == "z is a gate":
        x, kinds = flag, {"x": "binary"}
        z = 2 * flag * np.tanh(2 * y) + e
    elif form == "z has noise that multiplies":
        z = (1.5 + 0.5 * x + y ** 2) * np.exp(1.5 * e)
    elif form == "z is exactly x minus y":
        z = x - y
    elif form == "z is exactly 100x minus 30y":
        x, y = random.normal(size=count), random.normal(size=count)
        z = 100 * x - 30 * y
    else:
        x, y = np.exp(0.5 * random.normal(size=count)), np.exp(0.5 * random.normal(size=count))
        z = x * y
    context["data"], context["background"] = {"x": x, "y": y, "z": z}, Background(kinds=kinds)


def every_graph() -> list[set]:
    """The 25 graphs without loops over x, y and z."""
    graphs = []
    for choice in product((None, 0, 1), repeat=3):
        pairs = (("x", "y"), ("x", "z"), ("y", "z"))
        arrows = {pair if way == 0 else pair[::-1] for pair, way in zip(pairs, choice) if way is not None}
        if arrows not in ({("x", "y"), ("y", "z"), ("z", "x")}, {("y", "x"), ("z", "y"), ("x", "z")}):
            graphs.append(arrows)
    return graphs


def score_all(context, maker: NodeMaker) -> None:
    score = EvidenceScore(context["data"], maker)
    totals = {}
    for arrows in every_graph():
        graph = Graph(["x", "y", "z"])
        for parent, child in arrows:
            graph.add(parent, child)
        totals[frozenset(arrows)] = sum(score(name, graph.parents[name]) - 4 * len(graph.parents[name])
                                        for name in graph.variables)
    context["truth"] = totals.pop(frozenset(TRUTH))
    context["wrong"] = max(totals.values())


@when("every graph over the three variables is scored")
def with_everything(context):
    score_all(context, NodeMaker(context["background"]))


@when("every graph over the three variables is scored without scales and without surfaces")
def without(context):
    score_all(context, PlainMaker(context["background"]))


@then(parsers.parse("the true graph scores highest, or within {margin:g} of the highest"))
def truth_wins(context, margin):
    assert context["truth"] > context["wrong"] - margin


@then(parsers.parse("a wrong graph beats the true one by more than {margin:g}"))
def wrong_wins(context, margin):
    assert context["wrong"] > context["truth"] + margin
