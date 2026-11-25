"""Steps for levels.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.nodes import BayesianFitting, Node

scenarios("levels.feature")


def fitted(g: np.ndarray, y: np.ndarray, kind: str) -> Node:
    node = Node("y", ["g"], pairs=[], fitting=BayesianFitting(), kinds={"g": kind})
    node.fit({"g": g, "y": y})
    return node


@given(parsers.parse("{count:d} people where y depends on an unordered parent g with 4 levels"))
def unordered_people(context, count):
    random = np.random.default_rng(0)
    g = random.integers(0, 4, count).astype(float)
    context["g"], context["y"] = g, np.array([0.0, 1.0, -1.0, 0.5])[g.astype(int)] + random.normal(0, 0.5, count)


@given(parsers.parse("{count:d} people where y steps up by 1 between levels 2 and 3 of an ordered parent g"))
def step_people(context, count):
    random = np.random.default_rng(0)
    g = random.integers(0, 5, count).astype(float)
    context["g"], context["y"] = g, np.array([0.0, 0.0, 0.0, 1.0, 1.0])[g.astype(int)] + random.normal(0, 0.5, count)


@given(parsers.parse("{count:d} people where y is 0.8 higher when the yes/no parent g is 1"))
def binary_people(context, count):
    random = np.random.default_rng(1)
    g = random.integers(0, 2, count).astype(float)
    context["g"], context["y"] = g, 0.8 * g + random.normal(0, 0.5, count)


@given(parsers.parse("{count:d} people where y rises evenly over the 8 levels of an ordered parent g"))
def even_people(context, count):
    random = np.random.default_rng(0)
    g = random.integers(0, 8, count).astype(float)
    context["g"], context["y"] = g, 0.25 * g + random.normal(0, 1, count)


@when("y is fitted with g's codes and with the codes renumbered")
def fit_renumbered(context):
    renumbered = np.array([2.0, 0.0, 3.0, 1.0])[context["g"].astype(int)]
    context["evidences"] = [fitted(codes, context["y"], "nominal").evidence for codes in (context["g"], renumbered)]


@when(parsers.parse("y is fitted with g as an {kind} parent"))
@when(parsers.parse("y is fitted with g as a {kind} parent"))
def fit_with_kind(context, kind):
    context["node"] = fitted(context["g"], context["y"], {"ordered": "ordinal", "yes/no": "binary"}[kind])


@when("y is fitted with g's levels in their order and in a shuffled order")
def fit_shuffled(context):
    shuffled = np.array([3.0, 6.0, 0.0, 5.0, 1.0, 7.0, 2.0, 4.0])[context["g"].astype(int)]
    context["in order"] = fitted(context["g"], context["y"], "ordinal").evidence
    context["shuffled"] = fitted(shuffled, context["y"], "ordinal").evidence


@then("both evidences are equal")
def equal_evidences(context):
    first, second = context["evidences"]
    assert abs(first - second) < 1e-8


@then("the step between levels 2 and 3 is above 0.7 and every other step is below 0.2")
def step_recovered(context):
    steps = np.diff(context["node"].curves["g"].heights)
    assert steps[2] > 0.7
    assert np.all(np.abs(np.delete(steps, 2)) < 0.2)


@then("g's edge has two heights, 0.8 apart within 0.15")
def two_heights(context):
    heights = context["node"].curves["g"].heights
    assert len(heights) == 2 and abs(heights[1] - heights[0] - 0.8) < 0.15


@then("a level of g never seen pushes y by the average of g's heights")
def unseen_level(context):
    edge = context["node"].curves["g"]
    assert np.isclose(edge(np.array([9.0]))[0], edge.heights.mean())
    assert edge.outside(np.array([9.0]))[0] and not edge.outside(np.array([2.0]))[0]


@then(parsers.parse("the evidence in their order is higher by more than {margin:g}"))
def order_helps(context, margin):
    assert context["in order"] > context["shuffled"] + margin
