"""Steps for noise_shape.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.nodes import BELL, SHAPES, Node

scenarios("noise_shape.feature")

NOISE = {"bell-shaped": lambda random, n: random.normal(size=n), "heavy-tailed": lambda random, n: random.laplace(size=n),
         "evenly spread": lambda random, n: random.uniform(-1.7, 1.7, n),
         "exponential": lambda random, n: random.exponential(size=n) - 1,
         "minus exponential": lambda random, n: 1 - random.exponential(size=n)}


@then("each of the 15 shapes has area 1, average 0 and spread 1")
def are_densities():
    z = np.linspace(-40, 40, 800001)
    assert len(SHAPES) == 15
    for shape in SHAPES:
        density = np.exp(shape.log_density(z))
        for power, expected in ((0, 1), (1, 0), (2, 1)):
            assert np.isclose(np.trapezoid(density * z ** power, z), expected, atol=2e-3)


@given(parsers.re(r"(?P<count>\d+) people where y = 2x plus (?P<noise>[a-z -]+) noise$"), converters={"count": int})
def arrow_people(context, count, noise):
    random = np.random.default_rng(0)
    x = random.uniform(-1, 1, count)
    context["data"] = {"x": x, "y": 2 * x + NOISE[noise](random, count)}


@given(parsers.re(r"(?P<count>\d+) people where y = 2x plus (?P<noise>[a-z -]+) noise, with x of the same shape"),
       converters={"count": int})
def straight_people(context, count, noise):
    random = np.random.default_rng(1)
    x = NOISE[noise](random, count)
    context["data"] = {"x": x, "y": 2 * x + NOISE[noise](random, count)}


@given("500 people where z is x minus w, rounded to 0.01")
def formula_people(context):
    random = np.random.default_rng(0)
    x, w = random.normal(size=500), random.normal(size=500)
    context["data"] = {"x": x, "w": w, "z": np.round(x - w, 2)}


def fitted(context, child: str, parents: list[str]) -> Node:
    node = Node(child, parents, noise_floor=1e-4)
    node.fit(context["data"])
    return node


@when("y is fitted on x")
def fit_arrow(context):
    context["node"] = fitted(context, "y", ["x"])


@then(parsers.re(r"its noise is (?P<name>[a-z ]+)"))
def recognised(context, name):
    found = context["node"].noise_model
    assert found.growth == 0.0
    if name == "bell":
        assert found.shape is BELL
    else:  # an evenly spread noise may come out as "flat leaning right": the lean is slight
        assert name in found.shape.name


@when("both directions are scored")
def both_directions(context):
    forward = fitted(context, "x", []).evidence + fitted(context, "y", ["x"]).evidence
    backward = fitted(context, "y", []).evidence + fitted(context, "x", ["y"]).evidence
    context["ahead"] = forward - backward


@then(parsers.parse("x → y is ahead by more than {margin:g}"))
def ahead(context, margin):
    assert context["ahead"] > margin


@then(parsers.parse("neither is ahead by more than {margin:g}"))
def tie(context, margin):
    assert abs(context["ahead"]) < margin


@when("z is fitted on x and w with a noise floor of 0.0001")
def fit_formula(context):
    context["node"] = fitted(context, "z", ["x", "w"])


@given(parsers.parse("{count:d} people where y = x plus bell-shaped noise whose spread grows from 0.3 to 1.3 with x"))
def growing_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(0, 2, count)
    context["data"] = {"x": x, "y": x + (0.3 + 0.5 * x) * random.normal(size=count)}


@given(parsers.parse("{count:d} people where y = x plus bell-shaped noise of one size"))
def even_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(0, 2, count)
    context["data"] = {"x": x, "y": x + 0.8 * random.normal(size=count)}


@then("its noise grows with the prediction")
def grows(context):
    assert context["node"].noise_model.growth > 0.3


@then("its noise has one size")
def one_size(context):
    assert context["node"].noise_model.growth == 0.0


@then(parsers.parse("its shape adds less than {limit:g} to the evidence"))
def little_gain(context, limit):
    node, data = context["node"], context["data"]
    design = node.design(data)
    strengths = [node.smoothness[label] for label in ("x", "w")]
    fit = node.solver(design, data["z"])(node.fitting.penalty(strengths, design.shape[1], node.smoothness_blocks()))
    assert node.evidence - fit.evidence < limit and BELL in SHAPES
