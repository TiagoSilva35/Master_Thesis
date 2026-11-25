"""Steps for fitting.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.nodes import BayesianFitting, Node

scenarios("fitting.feature")

CURVES = {"sin 3x": lambda x: np.sin(3 * x), "2x": lambda x: 2 * x, "0": lambda x: 0 * x}


@given(parsers.parse("{count:d} people where y = {curve} plus noise of spread {noise:g}"))
def curve_people(context, count, curve, noise):
    random = np.random.default_rng(0)
    x = random.uniform(-1, 1, count)
    context["data"] = {"x": x, "y": CURVES[curve](x) + random.normal(0, noise, count)}


@when("y is fitted, and every λ of a fine grid is tried")
def fit_and_grid(context):
    node, data = Node("y", ["x"], intervals=12), context["data"]
    node.fit(data)
    design = node.design(data)
    solve, blocks = node.solver(design, data["y"]), node.smoothness_blocks()
    context["node"] = node
    context["grid"] = max(solve(node.fitting.penalty([strength], design.shape[1], blocks)).evidence
                          for strength in 10.0 ** np.arange(-5, 6, 0.1))


@then("the fit's evidence is at least the grid's best, within 0.1")
def as_good_as_grid(context):
    assert context["node"].evidence > context["grid"] - 0.1


@then("the λ chosen is below 0.1")
def weak_belief(context):
    assert context["node"].smoothness["x"] < 0.1


@given(parsers.parse("{count:d} people where z is exactly 2 times x minus w"))
def formula_people(context, count):
    random = np.random.default_rng(0)
    x, w = random.normal(0, 1, count), random.normal(0, 2, count)
    context["data"] = {"x": x, "w": w, "z": 2 * x - w}


def fitted(context, child: str, parents: list[str]) -> Node:
    node = Node(child, parents, noise_floor=1e-4)
    node.fit(context["data"])
    return node


@when("z is fitted on x and w with a noise floor of 0.0001")
def fit_formula(context):
    context["gain"] = fitted(context, "z", ["x", "w"]).evidence - fitted(context, "z", []).evidence


@then("its gain over no parents is at most N/2 · log(1 / 0.0001)")
def capped(context):
    assert 0 < context["gain"] <= len(context["data"]["z"]) / 2 * np.log(1e4)


@when("z is fitted on x and w, and x on z and w, with a noise floor of 0.0001")
def fit_both_ways(context):
    context["variances"] = {}
    for child, parents in (("z", ["x", "w"]), ("x", ["z", "w"])):
        node = Node(child, parents, noise_floor=1e-4, fitting=BayesianFitting(ridge=1e-8))  # so only the floor counts
        node.place_knots(context["data"])
        design = node.design(context["data"])
        fit = node.fitting.fit(node.solver(design, context["data"][child]), design.shape[1], node.smoothness_blocks())
        context["variances"][child] = fit.noise_variance


@then("z's noise variance is 4 times x's, to within 1%, because z moves 2 when x moves 1")
def same_floor(context):
    variances = context["variances"]
    assert np.isclose(variances["z"] / variances["x"], 4, rtol=0.01)
