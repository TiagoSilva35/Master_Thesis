"""Steps for screening.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.fitting import BayesianFitting
from src.causal_model.node import Node
from src.causal_model.screening import PairScreening

scenarios("screening.feature")


@given(parsers.parse("{count:d} people where y = sin(x1) + 0.5·x2 + {strength:g}·(x1 − 2)·(x2 − 2) plus noise"))
def interaction_people(context, count, strength):
    random = np.random.default_rng(3)
    x1, x2 = random.uniform(0, 4, count), random.uniform(0, 4, count)
    y = np.sin(x1) + 0.5 * x2 + strength * (x1 - 2) * (x2 - 2) + random.normal(0, 0.3, count)
    context["data"] = {"x1": x1, "x2": x2, "y": y}


@when("a node for y screens its pairs of parents")
def screen(context):
    node = Node("y", ["x1", "x2"], pairs=[], intervals=12, fitting=BayesianFitting(), screening=PairScreening())
    node.fit(context["data"])
    context["pairs"] = node.pairs


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


@then(parsers.parse("the surface x1×x2 is {decision}"))
def surface_decision(context, decision):
    kept = ("x1", "x2") in context["pairs"]
    assert kept == (decision == "kept")


@then(parsers.parse("the two gains differ by less than {tolerance:g}"))
def gains_close(context, tolerance):
    first, second = context["gains"]
    assert abs(first - second) < tolerance
