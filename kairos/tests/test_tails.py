"""Steps for tails.feature."""

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.helpers import world, world_one_model

scenarios("tails.feature")


@given(parsers.parse("{count:d} people from world 1"))
def world_one_people(context, count):
    context["people"] = world(0).world.sample(count, 5)


@given("a Bayesian causal KAN fitted to them")
def fitted_model(context):
    context["model"] = world_one_model(context["people"])


@when(parsers.parse("I predict savings for income {income:g}"))
def predict_savings(context, income):
    context["prediction"] = float(context["model"].node("savings").predict({"income": np.array([income])})[0])


@then(parsers.parse("the prediction is within {tolerance:g} of the true {truth:g}"))
def prediction_close(context, tolerance, truth):
    assert context["prediction"] == pytest.approx(truth, abs=tolerance)


@then(parsers.parse("the {parent} -> {child} band is wider at {far:g} than at {near:g}"))
def band_widens(context, parent, child, far, near):
    curve = context["model"].node(child).curves[parent]
    low, high = curve.band(np.array([near, far]))
    widths = high - low
    assert widths[1] > widths[0]
