"""Steps for spline.feature."""

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.spline import Spline
from tests.helpers import CoxDeBoor

scenarios("spline.feature")


@given(parsers.parse("a spline from {low:g} to {high:g} with {intervals:d} intervals"))
def a_spline(context, low, high, intervals):
    context["spline"] = Spline(low, high, intervals)


@given("its heights equal its bump centres")
def heights_on_centres(context):
    context["spline"].heights = context["spline"].centers.copy()


@when(parsers.parse("I compute its bump weights at {count:d} random points in its range"))
def weights_at_random_points(context, count):
    spline = context["spline"]
    context["x"] = np.random.default_rng(0).uniform(spline.knots[0], spline.knots[-1], count)
    context["weights"] = spline.weights(context["x"])


@given("its heights are the squares of its bump centres")
def heights_squared(context):
    context["spline"].heights = context["spline"].centers ** 2


@when(parsers.parse("I compute its bump weights at {first:g} and {second:g}"))
def weights_at_two_points(context, first, second):
    context["weights"] = context["spline"].weights(np.array([first, second]))


@when(parsers.parse("I look at the point {x:g}"))
def look_at_point(context, x):
    context["weights"] = context["spline"].weights(np.array([x]))[0]


@when("I ask for its band before fitting")
def band_before_fitting(context):
    try:
        context["band"] = context["spline"].band(np.array([1.0, 2.0]))
    except ValueError as error:
        context["error"] = error


@then(parsers.parse("they match the Cox-de Boor recursion to within {tolerance:g}"))
def match_textbook(context, tolerance):
    spline = context["spline"]
    gap = spline.gap
    knots = np.arange(spline.knots[0] - 3 * gap, spline.knots[-1] + 3.5 * gap, gap)  # 3 extra knots each side
    textbook = CoxDeBoor(knots).weights(context["x"])
    assert np.abs(context["weights"] - textbook).max() < tolerance


@then("every point's weights add up to 1")
def weights_sum_to_one(context):
    assert np.allclose(context["weights"].sum(axis=1), 1.0, atol=1e-12)


@then(parsers.parse("{count:d} bumps have a weight above zero"))
def active_bumps(context, count):
    assert np.sum(context["weights"] > 1e-12) == count


@then("the weights above zero are 1/6, 4/6 and 1/6")
def weights_on_a_knot(context):
    active = context["weights"][context["weights"] > 1e-12]
    assert np.allclose(active, [1 / 6, 4 / 6, 1 / 6])


@then("the curve is the straight line y = x inside its range")
def straight_line_inside(context):
    spline = context["spline"]
    x = np.linspace(spline.knots[0], spline.knots[-1], 101)
    assert np.allclose(spline(x), x, atol=1e-12)


@then(parsers.parse("the curve at {x:g} is {y:g}"))
def curve_value(context, x, y):
    assert context["spline"](np.array([x]))[0] == pytest.approx(y, abs=1e-9)


@then("it refuses because the curve has no spread yet")
def refuses_without_spread(context):
    assert isinstance(context.get("error"), ValueError)


@then("beyond the range the curve is a straight line with its slope at the edge")
def tail_follows_edge_slope(context):
    spline = context["spline"]
    for edge, direction in ((spline.knots[-1], 1.0), (spline.knots[0], -1.0)):
        step = 1e-6
        slope = (spline(np.array([edge])) - spline(np.array([edge - direction * step])))[0] / (direction * step)
        for distance in (0.5, 2.0):
            beyond = edge + direction * distance
            expected = spline(np.array([edge]))[0] + slope * direction * distance
            assert spline(np.array([beyond]))[0] == pytest.approx(expected, abs=1e-4)


@then(parsers.parse("the points {points} are {labels}"))
def outside_or_inside(context, points, labels):
    values = [float(v) for v in points.replace(" and ", ", ").split(", ")]
    expected = [label == "outside" for label in labels.replace(" and ", ", ").split(", ")]
    assert context["spline"].outside(np.array(values)).tolist() == expected
