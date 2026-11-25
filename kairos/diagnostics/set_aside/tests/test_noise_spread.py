"""Steps for noise_spread.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting, FlexibleNode

scenarios("noise_spread.feature")


def maker(learn_noise: bool, noise_floor: float = 0.01) -> NodeMaker:
    return NodeMaker({}, None, BayesianFitting(), intervals=8, noise_floor=noise_floor, learn_noise=learn_noise)


@given(parsers.parse("{count:d} people where y = x plus noise whose spread grows from 0.3 to 1.3 as x goes from 0 to 2"))
def growing_spread(context, count):
    random = np.random.default_rng(2)
    x = random.uniform(0, 2, count)
    context["data"] = {"x": x, "y": x + (0.3 + 0.5 * x) * random.normal(0, 1, count)}


@given(parsers.parse("{count:d} people where y = x plus noise with a constant spread"))
def constant_spread(context, count):
    random = np.random.default_rng(2)
    x = random.uniform(0, 2, count)
    context["data"] = {"x": x, "y": x + 0.5 * random.normal(0, 1, count)}


@given(parsers.parse("{count:d} people where y = x plus evenly spread noise whose spread follows a sigmoid of x"))
def sigmoid_spread(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(-2, 2, count)
    spread = 0.5 * (0.2 + 1 / (1 + np.exp(-3 * x)))
    context["data"] = {"x": x, "y": x + spread * random.uniform(-1.7, 1.7, count)}


@given(parsers.parse("{count:d} people where z is x minus w, rounded to 0.01"))
def rounded_formula(context, count):
    random = np.random.default_rng(0)
    x, w = random.normal(0, 1, count), random.normal(0, 1, count)
    context["data"] = {"x": x, "w": w, "z": np.round(x - w, 2)}


@when("y is fitted with learnt noise")
def fit_flexible(context):
    context["node"] = FlexibleNode("y", ["x"], pairs=[], fitting=BayesianFitting())
    context["node"].fit(context["data"])


@when("both directions are scored with bell-shaped and with learnt noise")
def score_directions(context):
    for label, learn_noise in (("bell", False), ("learnt", True)):
        score = EvidenceScore(context["data"], maker(learn_noise))
        context[label] = score("x", []) + score("y", ["x"]) - score("y", []) - score("x", ["y"])


@when("z is scored with learnt noise and noise floor 0.01, with and without its parents")
def score_formula(context):
    score = EvidenceScore(context["data"], maker(True))
    context["gain"] = score("z", ["w", "x"]) - score("z", [])


@then("the spread² at x near 2 is more than 5 times the spread² at x near 0")
def follows_spread(context):
    x, variance = context["data"]["x"], context["node"].spread.variance
    assert variance[x > 1.8].mean() > 5 * variance[x < 0.2].mean()


@then("the largest spread² is within 10% of the smallest")
def stays_flat(context):
    variance = context["node"].spread.variance
    assert variance.max() < 1.1 * variance.min()


@then(parsers.parse("the bell-shaped score prefers y → x by more than {margin:g}"))
def bell_wrong(context, margin):
    assert context["bell"] < -margin


@then(parsers.parse("the learnt noise prefers x → y by more than {margin:g}"))
def learnt_right(context, margin):
    assert context["learnt"] > margin


@then("z's gain is at most 1.1 times N/2 · log(1.01 / 0.01)")
def capped(context):
    assert context["gain"] <= 1.1 * len(context["data"]["z"]) / 2 * np.log(1.01 / 0.01)
