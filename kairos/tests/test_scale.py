"""Steps for scale.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import Background, EvidenceScore, NodeMaker
from src.causal_model.nodes import SCALES, LogScale, Node

scenarios("scale.feature")

BY_NAME = {scale.name: scale for scale in SCALES}


@given(parsers.parse("{count:d} people where z is {form}"))
def people(context, count, form):
    random = np.random.default_rng(0)
    x, y, noise = random.uniform(-1, 1, count), random.uniform(-1, 1, count), random.normal(0, 0.1, count)
    total = x + y + noise
    z = {"a sum": 2.5 + total, "e to a sum": np.exp(total), "a logistic of one": 1 / (1 + np.exp(-2 * total))}[form]
    context["data"] = {"x": x, "y": y, "z": z}


@when("z is scored with its parents x and y")
def score(context):
    context["scale"] = EvidenceScore(context["data"], NodeMaker(Background())).scale("z", ["x", "y"]).name


@then(parsers.parse("the {scale} scale is chosen"))
def chosen(context, scale):
    assert context["scale"] == scale


def offered(values) -> list[str]:
    return [scale.name for scale in NodeMaker(Background()).scales("v", np.array(values))]


@then("values with a zero are offered the plain scale only")
def zero():
    assert offered([0.0, 0.5, 0.9]) == ["plain"]


@then("values above 1 are offered plain and log")
def positive():
    assert offered([0.5, 2.0, 30.0]) == ["plain", "log"]


@then("values between 0 and 1 are offered plain, log and log-odds")
def share():
    assert offered([0.1, 0.5, 0.9]) == ["plain", "log", "log-odds"]


@given(parsers.parse("values between 0.05 and 0.95 on the {scale} scale"))
def some_values(context, scale):
    context["scale"], context["values"] = BY_NAME[scale], np.linspace(0.05, 0.95, 50)


@then("backward undoes forward")
def undone(context):
    scale, values = context["scale"], context["values"]
    assert np.allclose(scale.backward(scale.forward(values)), values)


@then("stretch is the log of forward's slope")
def slope(context):
    scale, values, step = context["scale"], context["values"], 1e-6
    slopes = (scale.forward(values + step) - scale.forward(values - step)) / (2 * step)
    assert np.allclose(scale.stretch(values), np.log(slopes), atol=1e-5)


@given(parsers.parse("{count:d} numbers whose logs are bell-shaped with spread {spread:g}"))
def lognormal(context, count, spread):
    context["v"] = np.exp(np.random.default_rng(0).normal(0, spread, count))


class BellNode(Node):
    """Always a bell of one size, so that only the scale differs."""

    def read_noise(self, fit, data):
        pass


@when("they are scored without parents on the plain and on the log scale")
def score_both(context):
    evidences = []
    for scale in (SCALES[0], LogScale()):
        node = BellNode("v", [], scale=scale)
        node.fit({"v": context["v"]})
        evidences.append(node.evidence)
    context["difference"] = evidences[1] - evidences[0]


@then("the difference is within 5% of the difference between the two bells' log-likelihoods")
def densities(context):
    v, logs = context["v"], np.log(context["v"])

    def bell(values):
        return np.sum(-0.5 * np.log(2 * np.pi * values.var()) - (values - values.mean()) ** 2 / (2 * values.var()))
    expected = (bell(logs) - logs.sum()) - bell(v)  # a lognormal's density is the bell's of the log, divided by v
    assert abs(context["difference"] - expected) < 0.05 * abs(expected)


@when("z is fitted on the log scale")
def fit_log(context):
    context["node"] = Node("z", ["x", "y"], scale=LogScale())
    context["node"].fit(context["data"])


@then("its predictions are within 15% of the true values on average, and its noise plus its sum gives log z")
def own_units(context):
    node, data = context["node"], context["data"]
    assert np.mean(np.abs(node.predict(data) / data["z"] - 1)) < 0.15
    assert np.allclose(node.noise(data) + node.total(data), np.log(data["z"]))
