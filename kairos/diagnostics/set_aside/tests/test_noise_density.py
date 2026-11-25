"""Steps for noise_density.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.nodes import BayesianFitting, FlexibleNode, Node, NoiseDensity
from tests.helpers import ImportanceSampling

scenarios("noise_density.feature")


def bell_log_likelihood(numbers: np.ndarray) -> float:
    """Σ log of the bell with the numbers' own average and spread."""
    variance = numbers.var()
    return float(np.sum(-0.5 * np.log(2 * np.pi * variance) - (numbers - numbers.mean()) ** 2 / (2 * variance)))


@given(parsers.parse("{count:d} bell-shaped numbers"))
def bell_numbers(context, count):
    context["numbers"] = np.random.default_rng(0).normal(0, 1, count)


@given(parsers.parse("{count:d} numbers spread evenly between −1 and 1"))
def even_numbers(context, count):
    context["numbers"] = np.random.default_rng(0).uniform(-1, 1, count)


@given(parsers.parse("{count:d} people where y = 2x plus noise, with x and the noise spread evenly"))
def line_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(-1, 1, count)
    context["data"] = {"x": x, "y": 2 * x + random.uniform(-1, 1, count)}


@given(parsers.parse("{count:d} people where y = x² plus small bell-shaped noise, with x spread evenly"))
def square_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(0, 1, count)
    context["data"] = {"x": x, "y": x ** 2 + random.normal(0, 0.05, count)}


@given(parsers.parse("{count:d} people where y = sin(x) plus heavy-tailed noise"))
def heavy_tailed_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(0, 4, count)
    context["data"] = {"x": x, "y": np.sin(x) + random.laplace(0, 0.3, count)}


@when("their shape is learnt")
def learn_shape(context):
    context["density"] = NoiseDensity()
    context["density"].fit(context["numbers"])


def total(data: dict, cause: str, effect: str, kind) -> float:
    """The evidence of cause → effect: the cause as a root plus the effect given the cause."""
    root = kind(cause, [], pairs=[], fitting=BayesianFitting())
    child = kind(effect, [cause], pairs=[], fitting=BayesianFitting())
    root.fit(data)
    child.fit(data)
    return root.evidence + child.evidence


@when("both directions are scored with bell-shaped and with learnt noise")
def score_directions(context):
    data = context["data"]
    for label, kind in (("bell", Node), ("learnt", FlexibleNode)):
        context[label] = total(data, "x", "y", kind) - total(data, "y", "x", kind)


@when(parsers.parse("y is fitted with learnt noise and {intervals:d} intervals"))
def fit_flexible(context, intervals):
    context["node"] = FlexibleNode("y", ["x"], pairs=[], intervals=intervals, fitting=BayesianFitting())
    context["node"].fit(context["data"])


@then(parsers.parse("λ is at least {smallest:g} and the log-likelihood is within {tolerance:g} of the bell's"))
def stays_bell(context, smallest, tolerance):
    density, numbers = context["density"], context["numbers"]
    assert density.smoothness >= smallest
    assert abs(density.log_density(numbers).sum() - bell_log_likelihood(numbers)) < tolerance


@then("the shape's flexibility costs more than it gains over the bell")
def does_not_pay(context):
    density, numbers = context["density"], context["numbers"]
    assert density.log_density(numbers).sum() + density.occam < bell_log_likelihood(numbers)


@then("f is within 10% of ½ at −0.5, 0 and 0.5")
def flat(context):
    values = np.exp(context["density"].log_density(np.array([-0.5, 0.0, 0.5])))
    assert np.all(np.abs(values - 0.5) < 0.05)


@then(parsers.parse("the log-likelihood beats the bell's by more than {margin:g}"))
def beats_bell(context, margin):
    numbers = context["numbers"]
    assert context["density"].log_density(numbers).sum() > bell_log_likelihood(numbers) + margin


@then(parsers.parse("the learnt noise prefers x → y by more than {margin:g}"))
def learnt_prefers(context, margin):
    assert context["learnt"] > margin


@then(parsers.parse("the bell-shaped score does not prefer x → y by more than {margin:g}"))
def bell_blind(context, margin):
    assert context["bell"] < margin


@then(parsers.parse("the bell-shaped score also prefers x → y by more than {margin:g}"))
def bell_prefers(context, margin):
    assert context["bell"] > margin


@then(parsers.parse("importance sampling over the curve's heights, with the spread and f fixed, "
                    "gives the same value within {tolerance:g}"))
def matches_sampling(context, tolerance):
    node, data = context["node"], context["data"]
    node.place_knots(data)
    design = node.design(data)
    blocks = node.smoothness_blocks()
    penalty = node.fitting.penalty([node.smoothness[block.label] for block in blocks], design.shape[1], blocks)
    fit = node._solve(design, data["y"], penalty)
    density, variance = node.density, node.spread.variance

    def log_likelihood(samples):
        standardised = (data["y"][None, :] - samples @ design.T) / np.sqrt(variance)
        log_density = density.log_density(standardised.ravel()).reshape(standardised.shape)
        return (log_density - 0.5 * np.log(variance)).sum(axis=1)

    sampled = ImportanceSampling(40000).log_evidence(log_likelihood, penalty / fit.noise_variance,
                                                     fit.coefficients, fit.covariance)
    assert abs(node.evidence - density.occam - node.spread.occam - sampled) < tolerance


@then("values rounded to 0.01 have a rounding step of 0.01, and values not rounded have none")
def rounding_step():
    values = np.random.default_rng(0).normal(0, 1, 500)
    assert np.isclose(FlexibleNode.rounding_step(np.round(values, 2)), 0.01)
    assert FlexibleNode.rounding_step(values) == 0.0
