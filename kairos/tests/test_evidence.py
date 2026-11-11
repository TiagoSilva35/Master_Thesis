"""Steps for evidence.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.fitting import BayesianFitting
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from tests.helpers import sigmoid, world

scenarios("evidence.feature")


@given(parsers.parse("{count:d} people where y = sin(x) plus bell-shaped noise"))
def sine_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(0, 4, count)
    context["data"] = {"x": x, "y": np.sin(x) + random.normal(0, 0.3, count)}


@given(parsers.parse("{count:d} people with a yes/no outcome whose chance rises with x"))
def yes_no_people(context, count):
    random = np.random.default_rng(1)
    x = random.uniform(0, 4, count)
    context["data"] = {"x": x, "Y": (random.uniform(size=count) < sigmoid(-2 + x)).astype(float)}


@given(parsers.parse("a Bayesian node for y with parent x and {intervals:d} intervals"))
def feature_node(context, intervals):
    context["node"] = Node("y", ["x"], pairs=[], intervals=intervals, fitting=BayesianFitting())


@given(parsers.parse("a Bayesian outcome node for Y with parent x and {intervals:d} intervals"))
def outcome_node(context, intervals):
    context["node"] = OutcomeNode("Y", ["x"], pairs=[], intervals=intervals, fitting=BayesianFitting())


@given(parsers.parse("{count:d} people from the large world"))
def large_world_people(context, count):
    context["data"] = world(5).world.sample(count, 3)
    context["parents"] = world(5).world.parents()["Y"]


@given("its outcome node with surfaces income×debt and age×savings")
def large_world_outcome(context):
    context["node"] = OutcomeNode("Y", context["parents"], pairs=[("income", "debt"), ("age", "savings")],
                                  intervals=16, fitting=BayesianFitting())


def _design_and_penalty(context, strength):
    node, data = context["node"], context["data"]
    node.place_knots(data)
    design = node.design(data)
    blocks = node.smoothness_blocks()
    penalty = node.fitting.penalty([strength] * len(blocks), design.shape[1], blocks)
    return design, penalty


@when(parsers.parse("I compute its evidence with λ = {strength:g}"))
def feature_evidence(context, strength):
    design, penalty = _design_and_penalty(context, strength)
    context["design"], context["penalty"] = design, penalty
    context["fit"] = context["node"]._solve(design, context["data"]["y"], penalty)


@when(parsers.parse("I compute its Laplace evidence with λ = {strength:g}"))
def laplace_evidence(context, strength):
    design, penalty = _design_and_penalty(context, strength)
    context["design"], context["penalty"] = design, penalty
    context["node"]._start = None
    context["fit"] = context["node"]._solve(design, context["data"]["Y"], penalty)


@when(parsers.parse("I fit the outcome with every λ = {strength:g}"))
def fit_large_outcome(context, strength):
    laplace_evidence(context, strength)


@then("it equals the direct marginal likelihood up to the constant n/2·(log 2π + 1)")
def equals_direct_formula(context):
    y, design, penalty, fit = context["data"]["y"], context["design"], context["penalty"], context["fit"]
    n = len(y)
    noise = (np.sum((y - design @ fit.coefficients) ** 2) + fit.coefficients @ penalty @ fit.coefficients) / n
    covariance = noise * (np.eye(n) + design @ np.linalg.solve(penalty, design.T))
    direct = -0.5 * (np.linalg.slogdet(covariance)[1] + y @ np.linalg.solve(covariance, y) + n * np.log(2 * np.pi))
    constant = n / 2 * (np.log(2 * np.pi) + 1)
    assert abs(fit.evidence - direct - constant) < 1e-6


@then(parsers.parse("importance sampling with {draws:d} draws gives the same value within {tolerance:g}"))
def matches_importance_sampling(context, draws, tolerance):
    y, design, penalty, fit = context["data"]["Y"], context["design"], context["penalty"], context["fit"]
    random = np.random.default_rng(2)
    spread = 2.0 * fit.covariance  # a wider bell than Laplace's, so the samples cover the tails
    samples = random.multivariate_normal(fit.coefficients, spread, draws)
    score = samples @ design.T
    log_likelihood = -(y * np.logaddexp(0, -score) + (1 - y) * np.logaddexp(0, score)).sum(axis=1)
    size = design.shape[1]
    log_prior = (0.5 * np.linalg.slogdet(penalty)[1] - size / 2 * np.log(2 * np.pi)
                 - 0.5 * np.einsum("ij,jk,ik->i", samples, penalty, samples))
    offset = samples - fit.coefficients
    log_proposal = (-0.5 * np.einsum("ij,jk,ik->i", offset, np.linalg.inv(spread), offset)
                    - 0.5 * np.linalg.slogdet(spread)[1] - size / 2 * np.log(2 * np.pi))
    log_weights = log_likelihood + log_prior - log_proposal
    sampled = np.log(np.mean(np.exp(log_weights - log_weights.max()))) + log_weights.max()
    assert abs(fit.evidence - sampled) < tolerance


@then(parsers.parse("the remaining slope is below {tolerance:g}"))
def newton_converged(context, tolerance):
    y, design, penalty, fit = context["data"]["Y"], context["design"], context["penalty"], context["fit"]
    slope = design.T @ (y - sigmoid(design @ fit.coefficients)) - penalty @ fit.coefficients
    assert np.abs(slope).max() < tolerance
