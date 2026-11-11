"""Steps for bands.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from tests.helpers import world, world_one_model

scenarios("bands.feature")


@given(parsers.parse("{samples:d} samples of {count:d} people from world 1"))
def samples_of_people(context, samples, count):
    context["samples"] = [world(0).world.sample(count, 100 + seed) for seed in range(samples)]


@when("I fit a Bayesian causal KAN to each sample")
def fit_each(context):
    context["models"] = [world_one_model(people) for people in context["samples"]]


@then(parsers.parse("on average at least {share:d}% of points of every band contain the true curve"))
def bands_cover(context, share):
    inside = {"income -> savings": [], "income -> Y": [], "savings -> Y": []}
    for people, model in zip(context["samples"], context["models"]):
        income_grid = np.linspace(0.2, 3.8, 30)
        savings_grid = np.linspace(np.quantile(people["savings"], 0.05), np.quantile(people["savings"], 0.95), 30)
        checks = {
            "income -> savings": (model.node("savings").curves["income"], lambda v: 0.5 * v ** 2, "income", income_grid),
            "income -> Y": (model.node("Y").curves["income"], lambda v: v, "income", income_grid),
            "savings -> Y": (model.node("Y").curves["savings"], lambda v: 0.5 * v, "savings", savings_grid),
        }
        for name, (curve, truth, parent, grid) in checks.items():
            low, high = curve.band(grid)
            centred = truth(grid) - truth(people[parent]).mean()  # the model's curves average zero over the data
            inside[name].append(np.mean((low <= centred) & (centred <= high)))
    for name, shares in inside.items():
        assert 100 * np.mean(shares) >= share, f"{name}: only {100 * np.mean(shares):.0f}% inside"
