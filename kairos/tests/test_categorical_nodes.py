"""Steps for categorical_nodes.feature. The likelihoods used for importance
sampling are written here again, without the node classes."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import Background, EvidenceScore, NodeMaker
from src.causal_model.discovery.node_maker import LOG_ODDS_RIDGE
from src.causal_model.nodes import BayesianFitting, BinaryNode, CategoryNode, OrdinalNode
from tests.helpers import ImportanceSampling, sigmoid

scenarios("categorical_nodes.feature")

NODES = {"yes/no": BinaryNode, "ordered": OrdinalNode, "category": CategoryNode}


def ordered_codes(x: np.ndarray, random) -> np.ndarray:
    """Three levels pushed up by x: the ordered logit with cut-points −0.5 and 1."""
    return np.digitize(1.5 * x + random.logistic(size=len(x)), [-0.5, 1.0]).astype(float)


def category_codes(x: np.ndarray, random) -> np.ndarray:
    """Three categories coded 1, 2 and 5 (0, 3 and 4 never occur), with scores 0, x and 0.5 − x."""
    scores = np.column_stack([np.zeros(len(x)), x, 0.5 - x])
    chances = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    picked = np.array([random.choice(3, p=row) for row in chances])
    return np.array([5.0, 1.0, 2.0])[picked]


# --- Given -------------------------------------------------------------------

@given(parsers.parse("{count:d} people where b is yes with a chance that rises with x, and w is unrelated"))
def binary_people(context, count):
    random = np.random.default_rng(0)
    x, w = random.uniform(-2, 2, count), random.uniform(-2, 2, count)
    context["data"] = {"x": x, "w": w, "b": (random.uniform(size=count) < sigmoid(1.5 * x)).astype(float)}


@given(parsers.parse("{count:d} people with a {kind} child of x"))
def categorical_people(context, count, kind):
    random = np.random.default_rng(3)
    x = random.uniform(-2, 2, count)
    codes = ordered_codes(x, random) if kind == "ordered" else category_codes(x, random)
    context["data"], context["child"], context["kind"] = {"x": x, "y": codes}, "y", kind


@given(parsers.parse("{count:d} people where only 2% have b = 1"))
def rare_people(context, count):
    random = np.random.default_rng(4)
    x = random.uniform(-2, 2, count)
    context["data"] = {"x": x, "y": (random.uniform(size=count) < sigmoid(-4.2 + 0.5 * x)).astype(float)}
    context["child"], context["kind"] = "y", "yes/no"
    assert 0.01 < context["data"]["y"].mean() < 0.03


@given(parsers.parse("{count:d} people where b is yes exactly when x > 0"))
def separated_people(context, count):
    x = np.random.default_rng(5).uniform(-2, 2, count)
    context["data"] = {"x": x, "b": (x > 0).astype(float)}


@given(parsers.parse("{count:d} people with an ordered child of x whose codes skip 2"))
def skipping_people(context, count):
    random = np.random.default_rng(3)
    x = random.uniform(-2, 2, count)
    codes = ordered_codes(x, random)
    context["data"] = {"x": x, "y": np.where(codes == 2, 3.0, codes)}


@given(parsers.parse("{count:d} people where c is always 4"))
def constant_people(context, count):
    random = np.random.default_rng(6)
    context["data"] = {"x": random.uniform(-2, 2, count), "w": random.uniform(-2, 2, count), "c": np.full(count, 4.0)}


@given(parsers.parse("a {kind} node for it with {intervals:d} intervals"))
def categorical_node(context, kind, intervals):
    context["node"] = NODES[kind]("y", ["x"], pairs=[], intervals=intervals, fitting=BayesianFitting(ridge=LOG_ODDS_RIDGE))


# --- When ----------------------------------------------------------------------

def maker(kinds: dict) -> NodeMaker:
    return NodeMaker(Background(kinds=kinds))


@when("b is scored as a yes/no variable with parent x, and with parent w")
def score_binary(context):
    score = EvidenceScore(context["data"], maker({"b": "binary"}))
    context["with x"], context["with w"] = score("b", ["x"]), score("b", ["w"])


@when(parsers.parse("I compute its Laplace evidence with λ = {strength:g}"))
def laplace_evidence(context, strength):
    node, data = context["node"], context["data"]
    node.place_knots(data)
    design = node.design(data)
    solve = node.solver(design, data["y"])
    blocks = node.smoothness_blocks()
    penalty = node.fitting.penalty([strength] * len(blocks), node.size(design), blocks)
    context["fit"] = solve(penalty)
    context["design"], context["penalty"] = design, penalty


@when("it is fitted with its codes and with the codes renumbered")
def fit_renumbered(context):
    data = context["data"]
    renumbered = np.select([data["y"] == 1, data["y"] == 2, data["y"] == 5], [7.0, 0.0, 3.0])
    context["nodes"] = []
    for codes in (data["y"], renumbered):
        node = CategoryNode("y", ["x"], pairs=[], fitting=BayesianFitting(ridge=LOG_ODDS_RIDGE))
        node.fit({"x": data["x"], "y": codes})
        context["nodes"].append(node)


@when("b is fitted as a yes/no variable with parent x")
def fit_separated(context):
    context["node"] = BinaryNode("b", ["x"], pairs=[], fitting=BayesianFitting(ridge=LOG_ODDS_RIDGE))
    context["node"].fit(context["data"])


@when("it is fitted as an ordered variable with parent x")
def fit_ordered(context):
    context["node"] = OrdinalNode("y", ["x"], pairs=[], fitting=BayesianFitting(ridge=LOG_ODDS_RIDGE))
    context["node"].fit(context["data"])


# --- the likelihoods, written again ---------------------------------------------

def binary_likelihood(design, y):
    def log_likelihood(samples):
        score = samples @ design.T
        return -(y * np.logaddexp(0, -score) + (1 - y) * np.logaddexp(0, score)).sum(axis=1)
    return log_likelihood


def ordered_likelihood(design, y, levels):
    codes, columns = np.searchsorted(levels, y), design.shape[1]

    def log_likelihood(samples):
        score = samples[:, :columns] @ design.T
        cuts = np.column_stack([np.full(len(samples), -np.inf), np.zeros(len(samples)), samples[:, columns:],
                                np.full(len(samples), np.inf)])
        chance = (sigmoid(cuts[:, codes + 1] - score) - sigmoid(cuts[:, codes] - score))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((chance > 0).all(axis=1), np.log(np.maximum(chance, 1e-300)).sum(axis=1), -np.inf)
    return log_likelihood


def category_likelihood(design, y, node):
    """One score per category: 0 for the most common, and design @ its numbers
    for the others, in the order the node lays its numbers out."""
    order = list(node.others)

    def log_likelihood(samples):
        scores = np.stack([np.zeros((len(samples), len(y)))] +
                          [samples[:, node._layout[row]] @ design.T for row in range(len(order))], axis=2)
        picked = np.array([0 if level == node.reference else order.index(level) + 1 for level in y])
        chosen = np.take_along_axis(scores, np.broadcast_to(picked[None, :, None], scores.shape[:2] + (1,)), 2)[..., 0]
        return (chosen - np.logaddexp.reduce(scores, axis=2)).sum(axis=1)
    return log_likelihood


# --- Then ------------------------------------------------------------------------

@then(parsers.parse("the evidence with x is higher by more than {margin:g}"))
def prefers_true_parent(context, margin):
    assert context["with x"] > context["with w"] + margin


@then(parsers.parse("importance sampling with {draws:d} draws gives the same value within {tolerance:g}"))
def matches_importance_sampling(context, draws, tolerance):
    node, design, y, fit = context["node"], context["design"], context["data"]["y"], context["fit"]
    likelihood = {"yes/no": lambda: binary_likelihood(design, y),
                  "ordered": lambda: ordered_likelihood(design, y, node.levels),
                  "category": lambda: category_likelihood(design, y, node)}[context["kind"]]()
    sampled = ImportanceSampling(draws).log_evidence(likelihood, context["penalty"], fit.coefficients, fit.covariance)
    assert abs(fit.evidence - sampled) < tolerance


@then("both evidences are equal")
def same_evidence(context):
    first, second = context["nodes"]
    assert abs(first.evidence - second.evidence) < 1e-6


@then(parsers.parse("its evidence is finite and no height is beyond {limit:g} log-odds"))
def finite_fit(context, limit):
    node = context["node"]
    assert np.isfinite(node.evidence)
    assert abs(node.constant) < limit and all(np.abs(curve.heights).max() < limit for curve in node.curves.values())


@then("its levels are 0, 1 and 3, with two cut-points")
def levels_dropped(context):
    node = context["node"]
    assert node.levels.tolist() == [0, 1, 3] and len(node.cut_points) == 2
    assert np.isclose(node.probabilities(context["data"]).sum(axis=1), 1).all()


@then("c scores 0 with any parent set")
def constant_skipped(context):
    score = EvidenceScore(context["data"], maker({"c": "nominal"}))
    assert score("c", []) == score("c", ["x"]) == score("c", ["x", "w"]) == 0.0
    assert score("x", ["c"]) == score("x", [])  # as a parent it is left out


@then("a category never seen gets the average of the seen categories' scores")
def unseen_category(context):
    node = context["nodes"][0]
    scores = node.scores(context["data"], levels=[1.0, 2.0, 3.0, 5.0])
    assert np.allclose(scores[:, 2], scores[:, [0, 1, 3]].mean(axis=1))
    assert np.allclose(node.probabilities(context["data"]).sum(axis=1), 1.0)
