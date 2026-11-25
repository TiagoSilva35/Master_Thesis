"""Steps for noise_floor.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting, Node, NoiseFloor

scenarios("noise_floor.feature")

FORMULAS = (lambda x: 2 * x, np.exp, lambda x: x + x**3, lambda x: np.tanh(2 * x), lambda x: np.sqrt(np.maximum(x + 2, 0.02)))


class OwnShareMaker(NodeMaker):
    """Makes nodes whose floor is only the child's own share, for comparison."""

    def make(self, name, parents, screening=None, for_scoring=True):
        node = super().make(name, parents, screening, for_scoring)
        node.carried_floor = lambda data: NoiseFloor(node.noise_floor)
        return node


def score_of(data: dict, noise_floor: float, learn_noise: bool = False, maker=NodeMaker) -> EvidenceScore:
    return EvidenceScore(data, maker({}, None, BayesianFitting(), intervals=8, noise_floor=noise_floor,
                                     learn_noise=learn_noise))


def gain(data: dict, child: str, parents: list[str], noise_floor: float) -> float:
    score = score_of(data, noise_floor)
    return score(child, parents) - score(child, [])


def draw(shape: str, random: np.random.Generator, count: int) -> np.ndarray:
    return random.normal(0, 1, count) if shape == "bell" else random.uniform(-1.7, 1.7, count)


def fitted(name: str, parents: list[str], data: dict, noise_floor: float, **options) -> Node:
    node = Node(name, parents, pairs=options.pop("pairs", []), fitting=BayesianFitting(), noise_floor=noise_floor, **options)
    node.fit(data)
    return node


def wrong_minus_true(score: EvidenceScore) -> float:
    """How far the better of the two rewritten graphs is ahead of x → z ← y."""
    true = score("x", []) + score("y", []) + score("z", ["x", "y"])
    solved_for_x = score("z", []) + score("y", ["z"]) + score("x", ["y", "z"])
    solved_for_y = score("z", []) + score("x", ["z"]) + score("y", ["x", "z"])
    return max(solved_for_x, solved_for_y) - true


@given(parsers.parse("{count:d} people where z is x minus w, rounded to 0.01"))
def rounded_formula(context, count):
    random = np.random.default_rng(0)
    x, w = random.normal(0, 1, count), random.normal(0, 1, count)
    context["data"] = {"x": x, "w": w, "z": np.round(x - w, 2)}


@given(parsers.parse("{count:d} people where y = sin(x) plus bell-shaped noise"))
def sine_people(context, count):
    random = np.random.default_rng(0)
    x = random.uniform(0, 4, count)
    context["data"] = {"x": x, "y": np.sin(x) + random.normal(0, 0.3, count)}


@given(parsers.parse("{count:d} people where z is exactly 2 times x minus w"))
def exact_formula(context, count):
    random = np.random.default_rng(0)
    x, w = random.normal(0, 1, count), random.normal(0, 3, count)
    context["data"] = {"x": x, "w": w, "z": 2 * x - w}


@given(parsers.parse("{count:d} people where y is 2 higher in group 1 than in group 0, plus noise"))
def groups(context, count):
    random = np.random.default_rng(0)
    group = random.integers(0, 2, count).astype(float)
    context["data"] = {"group": group, "y": 2 * group + random.normal(0, 1, count)}


@given(parsers.parse("{count:d} people where z is exactly {a:d} times x minus {b:d} times y, with {shape} x and y"))
def scaled_formula(context, count, a, b, shape):
    random = np.random.default_rng(0)
    x, y = draw(shape, random, count), draw(shape, random, count)
    context["data"] = {"x": x, "y": y, "z": a * x - b * y}


@given("pairs y = f(x) plus noise of 20% of y's spread, for 5 formulas, 2 shapes and 3 seeds")
def ordinary_pairs(context):
    context["pairs"] = []
    for shape in ("bell", "flat"):
        for formula in FORMULAS:
            for seed in range(3):
                random = np.random.default_rng(seed)
                x = np.clip(draw(shape, random, 1000), -3, 3)
                y = formula(x)
                context["pairs"].append({"x": x, "y": y + 0.2 * np.std(y) * draw(shape, random, 1000)})


@given("a node with two curves and a surface, fitted with noise floor 0.0001")
def surface_node(context):
    random = np.random.default_rng(0)
    x, w = random.uniform(-1, 1, 400), random.uniform(0, 5, 400)
    data = {"x": x, "w": w, "y": np.sin(2 * x) + 0.3 * w + 0.5 * x * w + random.normal(0, 0.1, 400)}
    context["data"] = data
    context["node"] = fitted("y", ["x", "w"], data, 0.0001, pairs=[("x", "w")])


@when(parsers.parse("I score z with and without its parents x and w, with noise floor {share:g} and with none"))
def score_formula(context, share):
    context["with floor"] = gain(context["data"], "z", ["w", "x"], share)
    context["without floor"] = gain(context["data"], "z", ["w", "x"], 0.0)


@when(parsers.parse("I score y with and without its parent x, with noise floor {share:g} and with none"))
def score_sine(context, share):
    context["with floor"] = gain(context["data"], "y", ["x"], share)
    context["without floor"] = gain(context["data"], "y", ["x"], 0.0)


@when("I fit z on x and w with noise floor 0.01")
def fit_formula(context):
    context["node"] = fitted("z", ["x", "w"], context["data"], 0.01)


@when("I fit z on x and w, and x on z and w, with noise floor 0.01")
def fit_both(context):
    context["z"] = fitted("z", ["x", "w"], context["data"], 0.01)
    context["x"] = fitted("x", ["z", "w"], context["data"], 0.01)


@when("I fit y on its group with noise floor 0.01")
def fit_groups(context):
    context["node"] = fitted("y", ["group"], context["data"], 0.01, kinds={"group": "binary"})


@when(parsers.parse("I score the true graph and the two graphs that solve the formula for x or y, with {kind} noise"))
def score_graphs(context, kind):
    learn_noise = kind == "learnt"
    context["own"] = wrong_minus_true(score_of(context["data"], 0.01, learn_noise, OwnShareMaker))
    context["carried"] = {share: wrong_minus_true(score_of(context["data"], share, learn_noise))
                          for share in (0.01, 0.0001)}


@when("I score both directions of every pair with learnt noise and noise floor 0.0001")
def score_pairs(context):
    context["gaps"] = []
    for data in context["pairs"]:
        score = score_of(data, 0.0001, learn_noise=True)
        context["gaps"].append(score("x", []) + score("y", ["x"]) - score("y", []) - score("x", ["y"]))


def cap(context) -> float:
    return len(context["data"]["z"]) / 2 * np.log(1.01 / 0.01)


@then("with the floor z's gain is at most N/2 · log(1.01 / 0.01)")
def capped(context):
    assert context["with floor"] <= cap(context)


@then("without the floor z's gain is more than 1.5 times that")
def uncapped(context):
    assert context["without floor"] > 1.5 * cap(context)


@then("with the floor y keeps at least 99% of its gain")
def most_kept(context):
    assert context["with floor"] >= 0.99 * context["without floor"]


@then("a node with noise floor 0 has the same evidence as a node without one")
def same_evidence(context):
    plain = Node("y", ["x"], pairs=[], fitting=BayesianFitting())
    floored = Node("y", ["x"], pairs=[], fitting=BayesianFitting(), noise_floor=0.0)
    plain.fit(context["data"])
    floored.fit(context["data"])
    assert plain.evidence == floored.evidence


@then("each row's floor is 1% of var(z) + 4 var(x) + var(w), to within 1%")
def carried_sum(context):
    data = context["data"]
    expected = 0.01 * (np.var(data["z"]) + 4 * np.var(data["x"]) + np.var(data["w"]))
    assert np.allclose(context["node"].row_floor, expected, rtol=0.01)


@then("z's floor is 4 times x's floor, to within 1%, because z moves 2 when x moves 1")
def same_floor(context):
    assert np.isclose(context["z"].row_floor.mean(), 4 * context["x"].row_floor.mean(), rtol=0.01)


@then("each row's floor is 1% of var(y)")
def own_only(context):
    assert np.allclose(context["node"].row_floor, 0.01 * np.var(context["data"]["y"]))


@then(parsers.parse("with only the child's own share, 0.01, a wrong graph beats the true one by more than {margin:g}"))
def own_wrong(context, margin):
    assert context["own"] > margin


@then(parsers.parse("with the carried floor of share {share:g}, no wrong graph beats the true one by more than {margin:g}"))
def carried_right(context, share, margin):
    assert context["carried"][share] <= margin


@then("at least 90% of the pairs point the right way")
def right_way(context):
    assert np.mean(np.array(context["gaps"]) > 3) >= 0.9


@then("its slopes match the change in the prediction for a small step, along each parent")
def slopes_match(context):
    node, data, step = context["node"], context["data"], 1e-5
    for parent in ("x", "w"):
        low = {**data, parent: data[parent] - step}
        high = {**data, parent: data[parent] + step}
        numeric = (node.predict(high) - node.predict(low)) / (2 * step)
        curve = node.curves[parent].slopes(data[parent]) @ node.curves[parent].heights
        surface = node.surfaces[("x", "w")]
        free = np.linalg.lstsq(surface.free, surface.heights, rcond=None)[0]
        along = surface.slopes(data["x"], data["w"])[("x", "w").index(parent)] @ free
        assert np.allclose(curve + along, numeric, atol=1e-4)
