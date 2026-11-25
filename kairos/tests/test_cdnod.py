"""Steps for cdnod.feature."""

import numpy as np
from causallearn.utils.KCI.KCI import KCI_UInd
from pytest_bdd import given, parsers, scenarios, then, when

from src.baselines.cdnod import CDNOD, CDNODGraphs, CDNODSettings
from src.baselines.cdnod.gaussian_process import GaussianProcessWidths
from src.baselines.cdnod.independent_change import IndependentChangeOrientation
from src.baselines.cdnod.kci import KernelCITest
from src.baselines.cdnod.kci_rff import RandomFeatureCITest
from src.baselines.cdnod.meek import MeekRules
from src.baselines.cdnod.minimize import ConjugateGradients

scenarios("cdnod.feature")


def arrows(text: str) -> set[tuple[str, str]]:
    """'a → b and c → d' -> {('a', 'b'), ('c', 'd')}"""
    found = set()
    for part in text.replace(" and ", ", ").split(","):
        parent, child = part.split("→")
        found.add((parent.strip(), child.strip()))
    return found


def widths() -> GaussianProcessWidths:
    return GaussianProcessWidths(ConjugateGradients(length=-350))


# --- minimize.m and gpr_multi.m -------------------------------------------------

def rosenbrock(point: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = point
    value = (1 - a) ** 2 + 100 * (b - a * a) ** 2
    return value, np.array([-2 * (1 - a) - 400 * a * (b - a * a), 200 * (b - a * a)])


@when("minimize.m's port starts the Rosenbrock function at (-1, 1)")
def minimize_rosenbrock(context):
    context["found"] = ConjugateGradients(length=-350).minimize(rosenbrock, np.array([-1.0, 1.0]))


@then("it stops at (1, 1)")
def at_minimum(context):
    assert np.allclose(context["found"], [1.0, 1.0], atol=1e-4)


@given(parsers.parse("{rows:d} rows with {inputs:d} inputs and {outputs:d} outputs"))
def regression_data(context, rows, inputs, outputs):
    random = np.random.default_rng(1)
    context["inputs"] = random.normal(size=(rows, inputs))
    context["outputs"] = random.normal(size=(rows, outputs))
    context["log_hyper"] = np.array([0.1, -0.3, 0.2, np.log(0.4)])


@then("the gradient of the negative log likelihood matches finite differences")
def gradient_matches(context):
    def value(h):
        return GaussianProcessWidths.negative_log_likelihood(h, context["inputs"], context["outputs"])[0]
    h = context["log_hyper"]
    _, gradient = GaussianProcessWidths.negative_log_likelihood(h, context["inputs"], context["outputs"])
    numeric = [(value(h + step) - value(h - step)) / 2e-6 for step in np.eye(len(h)) * 1e-6]
    assert np.allclose(gradient, numeric, atol=1e-5)


# --- KCI ---------------------------------------------------------------------

@given(parsers.parse("{rows:d} rows where y = sin(2x) plus noise"))
def sine_pair(context, rows):
    random = np.random.default_rng(0)
    context["x"] = random.normal(size=(rows, 1))
    context["y"] = np.sin(2 * context["x"]) + 0.5 * random.normal(size=(rows, 1))


@then("the unconditional KCI statistic equals causal-learn's")
def same_statistic(context):
    test = KernelCITest(CDNODSettings(), np.random.default_rng(0), widths())
    kx, ky = test.unconditional_kernels(context["x"], context["y"])
    _, theirs = KCI_UInd(approx=True).compute_pvalue(context["x"], context["y"])
    assert np.isclose(np.trace(kx @ ky), theirs, rtol=1e-9)


@given("300 rows of a chain a → b → c")
def chain(context):
    random = np.random.default_rng(0)
    a = random.normal(size=(300, 1))
    b = a + 0.5 * random.normal(size=(300, 1))
    context["chain"] = (a, b, b + 0.5 * random.normal(size=(300, 1)))


@when(parsers.parse("the {kind} KCI test runs"))
def run_kci(context, kind):
    settings, random = CDNODSettings(gp_tests=False), np.random.default_rng(0)
    test = KernelCITest(settings, random, widths()) if kind == "exact" else RandomFeatureCITest(settings, random)
    a, b, c = context["chain"]
    context["p_values"] = (test.p_value(a, c, a[:, :0]), test.p_value(a, c, b))


@then("a and c are dependent, but independent given b")
def separated_by_middle(context):
    marginal, given_b = context["p_values"]
    assert marginal < 0.05 < given_b


# --- graph steps ---------------------------------------------------------------

@given("the graph a → b — c with a and c not linked")
def meek_input(context):
    # meeks.m's coding: -1 at (a, b) is a → b; 1 both ways is undirected
    context["pdag"] = np.array([[0, -1, 0], [0, 0, 1], [0, 1, 0]])
    context["adjacency"] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@when("Meek's rules run")
def run_meek(context):
    context["pdag"] = MeekRules.apply(context["pdag"], context["adjacency"])


@then("b — c becomes b → c")
def continued(context):
    assert context["pdag"][1, 2] == -1 and context["pdag"][2, 1] == 0


@given("400 rows over time where x1 → x2 → x3 and only x2's mechanism changes")
def changing_middle(context):
    random, time = np.random.default_rng(0), np.arange(1, 401)
    x1 = random.normal(size=400)
    x2 = 0.8 * x1 + 3 * np.sin(time / 50) + 0.5 * random.normal(size=400)
    x3 = 0.8 * x2 + 0.5 * random.normal(size=400)
    context["data"], context["time"] = np.column_stack([x1, x2, x3]), time
    context["names"] = ["x1", "x2", "x3"]


@when("CD-NOD runs with fixed kernel widths")
def run_cdnod(context):
    settings = CDNODSettings(gp_tests=False, gp_directions=False)
    context["read"] = CDNOD(settings).run(context["data"], context["time"]).read(context["names"])


@given("the undirected chain x1 — x2 — x3 where all three mechanisms change")
def changing_chain(context):
    # CD-NOD's coding, C last: -1 both ways is undirected; C → x1, x2, x3
    graph = np.array([[0, -1, 0, 0], [-1, 0, -1, 0], [0, -1, 0, 0], [1, 1, 1, 0]], dtype=float)
    context["graph"], context["names"] = graph, ["x1", "x2", "x3"]
    context["data"] = np.tile(np.arange(4.0), (5, 1))  # column k holds k, so a score can tell them apart


@when(parsers.parse("the Independent Change Principle step runs with a score that ranks {order} as most independent"))
def run_independent_change(context, order):
    ranking = [context["names"].index(name.strip()) for name in order.split(",")]

    def score(causes, effect, c_index):  # stands in for infer_nonsta_dir.m
        return float(ranking.index(int(effect[0, 0])))

    graph = context["graph"]
    final = IndependentChangeOrientation(score).run(graph, graph.copy(), context["data"], np.arange(5.0)[:, None])
    context["read"] = CDNODGraphs(graph, graph, final, []).read(context["names"])


@then(parsers.parse("the arrows found are {text}"))
def arrows_found(context, text):
    assert context["read"][0] == arrows(text)


# --- settings ------------------------------------------------------------------

@then("1000 rows use KCI with widths learnt by a Gaussian process")
def small_sample():
    settings = CDNODSettings.for_rows(1000)
    assert settings.gp_tests and not settings.random_features


@then("2000 rows use KCI with fixed widths")
def medium_sample():
    settings = CDNODSettings.for_rows(2000)
    assert not settings.gp_tests and not settings.random_features


@then("2001 rows use the random-feature KCI with fixed widths")
def large_sample():
    settings = CDNODSettings.for_rows(2001)
    assert not settings.gp_tests and settings.random_features
