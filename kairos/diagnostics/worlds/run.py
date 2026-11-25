"""Kairos, and parts of it, on made-up worlds whose true graph is known.

From kairos/:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
               venv/bin/python -I diagnostics/worlds/run.py [first seed] [last seed] [variants...]
Variants: full, no-scales, no-surfaces, bell-noise, no-floor, sums-only (a sum of curves, bell noise),
resit, resit-surfaces (the causal order by independence of the leftovers, see ../kacgm/resit.py)
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE), str(HERE.parent / "kacgm")]

import numpy as np  # noqa: E402

import special  # noqa: E402
from general import make  # noqa: E402
from resit import Resit  # noqa: E402
from src.causal_model.discovery import Background, Graph, GraphSearch, NodeMaker  # noqa: E402
from src.causal_model.nodes import SCALES  # noqa: E402

BEDS = ("general", "formulas", "mild", "skewed", "sizes")


class PartMaker(NodeMaker):
    variant = "full"

    def scales(self, name, values):
        return [SCALES[0]] if self.variant in ("no-scales", "sums-only") else super().scales(name, values)

    def make(self, name, parents, scale=None):
        node = super().make(name, parents, scale)
        if self.variant in ("no-surfaces", "sums-only"):
            node.screening = None
        if self.variant in ("bell-noise", "sums-only"):
            node.read_noise = lambda fit, data: None
        if self.variant == "no-floor":
            node.noise_floor = 0.0
        return node


class PartSearch(GraphSearch):
    variant = "full"

    def maker(self):
        maker = PartMaker(self.background, self.intervals)
        maker.variant = self.variant
        return maker


def world(bed, seed):
    if bed == "general":
        data, kinds, _, truth, _ = make(seed)
        return data, kinds, truth
    data, truth = getattr(special, bed)(seed)
    return data, {name: "continuous" for name in data}, truth


def f1(found, truth):
    return 2 * len(found & truth) / max(len(found) + len(truth), 1)


def one(task):
    variant, bed, seed = task
    data, kinds, parents = world(bed, seed)
    truth = {(parent, child) for child, ps in parents.items() for parent in ps}
    start = time.perf_counter()
    if variant.startswith("resit"):
        found = Resit(Background(kinds=kinds), surfaces=variant == "resit-surfaces").arrows(data)
        return f1(found, truth), np.nan, np.nan, time.perf_counter() - start
    search = PartSearch(Background(kinds=kinds))
    search.variant = variant
    score = search.score(data)
    graph = search.best_graph(sorted(data), score)
    seconds = time.perf_counter() - start
    true_graph = Graph(sorted(data))
    for parent, child in sorted(truth):
        true_graph.add(parent, child)
    near = search.climb(true_graph, score)
    return (f1(set(graph.arrows()), truth), f1(set(near.arrows()), truth),
            search.worth(graph, score) - search.worth(true_graph, score), seconds)


if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    variants = sys.argv[3:] or ["full"]
    with Pool(9) as pool:
        for bed in BEDS:
            for variant in variants:
                rows = np.array(pool.map(one, [(variant, bed, seed) for seed in range(first, last)], chunksize=1))
                error = rows[:, 0].std() / np.sqrt(len(rows))
                print(f"{bed:9s} {variant:12s} F1 {rows[:, 0].mean():.3f} (± {error:.3f})  "
                      f"climbing from the truth {rows[:, 1].mean():.3f}  found − truth {np.median(rows[:, 2]):8.0f}  "
                      f"{rows[:, 3].mean():.1f}s", flush=True)
