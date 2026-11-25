"""Kairos, its KaCGM-like cut-downs and PC on the two downloaded CausalArena scenarios (rep 0, both regimes).

From kairos/:  venv/bin/python -I diagnostics/kacgm/smoke.py <method>   (kairos, sums-only, resit, resit-surfaces, pc-fisherz)
"""
import sys
import time
import warnings
import zlib
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE), str(HERE.parent / "worlds")]

from resit import Resit  # noqa: E402
from run import PartSearch  # noqa: E402
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, GraphScore, ScenarioIndex  # noqa: E402
from src.benchmarks.causal_arena.methods import KairosInputs, PCSearch  # noqa: E402
from src.causal_model.discovery import Background  # noqa: E402

method = sys.argv[1]
files = ArenaFiles(HERE.parents[1] / "data" / "causal_arena")
total = []
for word in ("mortgage", "medication"):
    scenario = next(s for s in ScenarioIndex.load(files) if word in s.slug)
    for regime in ("obs_only", "obs_int"):
        data = ArenaData(files).load(scenario, regime, 0)
        inputs = KairosInputs(data)
        for background in ("no background", "roots given"):
            roots = data.roots if background == "roots given" else set()
            start = time.perf_counter()
            if method == "pc-fisherz":
                if roots:
                    continue
                found = PCSearch("fisherz").find(data, set()).oriented(data.truth, "random", zlib.crc32(data.key.encode()))
            elif method.startswith("resit"):
                found = Resit(Background(roots=roots, kinds=inputs.kinds), surfaces=method == "resit-surfaces").arrows(inputs.table())
            else:
                search = PartSearch(Background(roots=roots, kinds=inputs.kinds))
                search.variant = {"kairos": "full"}.get(method, method)
                found = set(search.run(inputs.table(), inputs.intervened()).arrows())
            score = GraphScore.compare(found, data.truth)
            total.append(score.f1)
            print(f"{method:15s} {word:10s} {regime:8s} {background:13s} F1 {score.f1:.2f} (correct {score.correct}, reversed {score.reversed}, "
                  f"extra {score.extra}, missing {score.missing}) {time.perf_counter() - start:.0f}s", flush=True)
print(f"{method:15s} average F1 {sum(total) / len(total):.3f} over {len(total)} runs")
