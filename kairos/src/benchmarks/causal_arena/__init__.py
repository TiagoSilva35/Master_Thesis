"""Causal discovery on CausalArena (Li, Liu, Wang and Ye, 2026): the baselines on
its public semantic SCMs, scored against their true graphs.

    files.py       downloads a repository file the first time it is needed
    scenarios.py   the 50 public scenarios (10 domains), from the release manifest
    data.py        one sampled file as ScenarioData: values, true graph, kinds, C
    methods/       the baselines, each with find(data, roots) -> FoundGraph
    found.py       what a method returned (FoundGraph), and how undirected pairs are oriented
    scoring.py     precision, recall, F1, SHD (GraphScore)
    benchmark.py   one file × every method × background -> Result
    report.py      summary.txt and results.csv
    run.py         the command line, and the list of baselines
"""

from src.benchmarks.causal_arena.benchmark import ArenaBenchmark, Result
from src.benchmarks.causal_arena.data import ArenaData, ScenarioData
from src.benchmarks.causal_arena.files import ArenaFiles
from src.benchmarks.causal_arena.found import FoundGraph
from src.benchmarks.causal_arena.report import Report
from src.benchmarks.causal_arena.scenarios import PEOPLE_DOMAINS, REGIMES, Scenario, ScenarioIndex
from src.benchmarks.causal_arena.scoring import GraphScore

__all__ = ["ArenaBenchmark", "Result", "ArenaData", "ScenarioData", "ArenaFiles", "FoundGraph", "Report",
           "PEOPLE_DOMAINS", "REGIMES", "Scenario", "ScenarioIndex", "GraphScore"]
