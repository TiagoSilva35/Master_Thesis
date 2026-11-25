"""Run the baselines on CausalArena's public semantic SCMs and write the results.

From kairos/:
    PYTHONPATH=. venv/bin/python -m src.benchmarks.causal_arena.run
    PYTHONPATH=. venv/bin/python -m src.benchmarks.causal_arena.run --domains people --replicates 1
    PYTHONPATH=. venv/bin/python -m src.benchmarks.causal_arena.run --domains finance_and_credit --limit 1 \\
        --replicates 1 --regimes obs_only --methods kairos pc-fisherz

Files are downloaded from Hugging Face the first time a run needs them, into --data
(about 50 MB for everything). summary.txt and results.csv are rewritten in --out
after each file. --workers runs that many files at once; a run started again with the
same --out and --methods skips the files it has finished (see parallel.py).
cdnod-authors is not in the default run: several hours per file.
"""

import argparse
from pathlib import Path

from src.benchmarks.causal_arena.benchmark import ArenaBenchmark
from src.benchmarks.causal_arena.data import ArenaData
from src.benchmarks.causal_arena.files import ArenaFiles
from src.benchmarks.causal_arena.methods import (AuthorsCDNODSearch, CDNODSearch, DiscoveryMethod, KairosSearch,
                                                 PCSearch)
from src.benchmarks.causal_arena.parallel import ParallelRun
from src.benchmarks.causal_arena.report import Report
from src.benchmarks.causal_arena.scenarios import PEOPLE_DOMAINS, REGIMES, Scenario, ScenarioIndex


def baselines() -> dict[str, DiscoveryMethod]:
    """Every baseline, by the short name the command line uses. To add one, write a
    class as in methods/base.py and add it here."""
    return {
        "kairos": KairosSearch(),
        "pc-fisherz": PCSearch("fisherz"),
        "pc-rcit": PCSearch("rcit"),
        "cdnod-fisherz": CDNODSearch("fisherz"),
        "cdnod-rcit": CDNODSearch("rcit"),
        "cdnod-authors": AuthorsCDNODSearch(),
    }


DEFAULT_METHODS = ["kairos", "pc-fisherz", "pc-rcit", "cdnod-fisherz", "cdnod-rcit"]


def chosen_scenarios(scenarios: list[Scenario], domains: list[str], limit: int | None) -> list[Scenario]:
    """The scenarios of the chosen domains ("people" = the five people domains), at
    most `limit` per domain."""
    wanted = set(PEOPLE_DOMAINS) if domains == ["people"] else set(domains)
    known = {scenario.domain for scenario in scenarios}
    if not wanted <= known and domains != ["all"]:
        raise SystemExit(f"unknown domains {sorted(wanted - known)}; choose from {sorted(known)}, all or people")
    picked, per_domain = [], {}
    for scenario in scenarios:
        if domains == ["all"] or scenario.domain in wanted:
            per_domain[scenario.domain] = per_domain.get(scenario.domain, 0) + 1
            if limit is None or per_domain[scenario.domain] <= limit:
                picked.append(scenario)
    return picked


def main() -> None:
    every = baselines()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domains", nargs="+", default=["all"], help="domain names, 'people' or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="at most this many scenarios per domain")
    parser.add_argument("--regimes", nargs="+", choices=list(REGIMES), default=list(REGIMES))
    parser.add_argument("--replicates", type=int, default=5, help="replicates per scenario and regime (up to 5)")
    parser.add_argument("--methods", nargs="+", choices=list(every), default=DEFAULT_METHODS)
    parser.add_argument("--data", type=Path, default=Path("data/causal_arena"), help="where files are kept")
    parser.add_argument("--out", type=Path, default=Path("figures/benchmarks/causal_arena"))
    parser.add_argument("--workers", type=int, default=1, help="how many files to run at once")
    args = parser.parse_args()

    files = ArenaFiles(args.data)
    scenarios = chosen_scenarios(ScenarioIndex.load(files), args.domains, args.limit)
    benchmark = ArenaBenchmark(ArenaData(files), [every[name] for name in args.methods])
    tasks = [(scenario, regime, replicate) for scenario in scenarios for regime in args.regimes
             for replicate in range(min(args.replicates, scenario.replicates(regime)))]
    results = []
    kept = args.out / "files" / "+".join(args.methods)  # other methods, other files
    for done, found in enumerate(ParallelRun(benchmark, kept, args.workers).results(tasks), 1):
        results += found
        for result in found:
            print(Report.line(result))
        print(f"--- {done} of {len(tasks)} files", flush=True)
        Report(results).save(args.out)
    print(Report(results).tables())
    print(f"saved {args.out / 'summary.txt'} and {args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
