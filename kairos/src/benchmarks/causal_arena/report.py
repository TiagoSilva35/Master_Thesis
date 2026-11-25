import csv
from pathlib import Path

import numpy as np

from src.benchmarks.causal_arena.benchmark import Result
from src.benchmarks.causal_arena.scenarios import PEOPLE_DOMAINS, REGIMES

SHORT = {"cybersecurity_and_it_operations": "cyber", "education": "educ", "finance_and_credit": "finance",
         "government_services_and_policy_implementation": "gov", "healthcare_delivery": "health",
         "housing_and_real_estate": "housing", "manufacturing": "manuf", "public_health": "pub.h",
         "urban_transportation": "transp", "water_and_sanitation": "water"}


class Report:
    """The results as text (per regime: means over every file, means over the
    people domains, mean F1 by domain, failures) and as a CSV with one row per run.
    Scores follow CausalArena's protocol; best/worst bounds are in the CSV."""

    COLUMNS = ["domain", "scenario", "regime", "replicate", "method", "background", "f1", "precision", "recall",
               "shd", "precision_best", "precision_worst", "recall_best", "recall_worst", "correct", "turned",
               "extra", "missed", "undirected", "rows", "seconds", "error"]

    def __init__(self, results: list[Result]):
        self.results = results

    def save(self, directory: Path) -> None:
        """summary.txt and results.csv in `directory`."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.txt").write_text(self.tables())
        with open(directory / "results.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, self.COLUMNS)
            writer.writeheader()
            writer.writerows(self._row(result) for result in self.results)

    def tables(self) -> str:
        lines = ["Causal discovery on CausalArena's semantic SCMs (Li et al., 2026).",
                 "Directed-edge F1, precision, recall and SHD, CausalArena's protocol (undirected pairs",
                 "pointed at random, one seed per file); means over files, ± the spread of F1.", ""]
        for regime in dict.fromkeys(result.regime for result in self.results):
            here = [result for result in self.results if result.regime == regime]
            lines += [f"Regime {regime}: {REGIMES[regime]}", ""]
            lines += self._means(here, "every scenario")
            lines += self._means([r for r in here if r.scenario.domain in PEOPLE_DOMAINS], "people domains only")
            lines += self._by_domain(here)
            lines += self._failures(here)
        return "\n".join(lines)

    @staticmethod
    def line(result: Result) -> str:
        """One line per finished run, for progress."""
        if result.found is None:
            return f"{result.key} · {result.method} · {result.background}: failed ({result.error})"
        score = result.score
        return (f"{result.key} · {result.method} · {result.background}: F1 {score.f1:.2f} "
                f"(precision {score.precision:.2f}, recall {score.recall:.2f}), {result.graph.seconds:.0f}s")

    def _means(self, results: list[Result], title: str) -> list[str]:
        if not results:
            return []
        lines = [f"  {title} ({len({r.key for r in results})} files)",
                 f"    {'method':18} {'background':15} {'runs':>4} {'failed':>6} {'F1':>11} {'precision':>9} "
                 f"{'recall':>6} {'SHD':>5} {'time':>6}"]
        for method, background in dict.fromkeys((r.method, r.background) for r in results):
            group = [r for r in results if r.method == method and r.background == background]
            scored = [r.score for r in group if r.found is not None]
            seconds = [r.graph.seconds for r in group if r.found is not None]
            if not scored:
                lines.append(f"    {method:18} {background:15} {len(group):4d} {len(group):6d}")
                continue
            f1 = np.array([s.f1 for s in scored])
            lines.append(f"    {method:18} {background:15} {len(group):4d} {len(group) - len(scored):6d} "
                         f"{f1.mean():5.2f} ±{f1.std():4.2f} {np.mean([s.precision for s in scored]):9.2f} "
                         f"{np.mean([s.recall for s in scored]):6.2f} {np.mean([s.shd for s in scored]):5.1f} "
                         f"{np.mean(seconds):5.1f}s")
        return lines + [""]

    def _by_domain(self, results: list[Result]) -> list[str]:
        domains = list(dict.fromkeys(r.scenario.domain for r in results))
        lines = ["  mean F1 by domain (* people domains)",
                 "    " + f"{'method':18} {'background':15} "
                 + " ".join(f"{SHORT.get(d, d[:7]) + ('*' if d in PEOPLE_DOMAINS else ''):>8}" for d in domains)]
        for method, background in dict.fromkeys((r.method, r.background) for r in results):
            cells = []
            for domain in domains:
                f1 = [r.score.f1 for r in results if r.method == method and r.background == background
                      and r.scenario.domain == domain and r.found is not None]
                cells.append(f"{np.mean(f1):8.2f}" if f1 else f"{'-':>8}")
            lines.append(f"    {method:18} {background:15} " + " ".join(cells))
        return lines + [""]

    @staticmethod
    def _failures(results: list[Result]) -> list[str]:
        failed = [r for r in results if r.found is None]
        if not failed:
            return []
        errors: dict[tuple[str, str], int] = {}
        for result in failed:
            errors[(result.method, result.error)] = errors.get((result.method, result.error), 0) + 1
        return (["  failures"] + [f"    {method}: {count}× {error}" for (method, error), count in errors.items()]
                + [""])

    @staticmethod
    def _row(result: Result) -> dict:
        row = {"domain": result.scenario.domain, "scenario": result.scenario.slug, "regime": result.regime,
               "replicate": result.replicate, "method": result.method, "background": result.background,
               "error": result.error}
        if result.found is not None:
            score, best, worst = result.score, result.best, result.worst
            row |= {"f1": round(score.f1, 4), "precision": round(score.precision, 4),
                    "recall": round(score.recall, 4), "shd": score.shd,
                    "precision_best": round(best.precision, 4), "precision_worst": round(worst.precision, 4),
                    "recall_best": round(best.recall, 4), "recall_worst": round(worst.recall, 4),
                    "correct": score.correct, "turned": score.reversed, "extra": score.extra,
                    "missed": score.missing, "undirected": len(result.found.undirected),
                    "rows": result.found.rows, "seconds": round(result.found.seconds, 1)}
        return row
