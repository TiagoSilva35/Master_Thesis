"""Joins the results of a parallel CausalArena run (scripts/run_causal_arena_tmux.sh):
every job's results.csv into one results.csv, and a summary.txt of mean scores.
It can be run at any time, also while jobs are still running (run.py rewrites a
job's results.csv after each scenario).

From kairos/:
    PYTHONPATH=. venv/bin/python scripts/merge_causal_arena.py figures/benchmarks/causal_arena/<run>
"""

import csv
import sys
from pathlib import Path

import numpy as np

from src.benchmarks.causal_arena.scenarios import PEOPLE_DOMAINS, REGIMES


class RunMerger:
    """The results of one parallel run, kept in <folder>/jobs/<domain>-<regime>/results.csv."""

    def __init__(self, folder: Path):
        self.folder = folder

    def rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for path in sorted(self.folder.glob("jobs/*/results.csv")):
            with open(path, newline="") as file:
                rows += list(csv.DictReader(file))
        return rows

    def save(self) -> str:
        """Write results.csv and summary.txt in the run's folder; return the summary."""
        rows = self.rows()
        if not rows:
            raise SystemExit(f"no results yet under {self.folder / 'jobs'}")
        columns = list(dict.fromkeys(column for row in rows for column in row))
        with open(self.folder / "results.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, columns)
            writer.writeheader()
            writer.writerows(rows)
        summary = self.summary(rows)
        (self.folder / "summary.txt").write_text(summary)
        return summary

    def summary(self, rows: list[dict[str, str]]) -> str:
        lines = ["Causal discovery on CausalArena's semantic SCMs (Li et al., 2026), joined from a parallel run.",
                 "Directed-edge F1 (± its spread), precision, recall and SHD, CausalArena's protocol; means over files.",
                 ""]
        for regime in dict.fromkeys(row["regime"] for row in rows):
            here = [row for row in rows if row["regime"] == regime]
            lines += [f"Regime {regime}: {REGIMES.get(regime, regime)}", ""]
            lines += self._table(here, "every scenario")
            lines += self._table([row for row in here if row["domain"] in PEOPLE_DOMAINS], "people domains only")
        return "\n".join(lines)

    @staticmethod
    def _table(rows: list[dict[str, str]], title: str) -> list[str]:
        if not rows:
            return []
        files = {(row["domain"], row["scenario"], row["replicate"]) for row in rows}
        lines = [f"  {title} ({len(files)} files)",
                 f"    {'method':18} {'background':15} {'runs':>4} {'failed':>6} {'F1':>11} {'precision':>9} "
                 f"{'recall':>6} {'SHD':>5} {'time':>7}"]
        for method, background in dict.fromkeys((row["method"], row["background"]) for row in rows):
            group = [row for row in rows if row["method"] == method and row["background"] == background]
            scored = [row for row in group if not row["error"]]
            if not scored:
                lines.append(f"    {method:18} {background:15} {len(group):4d} {len(group):6d}")
                continue
            f1 = np.array([float(row["f1"]) for row in scored])

            def mean(column: str) -> float:
                return float(np.mean([float(row[column]) for row in scored]))

            lines.append(f"    {method:18} {background:15} {len(group):4d} {len(group) - len(scored):6d} "
                         f"{f1.mean():5.2f} ±{f1.std():4.2f} {mean('precision'):9.2f} {mean('recall'):6.2f} "
                         f"{mean('shd'):5.1f} {mean('seconds'):6.1f}s")
        return lines + [""]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(RunMerger(Path(sys.argv[1])).save())
