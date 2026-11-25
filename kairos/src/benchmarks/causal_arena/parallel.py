import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.benchmarks.causal_arena.benchmark import ArenaBenchmark, Result
from src.benchmarks.causal_arena.scenarios import Scenario

Task = tuple[Scenario, str, int]  # one file: scenario, regime, replicate


class ParallelRun:
    """Runs the benchmark on many files, `workers` of them at a time, each in its own
    process. Every finished file is kept in `folder` (one small .pkl), so a run that was
    stopped goes on from there when it is started again with the same folder.

    Files are downloaded first, one by one: two processes must not fetch the same one.
    """

    def __init__(self, benchmark: ArenaBenchmark, folder: Path, workers: int = 1):
        self.benchmark = benchmark
        self.folder = folder
        self.workers = workers

    def results(self, tasks: list[Task]) -> Iterator[list[Result]]:
        """Each file's results: first those already kept, then the others as they finish."""
        self.folder.mkdir(parents=True, exist_ok=True)
        waiting = []
        for task in tasks:
            kept = self._kept(task)
            if kept.exists():
                yield pickle.loads(kept.read_bytes())
            else:
                waiting.append(task)
        for scenario, regime, replicate in waiting:
            self.benchmark.data.files.path(scenario.graph_file)
            self.benchmark.data.files.path(scenario.data_file(regime, replicate))
        yield from self._run(waiting)

    def _run(self, tasks: list[Task]) -> Iterator[list[Result]]:
        if self.workers == 1:
            for task in tasks:
                yield self._one(task)
            return
        with ProcessPoolExecutor(self.workers) as pool:
            for finished in as_completed([pool.submit(self._one, task) for task in tasks]):
                yield finished.result()

    def _one(self, task: Task) -> list[Result]:
        results = list(self.benchmark.run(*task))
        kept = self._kept(task)
        partial = kept.with_suffix(".part")  # a cut-off write is never mistaken for a result
        partial.write_bytes(pickle.dumps(results))
        partial.rename(kept)
        return results

    def _kept(self, task: Task) -> Path:
        scenario, regime, replicate = task
        return self.folder / f"{scenario.domain}-{scenario.slug}-{regime}-rep{replicate}.pkl"
