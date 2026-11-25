#!/usr/bin/env bash
# The CausalArena benchmark in the background, several files at a time (Linux or macOS).
#
# From kairos/:
#   bash scripts/run_causal_arena.sh                                    # everything, 5 processes
#   WORKERS=4 bash scripts/run_causal_arena.sh --domains people         # 4 processes, the people domains
#   bash scripts/run_causal_arena.sh --methods kairos --replicates 1    # kairos only, one sample per scenario
#   OUT=figures/benchmarks/causal_arena/run-... bash scripts/run_causal_arena.sh    # go on with a stopped run
#
# Everything after the script's name goes to src/benchmarks/causal_arena/run.py (see its --help).
# It keeps running after you log out. Files are downloaded first (about 50 MB for everything).

set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-5}
OUT=${OUT:-figures/benchmarks/causal_arena/run-$(date +%Y%m%d-%H%M)}
PYTHON=${PYTHON:-venv/bin/python}

if [[ ! -x "$PYTHON" ]]; then
    echo "no Python at $PYTHON. Make it with:"
    echo "  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# One thread per process, so the workers do not fight over the CPUs.
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUT"
set -m  # the run gets its own process group, so one kill stops it and its workers
nohup "$PYTHON" -u -m src.benchmarks.causal_arena.run --workers "$WORKERS" --out "$OUT" "$@" >> "$OUT/log.txt" 2>&1 < /dev/null &
echo "$!" > "$OUT/pid"

echo "started with $WORKERS processes; results in $OUT"
echo "  watch:  tail -f $OUT/log.txt"
echo "  so far: cat $OUT/summary.txt"
echo "  stop:   kill -- -\$(cat $OUT/pid)"
