#!/usr/bin/env bash
# The CausalArena benchmark in parallel, in tmux, for a remote machine.
#
# From kairos/:
#   bash scripts/run_causal_arena_tmux.sh                              # everything
#   DOMAINS=people REPLICATES=1 bash scripts/run_causal_arena_tmux.sh  # the people domains, one replicate
#   WORKERS=8 METHODS="kairos pc-fisherz" bash scripts/run_causal_arena_tmux.sh
#   DRY_RUN=1 bash scripts/run_causal_arena_tmux.sh                    # show the jobs, start nothing
#
# The work is split into one job per domain and regime (10 domains × 2 regimes = 20
# jobs at most), interventional ones first since they also run CD-NOD. WORKERS tmux
# windows share the queue: each takes the next job nobody has taken yet, so a worker
# that finishes early picks up more. A job is src/benchmarks/causal_arena/run.py on one
# domain and one regime, with its own output folder and log. Every process is held to
# one CPU thread, so the workers do not fight over cores. When all workers are done,
# the "merge" window joins the results into one results.csv and summary.txt
# (scripts/merge_causal_arena.py, which you can also run by hand at any time).
#
# Watch:  tmux attach -t arena   (Ctrl-b w lists the windows; Ctrl-b d leaves them running)
# Stop:   tmux kill-session -t arena
#
# Settings, as environment variables:
#   DOMAINS     all | people | domain names separated by spaces   (default: all)
#   REGIMES     "obs_int obs_only", or one of them                  (default: both)
#   REPLICATES  replicates per scenario and regime, up to 5          (default: 5)
#   LIMIT       at most this many scenarios per domain              (default: no limit)
#   METHODS     method names as in run.py, separated by spaces      (default: run.py's default)
#   WORKERS     windows running at once                             (default: the number of CPUs, at most the jobs)
#   PYTHON      the Python to use                                   (default: venv/bin/python)
#   DATA        where CausalArena's files are kept                  (default: data/causal_arena)
#   OUT         where the results go          (default: figures/benchmarks/causal_arena/run-<date>-<time>)
#   SESSION     the tmux session's name                             (default: arena)
#
# Files are downloaded from Hugging Face the first time they are needed (about 50 MB
# for everything). The manifest and every graph.json are fetched before the workers
# start, because two jobs share them; each data file belongs to one job only.

set -euo pipefail
cd "$(dirname "$0")/.."

DOMAINS=${DOMAINS:-all}
REGIMES=${REGIMES:-"obs_int obs_only"}
REPLICATES=${REPLICATES:-5}
LIMIT=${LIMIT:-}
METHODS=${METHODS:-}
PYTHON=${PYTHON:-venv/bin/python}
DATA=${DATA:-data/causal_arena}
OUT=${OUT:-figures/benchmarks/causal_arena/run-$(date +%Y%m%d-%H%M)}
SESSION=${SESSION:-arena}
DRY_RUN=${DRY_RUN:-0}

# --- checks -------------------------------------------------------------------------

[[ -x "$PYTHON" ]] || { echo "no Python at $PYTHON (set PYTHON=...)"; exit 1; }
if [[ "$DRY_RUN" != 1 ]]; then
    command -v tmux > /dev/null || { echo "tmux is not installed (sudo apt install tmux)"; exit 1; }
    if tmux has-session -t "$SESSION" 2> /dev/null; then
        echo "a tmux session named $SESSION is already running (tmux attach -t $SESSION, or SESSION=other)"
        exit 1
    fi
fi

mkdir -p "$DATA" "$OUT"/{jobs,logs,workers,done,taken}
ROOT=$(pwd)
DATA=$(cd "$DATA" && pwd)
OUT=$(cd "$OUT" && pwd)
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"

# --- the domains, and the files two jobs share ----------------------------------------

DOMAIN_LIST=$(DATA="$DATA" DOMAINS="$DOMAINS" PREFETCH=$((1 - DRY_RUN)) PYTHONPATH=. "$PYTHON" - << 'PYTHON_END'
import os
from pathlib import Path

from src.benchmarks.causal_arena import PEOPLE_DOMAINS, ArenaFiles, ScenarioIndex

files = ArenaFiles(Path(os.environ["DATA"]))
scenarios = ScenarioIndex.load(files)
known = list(dict.fromkeys(scenario.domain for scenario in scenarios))
asked = os.environ["DOMAINS"].split()
if asked == ["all"]:
    wanted = known
elif asked == ["people"]:
    wanted = [domain for domain in known if domain in PEOPLE_DOMAINS]
else:
    unknown = [domain for domain in asked if domain not in known]
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; choose from {known}, all or people")
    wanted = asked
if os.environ["PREFETCH"] == "1":
    for scenario in scenarios:
        if scenario.domain in wanted:
            files.path(scenario.graph_file)  # both regimes' jobs read it: fetch it once, now
print(" ".join(wanted))
PYTHON_END
)

# --- the jobs, and the workers that share them -----------------------------------------

JOBS=()
for regime in $REGIMES; do
    for domain in $DOMAIN_LIST; do
        JOBS+=("$domain $regime")
    done
done
CPUS=$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 4)
WORKERS=${WORKERS:-$CPUS}
(( WORKERS > ${#JOBS[@]} )) && WORKERS=${#JOBS[@]}
read -ra METHOD_LIST <<< "$METHODS"

# Every worker runs the same script. For each job in turn it claims the job by making
# the folder taken/<job>: mkdir succeeds for one worker only, the others skip the job.
{
    echo "#!/usr/bin/env bash"
    echo "# One worker (its number is \$1): takes the next job nobody has taken."
    echo "# Written by run_causal_arena_tmux.sh."
    echo "cd $(printf %q "$ROOT")"
    echo "export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
    for job in "${JOBS[@]}"; do
        read -r domain regime <<< "$job"
        name="$domain-$regime"
        command=("$PYTHON" -m src.benchmarks.causal_arena.run --domains "$domain" --regimes "$regime"
                 --replicates "$REPLICATES" --data "$DATA" --out "$OUT/jobs/$name")
        [[ -n "$LIMIT" ]] && command+=(--limit "$LIMIT")
        (( ${#METHOD_LIST[@]} > 0 )) && command+=(--methods "${METHOD_LIST[@]}")
        echo "if mkdir $(printf %q "$OUT/taken/$name") 2> /dev/null; then"
        echo "    echo \"=== $name: started \$(date '+%F %T')\""
        echo "    $(printf '%q ' "${command[@]}")2>&1 | tee $(printf %q "$OUT/logs/$name.log")"
        echo "    echo \"=== $name: finished \$(date '+%F %T'), exit \${PIPESTATUS[0]}\""
        echo "fi"
    done
    echo "touch $(printf %q "$OUT/done")/worker-\$1"
    echo "echo \"worker \$1 is done; this window stays open\""
    echo "exec bash"
} > "$OUT/workers/worker.sh"

cat > "$OUT/workers/merge.sh" << MERGE_END
#!/usr/bin/env bash
# Waits for every worker, then joins the results. Written by run_causal_arena_tmux.sh.
cd $(printf %q "$ROOT")
echo "waiting for $WORKERS workers (\$(date '+%F %T'))..."
while (( \$(ls $(printf %q "$OUT/done") | wc -l) < $WORKERS )); do sleep 30; done
echo "all workers done (\$(date '+%F %T'))"
PYTHONPATH=. $(printf %q "$PYTHON") scripts/merge_causal_arena.py $(printf %q "$OUT")
exec bash
MERGE_END

cat > "$OUT/settings.txt" << SETTINGS_END
started    $(date '+%F %T')
domains    $DOMAIN_LIST
regimes    $REGIMES
replicates $REPLICATES
limit      ${LIMIT:-none}
methods    ${METHODS:-run.py default}
workers    $WORKERS of $CPUS CPUs, ${#JOBS[@]} jobs
SETTINGS_END

echo "${#JOBS[@]} jobs on $WORKERS workers; results in $OUT"
if [[ "$DRY_RUN" == 1 ]]; then
    printf '  %s\n' "${JOBS[@]}"
    echo "(dry run: nothing started; the worker script is $OUT/workers/worker.sh)"
    exit 0
fi

# --- tmux -----------------------------------------------------------------------------

tmux new-session -d -s "$SESSION" -n merge "bash $(printf %q "$OUT/workers/merge.sh")"
for ((worker = 0; worker < WORKERS; worker++)); do
    tmux new-window -t "$SESSION" -n "w$worker" "bash $(printf %q "$OUT/workers/worker.sh") $worker"
done
echo "started tmux session $SESSION: tmux attach -t $SESSION"
