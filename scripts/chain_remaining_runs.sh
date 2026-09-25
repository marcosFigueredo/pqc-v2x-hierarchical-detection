#!/usr/bin/env bash
# Wait for the main pipeline run to finish, then run the two follow-on
# experiments sequentially. They are CPU-bound on the same machine, so
# running them concurrently would only slow all three down -- and on 32 GB
# two concurrent 5M-row runs risk exhausting memory.
#
# The wait MUST use a Windows-native process check. Git Bash `kill -0` does
# not understand Windows PIDs: it returns failure immediately for a live
# process, which makes the chain think the main run already finished and
# launch the follow-ons on top of it.
#
# Usage: scripts/chain_remaining_runs.sh <windows-pid-of-main-run>
set -u

MAIN_PID="${1:?usage: chain_remaining_runs.sh <windows-pid>}"
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d_%H%M%S)

is_running() {
    local state
    state=$(powershell -NoProfile -Command \
        "if (Get-Process -Id $MAIN_PID -ErrorAction SilentlyContinue) { 'ALIVE' } else { 'DEAD' }" \
        2>/dev/null | tr -d '[:space:]')
    [ "$state" = "ALIVE" ]
}

if ! is_running; then
    echo "[chain] pid $MAIN_PID is not running; refusing to start." >&2
    echo "[chain] pass the pid of a live run, or start the follow-ons by hand." >&2
    exit 1
fi

echo "[chain] waiting for main run (pid $MAIN_PID) to finish..."
while is_running; do
    sleep 60
done
echo "[chain] main run finished at $(date)"

echo "[chain] starting multi-seed robustness at $(date)"
python -u scripts/run_multiseed_robustness.py \
    > "logs/multiseed_test_partition_${STAMP}.log" 2>&1
echo "[chain] multi-seed exit=$? at $(date)"

echo "[chain] starting split ablation at $(date)"
python -u scripts/run_random_split_ablation.py \
    > "logs/split_ablation_${STAMP}.log" 2>&1
echo "[chain] split ablation exit=$? at $(date)"

echo "[chain] regenerating tables and figures from results/"
python -u scripts/make_figures_and_tables.py \
    > "logs/make_tables_${STAMP}.log" 2>&1
echo "[chain] table generation exit=$? at $(date)"
echo "[chain] all done at $(date)"
