#!/bin/bash
# Format with the CI-pinned Runic 1.9.0 in place, then check.
#SBATCH --partition=intelsr_devel
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=aog_runic
#SBATCH --output=logs/%x_%j.log
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd "$R"
julia --project=/home/mhuth1_hpc/NoLimitsAutoresearch/tools/runic-env -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src test docs ext dev
echo "inplace rc=$?"
julia --project=/home/mhuth1_hpc/NoLimitsAutoresearch/tools/runic-env -e 'using Runic; exit(Runic.main(ARGS))' -- --check src test docs ext
echo "check rc=$?"
