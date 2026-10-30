#!/bin/bash
# Build out/summary.csv, side-by-side images (out/compare/) and contact sheets (out/sheets/).
#   sbatch jobs/compare.sbatch                                  # everything, 1 task
#   FILTER='^plot_fits__' NCHUNKS=8 sbatch --array=1-8 jobs/compare.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=aog_compare
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%A_%a.log
set -u
B=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest
cd $B
julia --project=$B/envs/new $B/lib/compare.jl "${FILTER:-}" ${SLURM_ARRAY_TASK_ID:-1} ${NCHUNKS:-1}
