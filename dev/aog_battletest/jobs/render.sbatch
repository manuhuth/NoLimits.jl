#!/bin/bash
# Render the case matrix in chunks. Examples (from dev/aog_battletest):
#   MODE=baseline sbatch --array=1-40 jobs/render.sbatch            # NCHUNKS defaults to 40
#   MODE=new NCHUNKS=10 FILTER='^plot_(fits|data)__' sbatch --array=1-10 jobs/render.sbatch
#   FORCE=1 re-renders cases that already have a status file.
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --job-name=aog_render
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%A_%a.log
set -u
B=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest
MODE=${MODE:-new}
cd $B
echo "mode=$MODE chunk=$SLURM_ARRAY_TASK_ID/${NCHUNKS:-40} filter=${FILTER:-} host=$(hostname)"
julia --project=$B/envs/$MODE -t $SLURM_CPUS_PER_TASK $B/lib/render.jl $MODE $SLURM_ARRAY_TASK_ID ${NCHUNKS:-40} "${FILTER:-}"
