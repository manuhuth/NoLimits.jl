#!/bin/bash
#SBATCH --partition=intelsr_devel
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=aog_smoke
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%j.log
B=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest
cd $B && julia --project=$B/envs/${MODE:-new} $B/lib/smoke.jl
