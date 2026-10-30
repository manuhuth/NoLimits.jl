#!/bin/bash
# Format with the CI-pinned Runic 1.9.0 in place, then check.
#SBATCH --partition=intelsr_devel
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=aog_runic
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%j.log
cd /home/mhuth1_hpc/NoLimits-aog
julia --project=/home/mhuth1_hpc/NoLimitsAutoresearch/tools/runic-env -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace src test docs ext dev
echo "inplace rc=$?"
julia --project=/home/mhuth1_hpc/NoLimitsAutoresearch/tools/runic-env -e 'using Runic; exit(Runic.main(ARGS))' -- --check src test docs ext
echo "check rc=$?"
