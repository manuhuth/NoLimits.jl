#!/bin/bash
# Docs check. 1) instantiate docs/Project.toml as committed (reports whether it resolves);
# 2) build a scratch copy of the docs environment with the NoLimits compat entry removed
#    (docs/Project.toml pins NoLimits "0.2.0" while the package is 0.3.x) and
#    a) execute the @example blocks of the plotting pages, b) DOCS_FULL=1: full makedocs.
# Nothing in docs/ is modified.
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=aog_docs
#SBATCH --output=logs/%x_%j.log
set -u
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
echo "=== 1) docs env as committed"
D0=$B/logs/docs_env_committed_$SLURM_JOB_ID; mkdir -p $D0; cp $R/docs/Project.toml $R/docs/Manifest.toml $D0/
sed -i "s#NoLimits = {path = \"..\"}#NoLimits = {path = \"$R\"}#" $D0/Project.toml
julia --project=$D0 -e 'using Pkg; Pkg.instantiate()' ; echo "committed docs env instantiate rc=$?"
echo "=== 2) scratch docs env without the NoLimits compat pin"
D=$B/logs/docs_env_$SLURM_JOB_ID; mkdir -p $D
cp $R/docs/Project.toml $D/; cp $R/docs/Manifest.toml $D/
sed -i "s#NoLimits = {path = \"..\"}#NoLimits = {path = \"$R\"}#; /^NoLimits = \"0.2.0\"/d" $D/Project.toml
julia --project=$D -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'; echo "scratch env rc=$?"
cd $R/docs
julia --project=$D -t 4 $B/lib/run_doc_examples.jl $R/docs/src/plotting/index.md
echo "plotting examples rc=$?"
if [ "${DOCS_FULL:-0}" = "1" ]; then
  julia --project=$D -t 4 $R/docs/make.jl 2>&1 | tail -300
  echo "makedocs rc=${PIPESTATUS[0]}"
fi
