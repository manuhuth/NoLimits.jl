#!/bin/bash
# Runs each test file listed in runtests.jl in parallel, saves output to /tmp/nl_test_results/
set -euo pipefail

PROJ=/Users/manuel/Documents/LongitudinalData.jl/NoLimits
OUT=/tmp/nl_test_results
mkdir -p "$OUT"

# List of test files from runtests.jl (excluding commented ones)
TESTS=(
  softtrees_tests.jl
  ad_softtree.jl
  ad_flow.jl
  ad_random_effects.jl
  ad_random_effects_values.jl
  ad_fixed_prede.jl
  ad_differential_equation.jl
  ad_ode_solve_basic.jl
  ad_ode_solve_richer.jl
  ad_misc.jl
  ad_model_full.jl
  helpers_tests.jl
  parameters_tests.jl
  transform_tests.jl
  fixed_effects_tests.jl
  splines_tests.jl
  covariates_tests.jl
  random_effects_tests.jl
  prede_tests.jl
  differential_equation_tests.jl
  ode_solve_tests.jl
  formulas_tests.jl
  initialde_tests.jl
  model_macro_tests.jl
  model_tests.jl
  equation_display_tests.jl
  data_model_tests.jl
  identifiability_tests.jl
  data_model_ode_tests.jl
  summaries_data_model_tests.jl
  summaries_model_tests.jl
  summaries_fit_uq_tests.jl
  compact_show_tests.jl
  data_simulation_tests.jl
  ode_callbacks_tests.jl
  plot_cache_tests.jl
  plotting_functions_tests.jl
  vpc_tests.jl
  plot_observation_distributions_tests.jl
  residual_plots_tests.jl
  plot_random_effects_tests.jl
  random_effect_new_plots_tests.jl
  re_covariate_usage_tests.jl
  estimation_common_tests.jl
  accessors_tests.jl
  estimation_mle_tests.jl
  estimation_map_tests.jl
  estimation_mcmc_tests.jl
  estimation_mcmc_re_tests.jl
  estimation_vi_tests.jl
  estimation_laplace_tests.jl
  estimation_laplace_fit_tests.jl
  estimation_hutchinson_tests.jl
  estimation_laplace_map_tests.jl
  laplace_fastpath_baseline_tests.jl
  laplace_fastpath_config_tests.jl
  estimation_mcem_tests.jl
  estimation_mcem_is_tests.jl
  estimation_saem_tests.jl
  saem_schedule_tests.jl
  saem_multichain_tests.jl
  saem_sa_anneal_tests.jl
  saem_var_lb_tests.jl
  estimation_saem_autodetect_tests.jl
  estimation_saem_suffstats_tests.jl
  saem_schedule_tests.jl
  saem_multichain_tests.jl
  saem_sa_anneal_tests.jl
  saem_var_lb_tests.jl
  saem_saemixmh_tests.jl
  saem_adaptive_mh_tests.jl
  estimation_multistart_tests.jl
  estimation_ghquadrature_tests.jl
  uq_tests.jl
  uq_edge_cases_tests.jl
  uq_plotting_tests.jl
  hmm_continuous_tests.jl
  hmm_discrete_time_tests.jl
  hmm_estimation_method_matrix_tests.jl
  hmm_mv_discrete_tests.jl
  hmm_mv_continuous_tests.jl
  stickbreak_parameter_tests.jl
  stickbreak_transform_tests.jl
  stickbreak_uq_tests.jl
  stickbreak_uq_natural_extension_tests.jl
  ad_stickbreak_hmm.jl
  continuous_transition_matrix_tests.jl
  serialization_tests.jl
)

run_test() {
  local t="$1"
  local logfile="$OUT/${t%.jl}.log"
  local statusfile="$OUT/${t%.jl}.status"

  if julia --project="$PROJ" -e "
    using Test
    using NoLimits
    include(\"$PROJ/test/$t\")
  " > "$logfile" 2>&1; then
    echo "PASS" > "$statusfile"
    echo "PASS: $t"
  else
    echo "FAIL" > "$statusfile"
    echo "FAIL: $t"
  fi
}

export -f run_test
export PROJ OUT

# Run all tests in parallel (max 8 at a time to avoid resource exhaustion)
printf '%s\n' "${TESTS[@]}" | xargs -P 8 -I {} bash -c 'run_test "$@"' _ {}

echo ""
echo "=== SUMMARY ==="
PASSED=0
FAILED=0
for t in "${TESTS[@]}"; do
  statusfile="$OUT/${t%.jl}.status"
  if [ -f "$statusfile" ]; then
    status=$(cat "$statusfile")
    if [ "$status" = "PASS" ]; then
      PASSED=$((PASSED+1))
    else
      FAILED=$((FAILED+1))
      echo "FAILED: $t"
    fi
  else
    echo "NO STATUS: $t"
    FAILED=$((FAILED+1))
  fi
done
echo "Passed: $PASSED / $((PASSED+FAILED))"
