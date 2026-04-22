# Test Runtime Report

Measured on 2026-04-16. Single Julia process, sequential `include()` calls — the same execution
model as `Pkg.test()`. No per-file startup overhead.

## Summary

| Metric | Value |
| --- | --- |
| Files timed | 76 |
| **Sequential total** | **3106s (51m 46s)** |
| Previous suite total (pre-compaction) | ~6654s (1h 50m 54s) |
| Speedup | ~2.1× |

The speedup comes from three sources:
- Removal of FOCEI tests (`estimation_focei_tests.jl` 190s + `estimation_focei_map_tests.jl` 95s)
- Elimination of duplicate SAEM includes (was ~618s of re-run overhead)
- Shared fixtures in integration files avoid rebuilding the same models/fits repeatedly

## All Files (slowest first)

| Time (s) | Time | Test file |
| ---: | ---: | --- |
| 368 | 6m08s | `estimation_saem_tests.jl` |
| 223 | 3m43s | `laplace_fastpath_tests.jl` |
| 215 | 3m35s | `integration_simple_re.jl` |
| 208 | 3m28s | `uq_edge_cases_tests.jl` |
| 196 | 3m16s | `plot_random_effects_tests.jl` |
| 176 | 2m56s | `estimation_ghquadrature_tests.jl` |
| 161 | 2m41s | `hmm_estimation_method_matrix_tests.jl` |
| 133 | 2m13s | `estimation_laplace_tests.jl` |
| 107 | 1m47s | `integration_no_re.jl` |
| 104 | 1m44s | `saem_options_unit_tests.jl` |
| 104 | 1m44s | `estimation_multistart_tests.jl` |
| 104 | 1m44s | `estimation_mcem_tests.jl` |
| 97 | 1m37s | `uq_tests.jl` |
| 83 | 1m23s | `random_effect_new_plots_tests.jl` |
| 78 | 1m18s | `estimation_mcmc_re_tests.jl` |
| 72 | 1m12s | `plotting_functions_tests.jl` |
| 58 | 0m58s | `plot_cache_tests.jl` |
| 57 | 0m57s | `summaries_fit_uq_tests.jl` |
| 51 | 0m51s | `estimation_mcmc_tests.jl` |
| 46 | 0m46s | `estimation_saem_suffstats_tests.jl` |
| 33 | 0m33s | `estimation_mcem_is_tests.jl` |
| 33 | 0m33s | `identifiability_tests.jl` |
| 31 | 0m31s | `saem_adaptive_mh_tests.jl` |
| 30 | 0m30s | `saem_saemixmh_tests.jl` |
| 26 | 0m26s | `integration_plotting.jl` |
| 26 | 0m26s | `residual_plots_tests.jl` |
| 23 | 0m23s | `stickbreak_uq_natural_extension_tests.jl` |
| 20 | 0m20s | `hmm_continuous_tests.jl` |
| 19 | 0m19s | `re_covariate_usage_tests.jl` |
| 17 | 0m17s | `vpc_tests.jl` |
| 15 | 0m15s | `hmm_mv_discrete_tests.jl` |
| 15 | 0m15s | `hmm_discrete_time_tests.jl` |
| 15 | 0m15s | `plot_observation_distributions_tests.jl` |
| 12 | 0m12s | `estimation_hutchinson_tests.jl` |
| 12 | 0m12s | `hmm_mv_continuous_tests.jl` |
| 11 | 0m11s | `estimation_common_tests.jl` |
| 9 | 0m09s | `data_model_tests.jl` |
| 9 | 0m09s | `stickbreak_uq_tests.jl` |
| 9 | 0m09s | `ad_stickbreak_hmm.jl` |
| 8 | 0m08s | `ad_fixed_prede.jl` |
| 8 | 0m08s | `ad_ode_solve_basic.jl` |
| 8 | 0m08s | `ad_random_effects.jl` |
| 8 | 0m08s | `data_simulation_tests.jl` |
| 6 | 0m06s | `ad_model_full.jl` |
| 5 | 0m05s | `compact_show_tests.jl` |
| 5 | 0m05s | `fixed_effects_tests.jl` |
| 5 | 0m05s | `ode_callbacks_tests.jl` |
| 5 | 0m05s | `data_model_ode_tests.jl` |
| 4 | 0m04s | `ad_ode_solve_richer.jl` |
| 4 | 0m04s | `ad_random_effects_values.jl` |
| 4 | 0m04s | `stickbreak_parameter_tests.jl` |
| 3 | 0m03s | `transform_tests.jl` |
| 3 | 0m03s | `ad_flow.jl` |
| 3 | 0m03s | `model_tests.jl` |
| 3 | 0m03s | `estimation_saem_autodetect_tests.jl` |
| 3 | 0m03s | `ad_differential_equation.jl` |
| 2 | 0m02s | `continuous_transition_matrix_tests.jl` |
| 2 | 0m02s | `ode_solve_tests.jl` |
| 2 | 0m02s | `equation_display_tests.jl` |
| 1 | 0m01s | `random_effects_tests.jl` |
| 1 | 0m01s | `uq_plotting_tests.jl` |
| 1 | 0m01s | `summaries_data_model_tests.jl` |
| 1 | 0m01s | `splines_tests.jl` |
| 1 | 0m01s | `summaries_model_tests.jl` |
| 1 | 0m01s | `stickbreak_transform_tests.jl` |
| 1 | 0m01s | `parameters_tests.jl` |
| 1 | 0m01s | `formulas_tests.jl` |
| 1 | 0m01s | `ad_softtree.jl` |
| 1 | 0m01s | `softtrees_tests.jl` |
| 1 | 0m01s | `model_macro_tests.jl` |
| 0 | 0m00s | `initialde_tests.jl` |
| 0 | 0m00s | `differential_equation_tests.jl` |
| 0 | 0m00s | `prede_tests.jl` |
| 0 | 0m00s | `covariates_tests.jl` |
| 0 | 0m00s | `ad_misc.jl` |
| 0 | 0m00s | `helpers_tests.jl` |

## Concentration

| Slice | Files | Time (s) | Share |
| --- | ---: | ---: | ---: |
| Top 5 | 5 | 1211 | 39% |
| Top 10 | 10 | 1989 | 64% |
| Top 20 | 20 | 2613 | 84% |
| Bottom 40 | 40 | 187 | 6% |

## Notable Changes vs Previous Measurement

| File | Now (s) | Before (s) | Change |
| --- | ---: | ---: | --- |
| `estimation_saem_tests.jl` | 368 | 432 | −15% |
| `uq_tests.jl` | 97 | 224 | −57% |
| `plot_random_effects_tests.jl` | 196 | 265 | −26% |
| `plotting_functions_tests.jl` | 72 | 153 | −53% |
| `plot_cache_tests.jl` | 58 | 126 | −54% |
| `estimation_multistart_tests.jl` | 104 | 196 | −47% |
| `estimation_ghquadrature_tests.jl` | 176 | 213 | −17% |
| `hmm_estimation_method_matrix_tests.jl` | 161 | 269 | −40% |
| `estimation_mcem_tests.jl` | 104 | 148 | −30% |
| `estimation_laplace_tests.jl` | 133 | 173 | −23% |
| `random_effect_new_plots_tests.jl` | 83 | 159 | −48% |
| `laplace_fastpath_tests.jl` | 223 | 179 | +25% ↑ |
| `uq_edge_cases_tests.jl` | 208 | 161 | +29% ↑ |
| `estimation_focei_tests.jl` | — | 190 | removed |
| `estimation_focei_map_tests.jl` | — | 95 | removed |
| `integration_no_re.jl` | 107 | — | new |
| `integration_simple_re.jl` | 215 | — | new |
| `integration_plotting.jl` | 26 | — | new |

`laplace_fastpath_tests.jl` and `uq_edge_cases_tests.jl` both grew — worth investigating
whether new tests were added or existing tests became slower.
