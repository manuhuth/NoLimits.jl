module NoLimitsMakieExt

# Makie-dependent drawing layer. Loads when NoLimits and Makie are both present;
# users get it via `using CairoMakie` (any Makie backend pulls in Makie).
# This extension may only reference Makie — never a specific backend.
# The Makie-free core (PlotStyle, PlotCache, get_residuals, build_plot_cache and their
# helpers) lives in NoLimits proper; this module only adds the plot_* methods plus the
# panel model / styled-drawing layer in plotting/plotting.jl.

using Makie: Makie, Figure, Axis, GridLayout, lines!, scatter!, barplot!, heatmap!,
             poly!, band!, hlines!, vlines!, vspan!, text!, axislegend, xlims!,
             ylims!, Rect2f
using KernelDensity
using Distributions
using Random
using Statistics
using StatsFuns
using ComponentArrays
using MCMCChains
using OrdinaryDiffEq
using SciMLBase
using DataFrames

# Extend the core stub generics with the actual drawing methods.
import NoLimits: plot_data, plot_dv_ipred, plot_dv_pred, plot_em_trajectories,
                 plot_emission_distributions, plot_fits, plot_fits_comparison,
                 plot_hidden_states, plot_multistart_fixed_effect_variability,
                 plot_multistart_waterfall, plot_observation_distributions,
                 plot_observed_profiles, plot_random_effect_distributions,
                 plot_random_effect_pairplot, plot_random_effect_pit,
                 plot_random_effect_standardized, plot_random_effect_standardized_scatter,
                 plot_random_effects_pdf, plot_random_effects_scatter, plot_residual_acf,
                 plot_residual_distribution, plot_residual_pit, plot_residual_qq,
                 plot_residuals, plot_shrinkage, plot_uq_distributions, plot_vpc,
                 plot_wres_pred

# Makie-free helpers/types/accessors/consts the drawing code reads from core.
using NoLimits: COLOR_ACCENT, COLOR_CI, COLOR_PRIMARY, COLOR_REFERENCE, COLOR_SECONDARY,
                DEFAULT_DPI, DEFAULT_PLOT_COLS, DataModel, FitDiagnostics, FitParameters,
                FitResult,
                FitSummary, Individual, InverseTransform, MAX_FIGURE_HEIGHT, MCEM, MCMC,
                MIN_FIGURE_HEIGHT, MLE, MLEResult, MVDiscreteTimeDiscreteStatesHMM, Model,
                Multistart,
                MultistartFitResult, NormalizingPlanarFlow, PlotCache, PlotStyle, SAEM,
                UQResult,
                _acf_for_series, _apply_hmm_filter!, _apply_param_overrides,
                _as_fit_result_for_plotting, _assign_bins, _axis_label, _bin_edges_quantile,
                _can_dense_plot, _collect_ipred_series, _collect_multivariate_series,
                _collect_observed_xy, _collect_pred_series, _collect_scalar_series,
                _comparison_line_colors, _comparison_line_style, _default_random_effects,
                _default_random_effects_from_dm, _dense_time_grid, _density_grid_continuous,
                _density_grid_discrete, _density_interval_slice, _ebe_by_level,
                _ebe_by_level_mcmc,
                _ensure_save_path, _eta_vec_from_levels, _extend_bin_series,
                _filter_re_without_covariates, _fit_constants_re, _fit_curve_from_cache,
                _flatten_param_with_labels, _float_if_real, _get_dm, _get_observable,
                _get_x_values,
                _histogram_xy, _hmm_with_prior, _interp_linear, _is_bernoulli, _is_discrete,
                _is_posterior_draw_fit,
                _kde_xy, _kernel_quantiles, _level_to_individual, _marginal_colors,
                _marginal_label,
                _marginal_normal, _mean_pmf_support, _merge_limits, _multistart_data_model,
                _mv_n_outcomes, _needs_rowwise_random_effects,
                _normalize_top_level_parameter_selection,
                _obs_multivariate_info, _pad_limits, _pit_value, _posterior_drawn_params,
                _posterior_fixed_means, _pred_re_per_individual, _representative_dist,
                _require_re_supported, _require_varying_covariate, _res_constants_re,
                _residual_metric_column, _residual_metric_label, _resolve_emission_row,
                _resolve_individuals, _resolve_levels, _resolve_n_bins, _resolve_obs_rows,
                _resolve_observables, _resolve_plot_path, _resolve_re_names,
                _row_random_effects_at,
                _sample_random_effects_levels, _simulate_obs, _solve_dense_individual,
                _sol_accessors_with_crossings, _standardize_re,
                _stat_from_dist, _state_emission_marginals, _uq_density_ylabel, _uq_kde_xy,
                _uq_merge_limits, _uq_param_indices, _uq_param_label,
                _uq_wald_coord_transforms,
                _validate_plot_metric, _validate_same_data_model_for_comparison,
                _varying_at,
                _wald_closed_form_kind, _wald_density_xy, _with_posterior_warmup,
                build_plot_cache,
                calculate_formulas_obs, calculate_plot_size, calculate_prede,
                compute_shrinkage,
                compute_uq, default_axis_kwargs, flatten_re_names,
                get_create_random_effect_distribution, get_data_model,
                get_de_accessors_builder,
                get_de_compiler, get_formulas_meta, get_helper_funs, get_inverse_transform,
                get_method,
                get_model_funs, get_names, get_notes, get_objective, get_params,
                get_re_groups,
                get_residuals, get_uq_backend, get_uq_draws, get_uq_estimates,
                get_uq_intervals,
                get_uq_parameter_names, get_uq_vcov, get_θ0_untransformed,
                posterior_hidden_states,
                probabilities_hidden_states, transform

include("plotting/plotting.jl")
include("plotting/plots.jl")
include("plotting/plotting_vpc.jl")
include("plotting/plotting_observation_distributions.jl")
include("plotting/plotting_residuals.jl")
include("plotting/plotting_random_effects.jl")
include("plotting/plotting_uq.jl")

end # module NoLimitsMakieExt
