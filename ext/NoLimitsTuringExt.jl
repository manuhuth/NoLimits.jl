module NoLimitsTuringExt

# Turing-dependent sampling layer. Loads when NoLimits and Turing are both present.
# The Turing-free core (the `MCMC`/`VI` methods, their result types and accessors, and
# the chain-based UQ/plotting/serialization paths) lives in NoLimits proper; this module
# only fills in the pieces that build DynamicPPL models and run samplers.
#
# DynamicPPL and AdvancedVI are reached through Turing rather than declared as weakdeps
# of their own: a weakdep that is not itself a trigger is not visible inside an extension.

using Turing
using Turing: DynamicPPL
using MCMCChains
using Bijectors
using Distributions
using ComponentArrays
using SciMLBase
using SciMLBase: EnsembleSerial, EnsembleThreads
using Random
using Statistics
using LinearAlgebra

using NoLimits: NoLimits,
                DataModel, FitDiagnostics, FitParameters, FitResult, FitSummary,
                MCEM, MCMC, MCMCResult, Model, Priorless, SAEM, VI, VIResult,
                REBatchInfo,
                _WARNED_NUMERIC_ERROR, _as_namedtuple, _build_constants_cache,
                _build_eta_ind, _extract_b_samples, _filter_b_samples_by_prior,
                _is_numeric_error, _loglikelihood_individual, _mcem_sample_batch,
                _mcmc_objective, _normalize_constants_re, _re_logpdf_batch,
                _re_marginals, _saem_apply_anneal_dist, _symmetrize_psd_params,
                _turing_prior, _warn_if_scaled_params, _with_posterior_params,
                build_ll_cache, create_random_effect_distribution, get_chain,
                get_const_cov, get_df, get_fixed, get_helper_funs, get_individuals,
                get_inds, get_is_scalar, get_levels, get_model, get_model_funs, get_n_b,
                get_names, get_obs_cols, get_priors, get_random, get_ranges,
                get_re_group_info, get_re_groups, get_re_info, get_re_map, get_re_names,
                get_reps, get_θ0_untransformed

include("turing/mcmc.jl")
include("turing/vi.jl")
include("turing/mcem.jl")

end
