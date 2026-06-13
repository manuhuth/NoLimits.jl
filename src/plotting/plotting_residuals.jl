export get_residuals
export plot_residuals
export plot_residual_distribution
export plot_residual_qq
export plot_residual_pit
export plot_residual_acf

using DataFrames
using Distributions
using KernelDensity
using Random
using Statistics

const _RESIDUAL_ALLOWED = Set([:pit, :quantile, :raw, :pearson, :logscore])
const _RESIDUAL_PLOT_ALLOWED = Set([:pit, :quantile, :raw, :pearson, :logscore])

@inline function _residual_metric_column(metric::Symbol)
    if metric == :pit
        return :pit
    elseif metric == :quantile
        return :res_quantile
    elseif metric == :raw
        return :res_raw
    elseif metric == :pearson
        return :res_pearson
    elseif metric == :logscore
        return :logscore
    end
    error("Unknown residual metric $(metric).")
end

@inline function _residual_qlo_column(metric::Symbol)
    return Symbol(string(_residual_metric_column(metric)), "_qlo")
end

@inline function _residual_qhi_column(metric::Symbol)
    return Symbol(string(_residual_metric_column(metric)), "_qhi")
end

function _validate_residual_metrics(residuals)
    if residuals isa Symbol
        res = [residuals]
    elseif residuals isa AbstractVector
        res = Symbol.(collect(residuals))
    else
        error("residuals must be a Symbol or vector of Symbol.")
    end
    isempty(res) && error("residuals must include at least one metric.")
    for r in res
        r in _RESIDUAL_ALLOWED ||
            error("Unknown residual metric $(r). Allowed: $(_RESIDUAL_ALLOWED).")
    end
    return unique(res)
end

function _validate_plot_metric(metric::Symbol)
    metric in _RESIDUAL_PLOT_ALLOWED ||
        error("Unknown residual metric $(metric). Allowed: $(_RESIDUAL_PLOT_ALLOWED).")
    return metric
end

@inline function _residual_metric_label(metric::Symbol)
    metric == :pit && return "PIT"
    metric == :quantile && return "Quantile Residual"
    metric == :raw && return "Raw Residual"
    metric == :pearson && return "Pearson Residual"
    metric == :logscore && return "Negative Log-Likelihood"
    error("Unknown residual metric $(metric).")
end

function _resolve_residual_observables(dm::DataModel, observables)
    obs = get_formulas_meta(dm.model.formulas.formulas).obs_names
    if observables === nothing
        return collect(obs)
    end
    obs_list = observables isa AbstractVector ? Symbol.(collect(observables)) :
               [Symbol(observables)]
    for o in obs_list
        o in obs || error("Observable $(o) not found. Available: $(obs).")
    end
    return obs_list
end

function _resolve_residual_individuals(dm::DataModel, individuals_idx)
    n = length(dm.individuals)
    if individuals_idx === nothing
        return collect(1:n)
    end
    ids = individuals_idx isa AbstractVector ? collect(individuals_idx) : [individuals_idx]
    if all(x -> x isa Integer && 1 <= x <= n, ids)
        return Int.(ids)
    end
    out = Int[]
    for id in ids
        haskey(dm.id_index, id) || error("Unknown individual id $(id).")
        push!(out, dm.id_index[id])
    end
    return out
end

function _resolve_residual_obs_rows(obs_rows, obs_rows_all)
    if obs_rows === nothing
        return collect(1:length(obs_rows_all))
    end
    idxs = obs_rows isa AbstractVector ? collect(obs_rows) : [obs_rows]
    for idx in idxs
        1 <= idx <= length(obs_rows_all) || error("obs_rows index $(idx) out of bounds.")
    end
    return idxs
end

@inline function _to_float_or_missing(v)
    if ismissing(v)
        return missing
    end
    if v isa Number
        x = Float64(v)
        return isfinite(x) ? x : missing
    end
    return missing
end

function _metric_summary(vals::Vector{Union{Missing, Float64}}, qlo::Float64, qhi::Float64)
    vals_use = collect(skipmissing(vals))
    isempty(vals_use) && return (missing, missing, missing)
    m = mean(vals_use)
    lo = quantile(vals_use, qlo)
    hi = quantile(vals_use, qhi)
    return (Float64(m), Float64(lo), Float64(hi))
end

function _pit_from_dist(dist,
        y::Union{Missing, Float64};
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        rng::AbstractRNG = Random.default_rng())
    ismissing(y) && return missing
    if applicable(cdf, dist, y)
        if dist isa DiscreteDistribution
            hi = clamp(Float64(cdf(dist, y)), 0.0, 1.0)
            if randomize_discrete && applicable(pdf, dist, y)
                mass = Float64(pdf(dist, y))
                lo = clamp(hi - mass, 0.0, 1.0)
                lo > hi && ((lo, hi) = (hi, lo))
                return lo == hi ? lo : lo + rand(rng) * (hi - lo)
            end
            return hi
        end
        return clamp(Float64(cdf(dist, y)), 0.0, 1.0)
    end
    if cdf_fallback_mc > 0
        try
            samples = rand(rng, dist, cdf_fallback_mc)
            vals = vec(samples)
            isempty(vals) && return missing
            return count(v -> v <= y, vals) / length(vals)
        catch
            return missing
        end
    end
    return missing
end

function _compute_residual_metrics(dist,
        y::Union{Missing, Float64},
        residual_list::Vector{Symbol},
        fitted_stat,
        randomize_discrete::Bool,
        cdf_fallback_mc::Int,
        rng::AbstractRNG)
    req = Set(residual_list)
    fitted = try
        v = _stat_from_dist(dist, fitted_stat)
        v isa Number ? Float64(v) : missing
    catch
        missing
    end

    pit = missing
    if (:pit in req) || (:quantile in req)
        pit = _pit_from_dist(dist, y; randomize_discrete = randomize_discrete,
            cdf_fallback_mc = cdf_fallback_mc, rng = rng)
    end

    res_quantile = missing
    if :quantile in req
        if ismissing(pit)
            res_quantile = missing
        else
            p = clamp(pit, eps(Float64), 1.0 - eps(Float64))
            res_quantile = Float64(quantile(Normal(), p))
        end
    end

    μ = missing
    if (:raw in req) || (:pearson in req)
        try
            m = mean(dist)
            μ = m isa Number ? Float64(m) : missing
        catch
            μ = missing
        end
    end

    res_raw = missing
    if :raw in req
        if !ismissing(y) && !ismissing(μ)
            res_raw = y - μ
        end
    end

    res_pearson = missing
    if :pearson in req
        try
            v = var(dist)
            if !ismissing(y) && !ismissing(μ) && v isa Number
                vv = Float64(v)
                if vv > 0.0 && isfinite(vv)
                    res_pearson = (y - μ) / sqrt(vv)
                end
            end
        catch
            res_pearson = missing
        end
    end

    logscore = missing
    if :logscore in req
        if !ismissing(y) && applicable(logpdf, dist, y)
            ls = -Float64(logpdf(dist, y))
            isfinite(ls) && (logscore = ls)
        end
    end

    return (fitted = fitted, pit = pit, res_quantile = res_quantile,
        res_raw = res_raw, res_pearson = res_pearson, logscore = logscore)
end

function _residual_row(; individual_idx::Int,
        id,
        row::Int,
        obs_index::Int,
        observable::Symbol,
        time::Union{Missing, Float64},
        x::Union{Missing, Float64},
        y::Union{Missing, Float64},
        fitted::Union{Missing, Float64},
        pit::Union{Missing, Float64},
        pit_qlo::Union{Missing, Float64},
        pit_qhi::Union{Missing, Float64},
        res_quantile::Union{Missing, Float64},
        res_quantile_qlo::Union{Missing, Float64},
        res_quantile_qhi::Union{Missing, Float64},
        res_raw::Union{Missing, Float64},
        res_raw_qlo::Union{Missing, Float64},
        res_raw_qhi::Union{Missing, Float64},
        res_pearson::Union{Missing, Float64},
        res_pearson_qlo::Union{Missing, Float64},
        res_pearson_qhi::Union{Missing, Float64},
        logscore::Union{Missing, Float64},
        logscore_qlo::Union{Missing, Float64},
        logscore_qhi::Union{Missing, Float64},
        draw::Union{Missing, Int},
        n_draws::Int)
    return (; individual_idx, id, row, obs_index, observable, time, x, y, fitted,
        pit, pit_qlo, pit_qhi,
        res_quantile, res_quantile_qlo, res_quantile_qhi,
        res_raw, res_raw_qlo, res_raw_qhi,
        res_pearson, res_pearson_qlo, res_pearson_qhi,
        logscore, logscore_qlo, logscore_qhi,
        draw, n_draws)
end

function _maybe_with_mcmc_warmup(res::FitResult, mcmc_warmup::Union{Nothing, Int})
    return _with_posterior_warmup(res, mcmc_warmup)
end

function _ensure_obs_cache_nonmcmc(res::FitResult,
        dm::DataModel,
        cache::Union{Nothing, PlotCache},
        params::NamedTuple,
        constants_re::NamedTuple,
        ode_args::Tuple,
        ode_kwargs::NamedTuple,
        rng::AbstractRNG)
    if cache !== nothing && cache.obs_dists !== nothing
        return cache
    end
    return build_plot_cache(res; dm = dm, params = params, constants_re = constants_re,
        cache_obs_dists = true, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng)
end

"""
    get_residuals(res::FitResult; dm, cache, observables, individuals_idx, obs_rows,
                  x_axis_feature, params, constants_re, cache_obs_dists, residuals,
                  fitted_stat, randomize_discrete, cdf_fallback_mc, ode_args,
                  ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles, rng,
                  return_draw_level) -> DataFrame

    get_residuals(dm::DataModel; params, constants_re, observables, individuals_idx,
                  obs_rows, x_axis_feature, cache, cache_obs_dists, residuals,
                  fitted_stat, randomize_discrete, cdf_fallback_mc, ode_args,
                  ode_kwargs, rng) -> DataFrame

Compute residuals for each observation and return a `DataFrame`.

# Keyword Arguments
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
- `cache::Union{Nothing, PlotCache} = nothing`: pre-computed plot cache.
- `observables`: outcome name(s) to include, or `nothing` for all.
- `individuals_idx`: individuals to include, or `nothing` for all.
- `obs_rows`: specific observation row indices to include, or `nothing` for all.
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x column.
- `params::NamedTuple = NamedTuple()`: fixed-effect overrides.
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
- `cache_obs_dists::Bool = true`: cache observation distributions.
- `residuals`: residual metrics to compute. Allowed: `:pit`, `:quantile`, `:raw`,
  `:pearson`, `:logscore`.
- `fitted_stat = mean`: statistic applied to the predictive distribution for raw residuals.
- `randomize_discrete::Bool = true`: randomize PIT values for discrete outcomes.
- `cdf_fallback_mc::Int = 0`: MC samples for CDF approximation with non-analytic distributions.
- `ode_args::Tuple = ()`, `ode_kwargs::NamedTuple = NamedTuple()`: forwarded to ODE solver.
- `mcmc_draws::Int = 1000`, `mcmc_warmup`: MCMC draw settings.
- `mcmc_quantiles::Vector = [5, 95]`: percentiles for MCMC residual uncertainty bands.
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
- `return_draw_level::Bool = false`: if `true`, return draw-level residuals for MCMC.
"""
function get_residuals(res::FitResult;
        dm::Union{Nothing, DataModel} = nothing,
        cache::Union{Nothing, PlotCache} = nothing,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        residuals = [:quantile, :pit, :raw, :pearson, :logscore],
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        return_draw_level::Bool = false)
    dm = _get_dm(res, dm)
    constants_re_use = _res_constants_re(res, constants_re)
    residual_list = _validate_residual_metrics(residuals)
    obs_list = _resolve_residual_observables(dm, observables)
    inds = _resolve_residual_individuals(dm, individuals_idx)
    qvec = sort(Float64.(collect(mcmc_quantiles)))
    (length(qvec) >= 2 && all(0 .<= qvec .<= 100)) ||
        error("mcmc_quantiles must be in [0,100] with length >= 2.")
    qlo = qvec[1] / 100
    qhi = qvec[end] / 100

    x_axis_use = x_axis_feature
    if dm.model.de.de === nothing
        x_axis_use = _require_varying_covariate(dm, x_axis_feature)
    end

    rows = Vector{Any}()
    is_mcmc = _is_posterior_draw_fit(res)

    if is_mcmc
        mcmc_draws >= 1 || error("mcmc_draws must be >= 1.")
        res_use = _maybe_with_mcmc_warmup(res, mcmc_warmup)
        θ_draws, η_draws, _ = _posterior_drawn_params(
            res_use, dm, constants_re_use, params, mcmc_draws, rng)
        n_draws = length(θ_draws)
        isempty(θ_draws) && error("No posterior draws available for residual computation.")

        for i in inds
            ind = dm.individuals[i]
            obs_rows_all = dm.row_groups.obs_rows[i]
            obs_idx = _resolve_residual_obs_rows(obs_rows, obs_rows_all)
            xvals = _get_x_values(dm, ind, obs_rows_all, x_axis_use)
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

            sol_accessors_draw = Vector{Any}(undef, n_draws)
            for d in 1:n_draws
                θ = θ_draws[d]
                η_ind = η_draws[d][i]
                if dm.model.de.de === nothing
                    sol_accessors_draw[d] = nothing
                else
                    sol, compiled = _solve_dense_individual(
                        dm, ind, θ, η_ind; ode_args = ode_args, ode_kwargs = ode_kwargs)
                    sol_accessors_draw[d] = get_de_accessors_builder(dm.model.de.de)(
                        sol, compiled)
                end
            end

            for obs_name in obs_list
                yvals = getfield(ind.series.obs, obs_name)
                for j in obs_idx
                    row = obs_rows_all[j]
                    id_val = dm.df[row, dm.config.primary_id]
                    tval = _to_float_or_missing(dm.df[row, dm.config.time_col])
                    xval = _to_float_or_missing(xvals[j])
                    yval = _to_float_or_missing(yvals[j])

                    fitted_vals = Vector{Union{Missing, Float64}}(undef, n_draws)
                    pit_vals = Vector{Union{Missing, Float64}}(undef, n_draws)
                    q_vals = Vector{Union{Missing, Float64}}(undef, n_draws)
                    raw_vals = Vector{Union{Missing, Float64}}(undef, n_draws)
                    pearson_vals = Vector{Union{Missing, Float64}}(undef, n_draws)
                    ls_vals = Vector{Union{Missing, Float64}}(undef, n_draws)

                    for d in 1:n_draws
                        θ = θ_draws[d]
                        η_ind = η_draws[d][i]
                        sol_accessors = sol_accessors_draw[d]
                        vary = _varying_at_plot(dm, ind, j, row)
                        η_row = _row_random_effects_at(
                            dm, i, j, η_ind, rowwise_re; obs_only = true)
                        obs = sol_accessors === nothing ?
                              calculate_formulas_obs(
                            dm.model, θ, η_row, ind.const_cov, vary) :
                              calculate_formulas_obs(
                            dm.model, θ, η_row, ind.const_cov, vary, sol_accessors)
                        dist = getproperty(obs, obs_name)
                        met = _compute_residual_metrics(
                            dist, yval, residual_list, fitted_stat,
                            randomize_discrete, cdf_fallback_mc, rng)
                        fitted_vals[d] = met.fitted
                        pit_vals[d] = met.pit
                        q_vals[d] = met.res_quantile
                        raw_vals[d] = met.res_raw
                        pearson_vals[d] = met.res_pearson
                        ls_vals[d] = met.logscore
                    end

                    if return_draw_level
                        for d in 1:n_draws
                            push!(rows,
                                _residual_row(individual_idx = i, id = id_val, row = row,
                                    obs_index = j, observable = obs_name,
                                    time = tval, x = xval, y = yval, fitted = fitted_vals[d],
                                    pit = pit_vals[d], pit_qlo = missing, pit_qhi = missing,
                                    res_quantile = q_vals[d], res_quantile_qlo = missing, res_quantile_qhi = missing,
                                    res_raw = raw_vals[d], res_raw_qlo = missing, res_raw_qhi = missing,
                                    res_pearson = pearson_vals[d], res_pearson_qlo = missing, res_pearson_qhi = missing,
                                    logscore = ls_vals[d], logscore_qlo = missing, logscore_qhi = missing,
                                    draw = d, n_draws = n_draws))
                        end
                    else
                        fit_mean = collect(skipmissing(fitted_vals))
                        fitted = isempty(fit_mean) ? missing : Float64(mean(fit_mean))

                        pit_mean, pit_qlo, pit_qhi = _metric_summary(pit_vals, qlo, qhi)
                        q_mean, q_qlo, q_qhi = _metric_summary(q_vals, qlo, qhi)
                        raw_mean, raw_qlo, raw_qhi = _metric_summary(raw_vals, qlo, qhi)
                        p_mean, p_qlo, p_qhi = _metric_summary(pearson_vals, qlo, qhi)
                        ls_mean, ls_qlo, ls_qhi = _metric_summary(ls_vals, qlo, qhi)

                        push!(rows,
                            _residual_row(individual_idx = i, id = id_val, row = row,
                                obs_index = j, observable = obs_name,
                                time = tval, x = xval, y = yval, fitted = fitted,
                                pit = pit_mean, pit_qlo = pit_qlo, pit_qhi = pit_qhi,
                                res_quantile = q_mean, res_quantile_qlo = q_qlo, res_quantile_qhi = q_qhi,
                                res_raw = raw_mean, res_raw_qlo = raw_qlo, res_raw_qhi = raw_qhi,
                                res_pearson = p_mean, res_pearson_qlo = p_qlo, res_pearson_qhi = p_qhi,
                                logscore = ls_mean, logscore_qlo = ls_qlo, logscore_qhi = ls_qhi,
                                draw = missing, n_draws = n_draws))
                    end
                end
            end
        end
    else
        res_use = res
        cache_use = cache_obs_dists ?
                    _ensure_obs_cache_nonmcmc(
            res_use, dm, cache, params, constants_re_use, ode_args, ode_kwargs, rng) :
                    (cache === nothing ?
                     build_plot_cache(
            res_use; dm = dm, params = params, constants_re = constants_re_use,
            cache_obs_dists = false, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng) :
                     cache)

        for i in inds
            ind = dm.individuals[i]
            obs_rows_all = dm.row_groups.obs_rows[i]
            obs_idx = _resolve_residual_obs_rows(obs_rows, obs_rows_all)
            xvals = _get_x_values(dm, ind, obs_rows_all, x_axis_use)
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

            θ = cache_use.params
            η_ind = cache_use.random_effects[i]
            sol_accessors = nothing
            if dm.model.de.de !== nothing
                sol = cache_use.sols[i]
                compiled = get_de_compiler(dm.model.de.de)((;
                    fixed_effects = θ,
                    random_effects = η_ind,
                    constant_covariates = ind.const_cov,
                    varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
                    helpers = get_helper_funs(dm.model),
                    model_funs = get_model_funs(dm.model),
                    preDE = calculate_prede(dm.model, θ, η_ind, ind.const_cov)
                ))
                sol_accessors = get_de_accessors_builder(dm.model.de.de)(sol, compiled)
            end

            for obs_name in obs_list
                yvals = getfield(ind.series.obs, obs_name)
                # Non-cache path: do a full forward pass through all rows for HMM filtering,
                # storing filtered dists so obs_idx subsets get the correct filtered state.
                row_dists_res = if cache_use.obs_dists === nothing
                    hmm_priors_res = Dict{Symbol, Any}()
                    d_vec = Vector{Any}(undef, length(obs_rows_all))
                    for j in eachindex(obs_rows_all)
                        row_j = obs_rows_all[j]
                        vary_j = _varying_at_plot(dm, ind, j, row_j)
                        η_row_j = _row_random_effects_at(
                            dm, i, j, η_ind, rowwise_re; obs_only = true)
                        obs_j = sol_accessors === nothing ?
                                calculate_formulas_obs(
                            dm.model, θ, η_row_j, ind.const_cov, vary_j) :
                                calculate_formulas_obs(
                            dm.model, θ, η_row_j, ind.const_cov, vary_j, sol_accessors)
                        d_vec[j] = _apply_hmm_filter!(hmm_priors_res, obs_name,
                            getproperty(obs_j, obs_name), yvals[j])
                    end
                    d_vec
                else
                    nothing
                end
                for j in obs_idx
                    row = obs_rows_all[j]
                    id_val = dm.df[row, dm.config.primary_id]
                    tval = _to_float_or_missing(dm.df[row, dm.config.time_col])
                    xval = _to_float_or_missing(xvals[j])
                    yval = _to_float_or_missing(yvals[j])

                    dist = if cache_use.obs_dists !== nothing
                        getproperty(cache_use.obs_dists[i][j], obs_name)
                    else
                        row_dists_res[j]
                    end

                    met = _compute_residual_metrics(dist, yval, residual_list, fitted_stat,
                        randomize_discrete, cdf_fallback_mc, rng)
                    push!(rows,
                        _residual_row(individual_idx = i, id = id_val, row = row,
                            obs_index = j, observable = obs_name,
                            time = tval, x = xval, y = yval, fitted = met.fitted,
                            pit = met.pit, pit_qlo = missing, pit_qhi = missing,
                            res_quantile = met.res_quantile, res_quantile_qlo = missing, res_quantile_qhi = missing,
                            res_raw = met.res_raw, res_raw_qlo = missing, res_raw_qhi = missing,
                            res_pearson = met.res_pearson, res_pearson_qlo = missing, res_pearson_qhi = missing,
                            logscore = met.logscore, logscore_qlo = missing, logscore_qhi = missing,
                            draw = missing, n_draws = 1))
                end
            end
        end
    end

    if isempty(rows)
        return DataFrame(individual_idx = Int[], id = Any[], row = Int[],
            obs_index = Int[], observable = Symbol[],
            time = Union{Missing, Float64}[], x = Union{Missing, Float64}[], y = Union{
                Missing, Float64}[],
            fitted = Union{Missing, Float64}[],
            pit = Union{Missing, Float64}[], pit_qlo = Union{Missing, Float64}[], pit_qhi = Union{
                Missing, Float64}[],
            res_quantile = Union{Missing, Float64}[], res_quantile_qlo = Union{
                Missing, Float64}[],
            res_quantile_qhi = Union{Missing, Float64}[],
            res_raw = Union{Missing, Float64}[], res_raw_qlo = Union{Missing, Float64}[],
            res_raw_qhi = Union{Missing, Float64}[],
            res_pearson = Union{Missing, Float64}[], res_pearson_qlo = Union{
                Missing, Float64}[],
            res_pearson_qhi = Union{Missing, Float64}[],
            logscore = Union{Missing, Float64}[], logscore_qlo = Union{Missing, Float64}[],
            logscore_qhi = Union{Missing, Float64}[],
            draw = Union{Missing, Int}[], n_draws = Int[])
    end
    return DataFrame(rows)
end

function get_residuals(dm::DataModel;
        cache::Union{Nothing, PlotCache} = nothing,
        observables = nothing,
        individuals_idx = nothing,
        obs_rows = nothing,
        x_axis_feature::Union{Nothing, Symbol} = nothing,
        params::NamedTuple = NamedTuple(),
        constants_re::NamedTuple = NamedTuple(),
        cache_obs_dists::Bool = true,
        residuals = [:quantile, :pit, :raw, :pearson, :logscore],
        fitted_stat = mean,
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        mcmc_draws::Int = 1000,
        mcmc_warmup::Union{Nothing, Int} = nothing,
        mcmc_quantiles::Vector{<:Real} = [5, 95],
        rng::AbstractRNG = Random.default_rng(),
        return_draw_level::Bool = false)
    cache_use = cache
    if cache_use === nothing
        cache_use = build_plot_cache(dm; params = params, constants_re = constants_re,
            cache_obs_dists = cache_obs_dists, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng)
    elseif cache_obs_dists && cache_use.obs_dists === nothing
        # Rebuild obs distribution cache from DataModel inputs (starting parameters), not from a synthetic FitResult.
        cache_use = build_plot_cache(dm; params = params, constants_re = constants_re,
            cache_obs_dists = true, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng)
    end

    dummy_params = cache_use.params
    res = FitResult(MLE(), MLEResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
        FitSummary(0.0, true, FitParameters(dummy_params, dummy_params), NamedTuple()),
        FitDiagnostics((;), (;), (;), (;)), dm, (), (constants_re = constants_re,))

    return get_residuals(res; dm = dm, cache = cache_use, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re,
        cache_obs_dists = cache_obs_dists, residuals = residuals,
        fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles,
        rng = rng, return_draw_level = return_draw_level)
end

function _residual_kde_xy(
        vals::Vector{Float64}; bandwidth::Union{Nothing, Float64} = nothing)
    kd = bandwidth === nothing ? kde(vals) : kde(vals; bandwidth = bandwidth)
    return kd.x, kd.density
end

function _acf_for_series(v::Vector{Float64}, max_lag::Int)
    n = length(v)
    out = Vector{Union{Missing, Float64}}(undef, max_lag)
    if n < 2
        fill!(out, missing)
        return out
    end
    μ = mean(v)
    centered = v .- μ
    denom = sum(abs2, centered)
    if denom <= 0
        fill!(out, missing)
        return out
    end
    for lag in 1:max_lag
        if n - lag < 2
            out[lag] = missing
            continue
        end
        num = dot(view(centered, 1:(n - lag)), view(centered, (lag + 1):n))
        out[lag] = num / denom
    end
    return out
end
