export get_residuals
export predict
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
    metric in _RESIDUAL_ALLOWED ||
        error("Unknown residual metric $(metric). Allowed: $(_RESIDUAL_ALLOWED).")
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
    obs = get_formulas_meta(get_formulas(get_model(dm))).obs_names
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

@inline function _to_float_or_missing(v)
    if ismissing(v)
        return missing
    end
    if v isa Number
        x = Float64(v)
        return isfinite(x) ? x : missing
    end
    v isa AbstractVector &&
        @warn "Vector-valued entries have no scalar meaning in the `time`/`x` axis " *
        "columns and are reported as missing." maxlog = 1
    return missing
end

# Residual cells are scalar for univariate outcomes and per-component vectors for
# multivariate ones; joint scores (`logscore`) stay scalar in either case.
const _ResidualCell = Union{
    Missing, Float64, Vector{Float64}, Vector{Union{Missing, Float64}},
}

# Component vectors stay narrowly typed unless a component is actually missing.
@inline _narrow_components(v) =
    any(ismissing, v) ? convert(Vector{Union{Missing, Float64}}, v) :
    Float64[Float64(x) for x in v]

# Observation / fitted values keep their multivariate shape until after metric dispatch.
# Partially observed vectors keep their observed components rather than collapsing.
@inline function _residual_obs_value(v)
    ismissing(v) && return missing
    if v isa AbstractVector
        all(ismissing, v) && return missing
        return _narrow_components(
            Union{Missing, Float64}[ismissing(x) ? missing : Float64(x) for x in v]
        )
    end
    return _to_float_or_missing(v)
end

_residual_error(msg) = error("get_residuals: " * msg)

# Component marginals of a multivariate outcome distribution, or `nothing` when they
# are not available in closed form (component PIT/quantile is then refused).
function _mv_marginals(dist)
    m = _re_marginals(dist)
    m === nothing || return collect(m)
    # Shares the plotting component-marginal rule (MV HMMs, joint MvNormal emissions).
    m = _dist_marginals(dist)
    m === nothing || return collect(m)
    dist isa Distributions.AbstractMvNormal || return nothing
    return [Normal(μ, sqrt(v)) for (μ, v) in zip(mean(dist), diag(cov(dist)))]
end

function _metric_summary(vals::AbstractVector, qlo::Float64, qhi::Float64)
    vals_use = collect(skipmissing(vals))
    isempty(vals_use) && return (missing, missing, missing)
    if first(vals_use) isa AbstractVector
        n = length(first(vals_use))
        all(v -> length(v) == n, vals_use) ||
            _residual_error("component count differs across posterior draws.")
        m = reduce(hcat, vals_use)
        # Components can be missing independently across draws.
        comp = [collect(skipmissing(view(m, k, :))) for k in 1:n]
        summ(f) = _narrow_components(
            Union{Missing, Float64}[
                isempty(c) ? missing : Float64(f(c)) for c in comp
            ]
        )
        return (
            summ(mean),
            summ(c -> quantile(c, qlo)),
            summ(c -> quantile(c, qhi)),
        )
    end
    m = mean(vals_use)
    lo = quantile(vals_use, qlo)
    hi = quantile(vals_use, qhi)
    return (Float64(m), Float64(lo), Float64(hi))
end

function _pit_from_dist(
        dist,
        y::Union{Missing, Float64};
        randomize_discrete::Bool = true,
        cdf_fallback_mc::Int = 0,
        rng::AbstractRNG = Random.default_rng()
    )
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

function _mv_fitted(dist, fitted_stat)
    v = try
        fitted_stat === mean ? _re_mean(dist) : fitted_stat(dist)
    catch
        return missing
    end
    v isa AbstractMatrix && _residual_error(
        "fitted_stat returned a $(size(v, 1))x$(size(v, 2)) matrix for a multivariate " *
            "outcome; matrix-valued statistics (e.g. `cov`) have no per-observation " *
            "representation. Pass a vector-valued `fitted_stat` such as `mean`."
    )
    v isa AbstractVector && return Float64[Float64(x) for x in v]
    v isa Number && return Float64(v)
    return missing
end

# Per-component variances: `var`/`cov` diagonal, else the marginal variances.
function _mv_variances(dist, marg)
    v = try
        var(dist)
    catch
        nothing
    end
    v isa AbstractVector && return Float64[Float64(x) for x in v]
    c = try
        cov(dist)
    catch
        nothing
    end
    c isa AbstractMatrix && return Float64[Float64(x) for x in diag(c)]
    marg === nothing && return nothing
    return Float64[Float64(var(m)) for m in marg]
end

function _compute_mv_residual_metrics(
        dist,
        y::AbstractVector,
        residual_list::Vector{Symbol},
        fitted_stat,
        randomize_discrete::Bool,
        cdf_fallback_mc::Int,
        rng::AbstractRNG
    )
    req = Set(residual_list)
    fitted = _mv_fitted(dist, fitted_stat)

    pit = missing
    res_quantile = missing
    if (:pit in req) || (:quantile in req)
        marg = _mv_marginals(dist)
        marg === nothing && _residual_error(
            "PIT / quantile residuals for a multivariate outcome of type " *
                "$(nameof(typeof(dist))) require known component marginals, and a joint " *
                "PIT has no canonical definition. Request only " *
                "`[:raw, :pearson, :logscore]` for this outcome."
        )
        length(marg) == length(y) || _residual_error(
            "the outcome has $(length(y)) components but its distribution reports " *
                "$(length(marg)) marginals."
        )
        pit_vec = Union{Missing, Float64}[
            _pit_from_dist(
                    marg[k], y[k]; randomize_discrete = randomize_discrete,
                    cdf_fallback_mc = cdf_fallback_mc, rng = rng
                ) for k in eachindex(y)
        ]
        all(ismissing, pit_vec) || (pit = _narrow_components(pit_vec))
        if (:quantile in req) && !ismissing(pit)
            res_quantile = _narrow_components(
                Union{Missing, Float64}[
                    ismissing(p) ? missing :
                        quantile(Normal(), clamp(p, eps(Float64), 1.0 - eps(Float64)))
                        for p in pit
                ]
            )
        end
        (:pit in req) || (pit = missing)
    end

    μ = (:raw in req) || (:pearson in req) ? _mv_fitted(dist, mean) : missing
    μ isa AbstractVector && length(μ) != length(y) && _residual_error(
        "the outcome has $(length(y)) components but its distribution mean has " *
            "$(length(μ))."
    )

    res_raw = (:raw in req) && μ isa AbstractVector ? y .- μ : missing

    res_pearson = missing
    if (:pearson in req) && μ isa AbstractVector
        v = _mv_variances(dist, _mv_marginals(dist))
        if v !== nothing && length(v) == length(y) && all(x -> x > 0 && isfinite(x), v)
            res_pearson = (y .- μ) ./ sqrt.(v)
        end
    end

    logscore = missing
    # The joint score needs every component; a partial vector has no joint density.
    if (:logscore in req) && !any(ismissing, y)
        ls = try
            -Float64(logpdf(dist, Float64[y...]))
        catch
            missing
        end
        # `Inf` is a real score for an impossible observation; only `NaN` means failure.
        !ismissing(ls) && !isnan(ls) && (logscore = ls)
    end

    return (
        fitted = fitted, pit = pit, res_quantile = res_quantile,
        res_raw = res_raw, res_pearson = res_pearson, logscore = logscore,
    )
end

function _compute_residual_metrics(
        dist,
        y::_ResidualCell,
        residual_list::Vector{Symbol},
        fitted_stat,
        randomize_discrete::Bool,
        cdf_fallback_mc::Int,
        rng::AbstractRNG
    )
    if y isa AbstractVector
        return _compute_mv_residual_metrics(
            dist, y, residual_list, fitted_stat, randomize_discrete, cdf_fallback_mc, rng
        )
    elseif ismissing(y) && dist isa Distributions.MultivariateDistribution
        return (
            fitted = _mv_fitted(dist, fitted_stat), pit = missing, res_quantile = missing,
            res_raw = missing, res_pearson = missing, logscore = missing,
        )
    end
    req = Set(residual_list)
    fitted = try
        v = _stat_from_dist(dist, fitted_stat)
        v isa Number ? Float64(v) : missing
    catch
        missing
    end

    pit = missing
    if (:pit in req) || (:quantile in req)
        pit = _pit_from_dist(
            dist, y; randomize_discrete = randomize_discrete,
            cdf_fallback_mc = cdf_fallback_mc, rng = rng
        )
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
            # `Inf` is a real score for an impossible observation; only `NaN` means failure.
            isnan(ls) || (logscore = ls)
        end
    end

    return (
        fitted = fitted, pit = pit, res_quantile = res_quantile,
        res_raw = res_raw, res_pearson = res_pearson, logscore = logscore,
    )
end

function _residual_row(;
        individual_idx::Int,
        id,
        row::Int,
        obs_index::Int,
        observable::Symbol,
        time::Union{Missing, Float64},
        x::Union{Missing, Float64},
        y::_ResidualCell,
        fitted::_ResidualCell,
        pit::_ResidualCell,
        pit_qlo::_ResidualCell,
        pit_qhi::_ResidualCell,
        res_quantile::_ResidualCell,
        res_quantile_qlo::_ResidualCell,
        res_quantile_qhi::_ResidualCell,
        res_raw::_ResidualCell,
        res_raw_qlo::_ResidualCell,
        res_raw_qhi::_ResidualCell,
        res_pearson::_ResidualCell,
        res_pearson_qlo::_ResidualCell,
        res_pearson_qhi::_ResidualCell,
        logscore::Union{Missing, Float64},
        logscore_qlo::Union{Missing, Float64},
        logscore_qhi::Union{Missing, Float64},
        draw::Union{Missing, Int},
        n_draws::Int
    )
    return (;
        individual_idx, id, row, obs_index, observable, time, x, y, fitted,
        pit, pit_qlo, pit_qhi,
        res_quantile, res_quantile_qlo, res_quantile_qhi,
        res_raw, res_raw_qlo, res_raw_qhi,
        res_pearson, res_pearson_qlo, res_pearson_qhi,
        logscore, logscore_qlo, logscore_qhi,
        draw, n_draws,
    )
end

function _ensure_obs_cache_nonmcmc(
        res::FitResult,
        dm::DataModel,
        cache::Union{Nothing, PlotCache},
        params::NamedTuple,
        constants_re::NamedTuple,
        ode_args::Tuple,
        ode_kwargs::NamedTuple,
        rng::AbstractRNG
    )
    if cache !== nothing && cache.obs_dists !== nothing
        return cache
    end
    return build_plot_cache(
        res; dm = dm, params = params, constants_re = constants_re,
        cache_obs_dists = true, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng
    )
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

# `:logscore` sign and relation to `get_loglikelihood`

`logscore` is the **negative** log predictive density of the observation,
`-logpdf(dist, y)` (smaller is better), so `-sum(skipmissing(rdf.logscore))` is the
conditional log-likelihood at the random effects the residuals were evaluated at. For
HMM-family outcomes `dist` is the forward-filtered distribution, so this identity holds
through `missing` rows as well. On Laplace/FOCEI/SAEM/MCEM/Pooled fits it therefore
matches `get_loglikelihood(res)` (which also conditions on the EB modes) to round-off;
on `GHQuadrature` fits `get_loglikelihood` returns the *marginal* likelihood, which
integrates over the random effects and is a different quantity. An observation the model
assigns zero probability scores `Inf`; `missing` means the score could not be computed.

# Multivariate (vector-valued) outcomes

Vector-valued observations keep one row per observation; the `y`, `fitted`, `res_raw`,
`res_pearson`, `pit` and `res_quantile` cells (and their `_qlo` / `_qhi` bands) then hold
`Vector{Float64}` with one entry per outcome component, in the order of the observation.
`logscore` stays scalar: it is the joint score `-logpdf(dist, y)`. `time` and `x` remain
scalar. Scalar outcomes are unaffected and keep scalar cells throughout.

A partially observed vector keeps its observed components: the cell type widens to
`Vector{Union{Missing, Float64}}` and only the missing components are `missing`. Such a
row has no `logscore`, since the joint density needs every component. A row is only
wholly `missing` when every component is.

Component `pit` / `res_quantile` need the component marginals of the outcome
distribution (known for `MvNormal` and for distributions with an `_re_marginals`
method, such as `Copulas.SklarDist`); requesting them for a distribution without known
marginals throws, because a joint PIT has no canonical definition. A matrix-valued
`fitted_stat` such as `cov` also throws. Statistics a distribution simply does not
implement (e.g. no `mean`) stay `missing`, as for scalar outcomes.

# HMM-family outcomes

For HMM / observed-states outcomes every column is an **emission-level** quantity of the
forward-filtered mixture distribution, not a hidden-state summary: `fitted` is the mixture
mean (for a Bernoulli emission, a probability near the mixture average, not a state
probability), and `pit`/`res_*` are the usual mixture-distribution residuals, which for a
discrete outcome sit on the CDF steps. Use [`plot_hidden_states`](@ref) or
`posterior_hidden_states` for state-level diagnostics.
"""
function get_residuals(
        res::FitResult;
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
        return_draw_level::Bool = false
    )
    dm = _get_dm(res, dm)
    # Residuals of a fit whose objective is not finite are meaningless and used to fail
    # with an unrelated internal error from a NaN parameter vector (#212).
    let obj = get_objective(res)
        obj isa Real && !isfinite(obj) &&
            error("Cannot compute residuals: the fit objective is $(obj). Fix the fit (non-finite data, starting values, or an out-of-domain distribution argument) before running diagnostics.")
    end
    constants_re_use = _res_constants_re(res, constants_re, dm)
    residual_list = _validate_residual_metrics(residuals)
    obs_list = _resolve_residual_observables(dm, observables)
    inds = _resolve_individuals(dm, individuals_idx; default_all = true)
    qvec = sort(Float64.(collect(mcmc_quantiles)))
    (length(qvec) >= 2 && all(0 .<= qvec .<= 100)) ||
        error("mcmc_quantiles must be in [0,100] with length >= 2.")
    qlo = qvec[1] / 100
    qhi = qvec[end] / 100

    x_axis_use = x_axis_feature
    if get_de(get_model(dm)) === nothing
        x_axis_use = _require_varying_covariate(dm, x_axis_feature)
    end

    rows = Vector{Any}()
    is_mcmc = _is_posterior_draw_fit(res)

    if is_mcmc
        mcmc_draws >= 1 || error("mcmc_draws must be >= 1.")
        res_use = _with_posterior_warmup(res, mcmc_warmup)
        θ_draws, η_draws, _ = _posterior_drawn_params(
            res_use, dm, constants_re_use, params, mcmc_draws, rng
        )
        n_draws = length(θ_draws)
        isempty(θ_draws) && error("No posterior draws available for residual computation.")

        for i in inds
            ind = get_individuals(dm)[i]
            obs_rows_all = get_obs_rows(get_row_groups(dm))[i]
            obs_idx = _resolve_obs_rows(obs_rows, obs_rows_all)
            xvals = _get_x_values(dm, ind, obs_rows_all, x_axis_use)
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

            sol_accessors_draw = Vector{Any}(undef, n_draws)
            for d in 1:n_draws
                θ = θ_draws[d]
                η_ind = η_draws[d][i]
                if get_de(get_model(dm)) === nothing
                    sol_accessors_draw[d] = nothing
                else
                    sol, compiled = _solve_dense_individual(
                        dm, ind, θ, η_ind; ode_args = ode_args, ode_kwargs = ode_kwargs
                    )
                    sol_accessors_draw[d] = _sol_accessors_with_crossings(
                        get_model(dm), sol, compiled, θ, η_ind, get_const_cov(ind)
                    )
                end
            end

            for obs_name in obs_list
                yvals = getfield(get_obs(get_series(ind)), obs_name)
                # Full forward pass over ALL rows per draw, so HMM outcomes are
                # conditioned on the filtered posterior from the preceding
                # observations — mirrors the non-MCMC path below; for non-HMM
                # outcomes `_apply_hmm_filter!` is a passthrough. `obs_idx`
                # subsets the filtered distributions afterwards.
                dists_by_draw = Vector{Vector{Any}}(undef, n_draws)
                for d in 1:n_draws
                    θ = θ_draws[d]
                    η_ind = η_draws[d][i]
                    sol_accessors = sol_accessors_draw[d]
                    hmm_priors_d = Dict{Symbol, Any}()
                    d_vec = Vector{Any}(undef, length(obs_rows_all))
                    for j in eachindex(obs_rows_all)
                        row_j = obs_rows_all[j]
                        vary_j = _varying_at(dm, ind, j, row_j)
                        η_row_j = _row_random_effects_at(
                            dm, i, j, η_ind, rowwise_re; obs_only = true
                        )
                        obs_j = sol_accessors === nothing ?
                            calculate_formulas_obs(
                                get_model(dm), θ, η_row_j, get_const_cov(ind), vary_j
                            ) :
                            calculate_formulas_obs(
                                get_model(dm), θ, η_row_j, get_const_cov(ind), vary_j, sol_accessors
                            )
                        d_vec[j] = _apply_hmm_filter!(
                            hmm_priors_d, obs_name,
                            getproperty(obs_j, obs_name), yvals[j]
                        )
                    end
                    dists_by_draw[d] = d_vec
                end
                for j in obs_idx
                    row = obs_rows_all[j]
                    id_val = get_df(dm)[row, get_primary_id(dm)]
                    tval = _to_float_or_missing(get_df(dm)[row, get_time_col(dm)])
                    xval = _to_float_or_missing(xvals[j])
                    yval = _residual_obs_value(yvals[j])

                    fitted_vals = Vector{_ResidualCell}(undef, n_draws)
                    pit_vals = Vector{_ResidualCell}(undef, n_draws)
                    q_vals = Vector{_ResidualCell}(undef, n_draws)
                    raw_vals = Vector{_ResidualCell}(undef, n_draws)
                    pearson_vals = Vector{_ResidualCell}(undef, n_draws)
                    ls_vals = Vector{_ResidualCell}(undef, n_draws)

                    for d in 1:n_draws
                        met = _compute_residual_metrics(
                            dists_by_draw[d][j], yval, residual_list, fitted_stat,
                            randomize_discrete, cdf_fallback_mc, rng
                        )
                        fitted_vals[d] = met.fitted
                        pit_vals[d] = met.pit
                        q_vals[d] = met.res_quantile
                        raw_vals[d] = met.res_raw
                        pearson_vals[d] = met.res_pearson
                        ls_vals[d] = met.logscore
                    end

                    if return_draw_level
                        for d in 1:n_draws
                            push!(
                                rows,
                                _residual_row(
                                    individual_idx = i, id = id_val, row = row,
                                    obs_index = j, observable = obs_name,
                                    time = tval, x = xval, y = yval, fitted = fitted_vals[d],
                                    pit = pit_vals[d], pit_qlo = missing, pit_qhi = missing,
                                    res_quantile = q_vals[d], res_quantile_qlo = missing, res_quantile_qhi = missing,
                                    res_raw = raw_vals[d], res_raw_qlo = missing, res_raw_qhi = missing,
                                    res_pearson = pearson_vals[d], res_pearson_qlo = missing, res_pearson_qhi = missing,
                                    logscore = ls_vals[d], logscore_qlo = missing, logscore_qhi = missing,
                                    draw = d, n_draws = n_draws
                                )
                            )
                        end
                    else
                        fitted, _, _ = _metric_summary(fitted_vals, qlo, qhi)

                        pit_mean, pit_qlo, pit_qhi = _metric_summary(pit_vals, qlo, qhi)
                        q_mean, q_qlo, q_qhi = _metric_summary(q_vals, qlo, qhi)
                        raw_mean, raw_qlo, raw_qhi = _metric_summary(raw_vals, qlo, qhi)
                        p_mean, p_qlo, p_qhi = _metric_summary(pearson_vals, qlo, qhi)
                        ls_mean, ls_qlo, ls_qhi = _metric_summary(ls_vals, qlo, qhi)

                        push!(
                            rows,
                            _residual_row(
                                individual_idx = i, id = id_val, row = row,
                                obs_index = j, observable = obs_name,
                                time = tval, x = xval, y = yval, fitted = fitted,
                                pit = pit_mean, pit_qlo = pit_qlo, pit_qhi = pit_qhi,
                                res_quantile = q_mean, res_quantile_qlo = q_qlo, res_quantile_qhi = q_qhi,
                                res_raw = raw_mean, res_raw_qlo = raw_qlo, res_raw_qhi = raw_qhi,
                                res_pearson = p_mean, res_pearson_qlo = p_qlo, res_pearson_qhi = p_qhi,
                                logscore = ls_mean, logscore_qlo = ls_qlo, logscore_qhi = ls_qhi,
                                draw = missing, n_draws = n_draws
                            )
                        )
                    end
                end
            end
        end
    else
        res_use = res
        cache_use = cache_obs_dists ?
            _ensure_obs_cache_nonmcmc(
                res_use, dm, cache, params, constants_re_use, ode_args, ode_kwargs, rng
            ) :
            (
                cache === nothing ?
                build_plot_cache(
                    res_use; dm = dm, params = params, constants_re = constants_re_use,
                    cache_obs_dists = false, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng
                ) :
                cache
            )

        for i in inds
            ind = get_individuals(dm)[i]
            obs_rows_all = get_obs_rows(get_row_groups(dm))[i]
            obs_idx = _resolve_obs_rows(obs_rows, obs_rows_all)
            xvals = _get_x_values(dm, ind, obs_rows_all, x_axis_use)
            rowwise_re = _needs_rowwise_random_effects(dm, i; obs_only = true)

            θ = cache_use.params
            η_ind = cache_use.random_effects[i]
            sol_accessors = nothing
            if get_de(get_model(dm)) !== nothing
                sol = cache_use.sols[i]
                sol_accessors = _sol_accessors_from_cached(dm, ind, sol, θ, η_ind)
            end

            for obs_name in obs_list
                yvals = getfield(get_obs(get_series(ind)), obs_name)
                # Non-cache path: do a full forward pass through all rows for HMM filtering,
                # storing filtered dists so obs_idx subsets get the correct filtered state.
                row_dists_res = if cache_use.obs_dists === nothing
                    hmm_priors_res = Dict{Symbol, Any}()
                    d_vec = Vector{Any}(undef, length(obs_rows_all))
                    for j in eachindex(obs_rows_all)
                        row_j = obs_rows_all[j]
                        vary_j = _varying_at(dm, ind, j, row_j)
                        η_row_j = _row_random_effects_at(
                            dm, i, j, η_ind, rowwise_re; obs_only = true
                        )
                        obs_j = sol_accessors === nothing ?
                            calculate_formulas_obs(
                                get_model(dm), θ, η_row_j, get_const_cov(ind), vary_j
                            ) :
                            calculate_formulas_obs(
                                get_model(dm), θ, η_row_j, get_const_cov(ind), vary_j, sol_accessors
                            )
                        d_vec[j] = _apply_hmm_filter!(
                            hmm_priors_res, obs_name,
                            getproperty(obs_j, obs_name), yvals[j]
                        )
                    end
                    d_vec
                else
                    nothing
                end
                for j in obs_idx
                    row = obs_rows_all[j]
                    id_val = get_df(dm)[row, get_primary_id(dm)]
                    tval = _to_float_or_missing(get_df(dm)[row, get_time_col(dm)])
                    xval = _to_float_or_missing(xvals[j])
                    yval = _residual_obs_value(yvals[j])

                    dist = if cache_use.obs_dists !== nothing
                        getproperty(cache_use.obs_dists[i][j], obs_name)
                    else
                        row_dists_res[j]
                    end

                    met = _compute_residual_metrics(
                        dist, yval, residual_list, fitted_stat,
                        randomize_discrete, cdf_fallback_mc, rng
                    )
                    push!(
                        rows,
                        _residual_row(
                            individual_idx = i, id = id_val, row = row,
                            obs_index = j, observable = obs_name,
                            time = tval, x = xval, y = yval, fitted = met.fitted,
                            pit = met.pit, pit_qlo = missing, pit_qhi = missing,
                            res_quantile = met.res_quantile, res_quantile_qlo = missing, res_quantile_qhi = missing,
                            res_raw = met.res_raw, res_raw_qlo = missing, res_raw_qhi = missing,
                            res_pearson = met.res_pearson, res_pearson_qlo = missing, res_pearson_qhi = missing,
                            logscore = met.logscore, logscore_qlo = missing, logscore_qhi = missing,
                            draw = missing, n_draws = 1
                        )
                    )
                end
            end
        end
    end

    if isempty(rows)
        return DataFrame(
            individual_idx = Int[], id = Any[], row = Int[],
            obs_index = Int[], observable = Symbol[],
            time = Union{Missing, Float64}[], x = Union{Missing, Float64}[], y = Union{
                Missing, Float64,
            }[],
            fitted = Union{Missing, Float64}[],
            pit = Union{Missing, Float64}[], pit_qlo = Union{Missing, Float64}[], pit_qhi = Union{
                Missing, Float64,
            }[],
            res_quantile = Union{Missing, Float64}[], res_quantile_qlo = Union{
                Missing, Float64,
            }[],
            res_quantile_qhi = Union{Missing, Float64}[],
            res_raw = Union{Missing, Float64}[], res_raw_qlo = Union{Missing, Float64}[],
            res_raw_qhi = Union{Missing, Float64}[],
            res_pearson = Union{Missing, Float64}[], res_pearson_qlo = Union{
                Missing, Float64,
            }[],
            res_pearson_qhi = Union{Missing, Float64}[],
            logscore = Union{Missing, Float64}[], logscore_qlo = Union{Missing, Float64}[],
            logscore_qhi = Union{Missing, Float64}[],
            draw = Union{Missing, Int}[], n_draws = Int[]
        )
    end
    return DataFrame(rows)
end

function get_residuals(
        dm::DataModel;
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
        return_draw_level::Bool = false
    )
    cache_use = cache
    if cache_use === nothing
        cache_use = build_plot_cache(
            dm; params = params, constants_re = constants_re,
            cache_obs_dists = cache_obs_dists, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng
        )
    elseif cache_obs_dists && cache_use.obs_dists === nothing
        # Rebuild obs distribution cache from DataModel inputs (starting parameters), not from a synthetic FitResult.
        cache_use = build_plot_cache(
            dm; params = params, constants_re = constants_re,
            cache_obs_dists = true, ode_args = ode_args, ode_kwargs = ode_kwargs, rng = rng
        )
    end

    dummy_params = cache_use.params
    res = FitResult(
        MLE(), FrequentistResult(NamedTuple(), 0.0, 0, NamedTuple(), NamedTuple()),
        FitSummary(0.0, true, FitParameters(dummy_params, dummy_params), NamedTuple()),
        FitDiagnostics((;), (;), (;), (;)), dm, (), (constants_re = constants_re,)
    )

    return get_residuals(
        res; dm = dm, cache = cache_use, observables = observables,
        individuals_idx = individuals_idx, obs_rows = obs_rows,
        x_axis_feature = x_axis_feature, params = params, constants_re = constants_re,
        cache_obs_dists = cache_obs_dists, residuals = residuals,
        fitted_stat = fitted_stat, randomize_discrete = randomize_discrete,
        cdf_fallback_mc = cdf_fallback_mc, ode_args = ode_args, ode_kwargs = ode_kwargs,
        mcmc_draws = mcmc_draws, mcmc_warmup = mcmc_warmup, mcmc_quantiles = mcmc_quantiles,
        rng = rng, return_draw_level = return_draw_level
    )
end

function _acf_for_series(v::Vector{Float64}, max_lag::Int)
    _check_positive_int(max_lag, "max_lag")
    n = length(v)
    out = Vector{Union{Missing, Float64}}(undef, max_lag)
    if n < 2
        fill!(out, missing)
        return out
    end
    max_lag >= n &&
        @warn "max_lag=$(max_lag) is not smaller than the series length $(n); lags with fewer than 2 usable pairs are reported as missing." maxlog = 1
    μ = mean(v)
    centered = v .- μ
    denom = sum(abs2, centered)
    if denom <= 0
        @warn "Autocorrelation is undefined for a zero-variance residual series; all lags are missing." maxlog = 1
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

"""
    predict(res::FitResult, newdata; re_mode = :population, fitted_stat = mean, kwargs...)
    predict(res::FitResult, newdata::DataModel; re_mode = :population, fitted_stat = mean, kwargs...)

Predict the response on new data from a fitted model. The fixed effects are taken
from `res`; how the random effects are chosen is controlled by `re_mode`:

- `:population` (default): random effects at their prior mean (the typical-subject
  "PRED"), so the predictions apply to previously unseen subjects.
- `:ebe`: reuse the empirical-Bayes estimate from the fit for any subject whose
  random-effect grouping signature is present in the training data (the individual
  "IPRED"); subjects not seen in training fall back to the population value.
- `:reestimate`: compute a fresh empirical-Bayes estimate on `newdata` while holding
  the fitted fixed effects, so new subjects that carry their own observations get an
  individual prediction.
- `:marginal`: integrate the random effects out over their prior by Monte Carlo — the
  prediction is the average conditional mean over `marginal_draws` prior draws. This
  is subject-agnostic like `:population` and differs from it only through the
  nonlinearity of the model (`E[f(η)]` vs `f(E[η])`); use `:ebe`/`:reestimate` for
  subject-specific predictions.

Matching in `:ebe` is on the whole random-effect grouping signature, so in
a hierarchical model a subject with a new primary id counts as unseen even if it shares
a known upper level. `:ebe`/`:reestimate`/`:marginal` need a random-effects method
(Laplace, FOCEI, GHQuadrature, MCEM, or SAEM; `:reestimate` excludes GHQuadrature) and
are not available for MCMC/VI posterior-draw fits (whose `:population` path already
integrates the posterior).

Pass `newdata` as a `DataFrame`, rebuilt into a `DataModel` using the fitted model and
the grouping, time and event columns of the original fit, or as a ready-made
`DataModel`. Returns a `DataFrame` with the individual identifier, the time, the
observable, and the predicted response in the `prediction` column. For a multivariate
outcome each `prediction` cell is a `Vector{Float64}` with one entry per outcome
component. Extra keyword
arguments are forwarded to [`get_residuals`](@ref).

# Keyword Arguments
- `re_mode::Symbol = :population`: random-effect strategy (see above).
- `fitted_stat = mean`: summary of the predicted observation distribution (`:marginal`
  is exact only for the mean).
- `constants_re::NamedTuple = NamedTuple()`: fix random-effect levels on the natural scale.
- `marginal_draws::Int = 100`: Monte Carlo draws for `re_mode = :marginal`.
- `reestimate_kwargs::NamedTuple = NamedTuple()`: forwarded to [`reestimate_ebes`](@ref).
- `rng::AbstractRNG = Random.default_rng()`: random source for `:marginal`.
"""
function predict(
        res::FitResult, dm_new::DataModel;
        re_mode::Symbol = :population,
        fitted_stat = mean,
        constants_re::NamedTuple = NamedTuple(),
        marginal_draws::Int = 100,
        reestimate_kwargs::NamedTuple = NamedTuple(),
        rng::AbstractRNG = Random.default_rng(),
        ode_args::Tuple = (),
        ode_kwargs::NamedTuple = NamedTuple(),
        kwargs...
    )
    # t0 is baked into every individual's tspan at DataModel construction, so a
    # dm_new built with a different t0 silently integrates from the wrong start (#148).
    dm_old = get_data_model(res)
    if dm_old !== nothing && !isequal(get_t0(dm_old), get_t0(dm_new))
        error(
            "predict: dm_new was built with t0 = $(get_t0(dm_new)), but the fit " *
                "used t0 = $(get_t0(dm_old)). Rebuild it as DataModel(...; t0 = " *
                "$(get_t0(dm_old))), or pass the new data as a DataFrame, which " *
                "reuses the fit's t0 automatically."
        )
    end
    θ = get_params(res; scale = :untransformed)
    if re_mode == :population
        constants_re = _res_constants_re(res, constants_re, dm_new)
        df = get_residuals(
            dm_new; params = NamedTuple(θ), residuals = [:raw],
            fitted_stat = fitted_stat, constants_re = constants_re,
            ode_args = ode_args, ode_kwargs = ode_kwargs, kwargs...
        )
        return _prediction_frame(df.id, df.time, df.observable, df.fitted)
    end
    _validate_predict_re_mode(res, dm_new, re_mode)
    if re_mode == :marginal
        return _predict_marginal(
            dm_new, θ; fitted_stat = fitted_stat,
            constants_re = constants_re, marginal_draws = marginal_draws, rng = rng,
            ode_args = ode_args, ode_kwargs = ode_kwargs, kwargs...
        )
    end
    if re_mode == :reestimate
        re_kw = merge(
            (
                constants_re = constants_re, ode_args = ode_args,
                ode_kwargs = ode_kwargs,
            ),
            reestimate_kwargs
        )
        res2 = reestimate_ebes(dm_new, res; re_kw...)
        df = get_residuals(
            res2; dm = dm_new, params = NamedTuple(θ), residuals = [:raw],
            fitted_stat = fitted_stat, constants_re = constants_re,
            ode_args = ode_args, ode_kwargs = ode_kwargs, kwargs...
        )
        return _prediction_frame(df.id, df.time, df.observable, df.fitted)
    end
    # :ebe
    η_vec = _predict_eta_ebe(res, dm_new, θ, constants_re)
    cache = _fill_plot_cache(
        dm_new, θ, η_vec, constants_re, true,
        ode_args, ode_kwargs
    )
    df = get_residuals(
        dm_new; cache = cache, params = NamedTuple(θ), residuals = [:raw],
        fitted_stat = fitted_stat, constants_re = constants_re,
        ode_args = ode_args, ode_kwargs = ode_kwargs, kwargs...
    )
    return _prediction_frame(df.id, df.time, df.observable, df.fitted)
end

function predict(res::FitResult, newdata; kwargs...)
    dm_old = get_data_model(res)
    dm_old === nothing &&
        error(
        "predict requires the fit to store its DataModel; refit with " *
            "store_data_model = true."
    )
    cfg = dm_old.config
    dm_new = DataModel(
        get_model(dm_old), newdata;
        primary_id = cfg.primary_id, time_col = cfg.time_col,
        evid_col = cfg.evid_col, amt_col = cfg.amt_col,
        rate_col = cfg.rate_col, cmt_col = cfg.cmt_col, t0 = cfg.t0,
        serialization = cfg.serialization
    )
    return predict(res, dm_new; kwargs...)
end

function _prediction_frame(id, time, observable, prediction)
    return DataFrame(id = id, time = time, observable = observable, prediction = prediction)
end

# Validate that re_mode is supported for this fit; :population is handled before this.
function _validate_predict_re_mode(res::FitResult, dm::DataModel, re_mode::Symbol)
    re_mode in (:population, :ebe, :reestimate, :marginal) ||
        error(
        "predict: re_mode must be :population, :ebe, :reestimate, or :marginal; " *
            "got :$re_mode."
    )
    isempty(get_re_names(get_random(get_model(dm)))) &&
        error(
        "predict: re_mode=:$re_mode requires a model with random effects; " *
            "use re_mode=:population."
    )
    _is_posterior_draw_fit(res) &&
        error(
        "predict: re_mode=:$re_mode is not supported for MCMC/VI posterior-draw " *
            "fits; use re_mode=:population (it integrates the posterior draws)."
    )
    if re_mode == :reestimate
        (
            get_result(res) isa FrequentistREResult || get_result(res) isa MCEMResult ||
                get_result(res) isa SAEMResult
        ) ||
            error(
            "predict: re_mode=:reestimate requires a Laplace, FOCEI, MCEM, or SAEM " *
                "fit (GHQuadrature is unsupported); use re_mode=:ebe instead."
        )
    else
        _cv_has_re_support(res) ||
            error(
            "predict: re_mode=:$re_mode requires a Laplace, FOCEI, GHQuadrature, " *
                "MCEM, or SAEM fit; got $(typeof(get_result(res)))."
        )
    end
    return nothing
end

# Build the per-individual η for :ebe — training EBE for seen subjects (matched on the
# whole re_groups signature), RE prior mean for the rest. Mirrors _cv_evaluate_ebe.
function _predict_eta_ebe(
        res::FitResult, dm_new::DataModel, θ::ComponentArray,
        constants_re::NamedTuple
    )
    dm_old = get_data_model(res)
    dm_old === nothing &&
        error(
        "predict: re_mode=:ebe requires the fit to store its DataModel; " *
            "refit with store_data_model = true."
    )
    η_vec = _default_random_effects_from_dm(dm_new, constants_re, θ)
    bstars, batch_infos, θu, const_cache, _, _ = _resolve_bstars_for_re(
        dm_old, res, constants_re
    )
    η_train = _eta_from_eb(dm_old, batch_infos, bstars, const_cache, θu)
    re_to_eta = Dict{Any, ComponentArray}(
        get_re_groups(get_individuals(dm_old)[i]) => η_train[i]
            for i in 1:length(get_individuals(dm_old))
    )
    for j in 1:length(get_individuals(dm_new))
        key = get_re_groups(get_individuals(dm_new)[j])
        haskey(re_to_eta, key) && (η_vec[j] = re_to_eta[key])
    end
    return η_vec
end

# Monte-Carlo marginal prediction: integrate the random effects over their prior,
# averaging the per-draw predicted means.
function _predict_marginal(
        dm_new::DataModel, θ::ComponentArray;
        fitted_stat, constants_re::NamedTuple, marginal_draws::Int,
        rng::AbstractRNG, ode_args::Tuple, ode_kwargs::NamedTuple, kwargs...
    )
    marginal_draws >= 1 || error("predict: marginal_draws must be >= 1.")
    dists_builder = create_random_effect_distribution(get_random(get_model(dm_new)))
    model_funs = get_model_funs(get_model(dm_new))
    helpers = get_helper_funs(get_model(dm_new))
    re_names = get_re_names(get_random(get_model(dm_new)))
    # Draw one value per free RE level and assemble per individual (same shape logic
    # as :population). A per-individual scalar draw breaks crossed designs where a
    # grouping column varies within an individual (#152), and drawing by level also
    # shares a draw between individuals on the same level and keeps constants_re
    # levels fixed.
    fixed_maps = _normalize_constants_re(dm_new, constants_re)
    re_meta = _re_free_meta(
        dm_new, θ, fixed_maps, re_names, dists_builder, model_funs, helpers
    )
    level_dims = Dict{Symbol, Int}(re => re_meta[re].dim for re in re_names)
    sample_rngs = _spawn_child_rngs(rng, marginal_draws)
    n_new = length(get_individuals(dm_new))

    # Accumulators are untyped so multivariate fitted values stay per-component vectors.
    sum_acc = Any[]
    cnt_acc = Int[]
    id_col = nothing
    time_col = nothing
    obs_col = nothing
    for s in 1:marginal_draws
        srng = sample_rngs[s]
        level_vals = Dict{Symbol, Dict{Any, Any}}()
        for re in re_names
            m = Dict{Any, Any}()
            for lvl in re_meta[re].levels_free
                m[lvl] = rand(srng, re_meta[re].dist)
            end
            level_vals[re] = m
        end
        get_free_value = (re, lvl, dim) -> level_vals[re][lvl]
        η_vec = Vector{ComponentArray}(undef, n_new)
        for j in 1:n_new
            η_vec[j] = _assemble_individual_eta(
                get_individuals(dm_new)[j], re_names, level_dims, fixed_maps,
                get_free_value
            )
        end
        cache = _fill_plot_cache(
            dm_new, θ, η_vec, constants_re, true,
            ode_args, ode_kwargs
        )
        df = get_residuals(
            dm_new; cache = cache, params = NamedTuple(θ),
            residuals = [:raw], fitted_stat = fitted_stat, constants_re = constants_re,
            ode_args = ode_args, ode_kwargs = ode_kwargs, kwargs...
        )
        if isempty(sum_acc)
            sum_acc = Any[nothing for _ in 1:nrow(df)]
            cnt_acc = zeros(Int, nrow(df))
            id_col, time_col, obs_col = df.id, df.time, df.observable
        end
        for r in 1:length(sum_acc)
            fr = df.fitted[r]
            ismissing(fr) && continue
            sum_acc[r] = cnt_acc[r] == 0 ? fr : sum_acc[r] .+ fr
            cnt_acc[r] += 1
        end
    end
    prediction = [
        cnt_acc[r] > 0 ? sum_acc[r] ./ cnt_acc[r] : missing
            for r in 1:length(sum_acc)
    ]
    return _prediction_frame(id_col, time_col, obs_col, identity.(prediction))
end
