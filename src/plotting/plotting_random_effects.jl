export plot_random_effect_distributions
export plot_random_effect_pit
export plot_random_effect_standardized
export plot_random_effect_standardized_scatter
export plot_random_effects_pdf
export plot_random_effect_pairplot
export plot_random_effects_scatter

using Distributions
using KernelDensity
using Random
using Statistics

function _require_re_supported(res::FitResult)
    if res.result isa MLEResult || res.result isa MAPResult
        @warn "Random-effects diagnostics are not available for MLE/MAP."
        error("Random-effects diagnostics require Laplace/MCEM/SAEM/MCMC.")
    end
end

function _resolve_re_names(dm::DataModel, re_names)
    names = get_re_names(dm.model.random.random)
    isempty(names) && error("Model has no random effects.")
    re_list = re_names === nothing ? names :
              (re_names isa AbstractVector ? collect(re_names) : [re_names])
    for r in re_list
        r in names || error("Random effect $(r) not found. Available: $(names).")
    end
    return re_list
end

function _fit_constants_re(res::FitResult)
    if hasproperty(res, :fit_kwargs)
        kw = res.fit_kwargs
        return haskey(kw, :constants_re) ? getfield(kw, :constants_re) : NamedTuple()
    end
    return NamedTuple()
end

function _filter_re_without_covariates(res::FitResult, re_list)
    usage = get_re_covariate_usage(res)
    out = Symbol[]
    for r in re_list
        used = hasproperty(usage, r) ? getfield(usage, r) : Symbol[]
        isempty(used) && push!(out, r)
    end
    return out
end

function _resolve_levels(dm::DataModel, re::Symbol, levels, individuals_idx)
    levels_all = getfield(dm.re_group_info.values, re)
    if individuals_idx !== nothing
        inds = _resolve_individuals(dm, individuals_idx)
        re_groups = get_re_groups(dm.model.random.random)
        col = getfield(re_groups, re)
        selected = Set{Any}()
        for i in inds
            g = getfield(dm.individuals[i].re_groups, re)
            if g isa AbstractVector
                for lvl in g
                    push!(selected, lvl)
                end
            else
                push!(selected, g)
            end
        end
        levels_all = [lvl for lvl in levels_all if lvl in selected]
    end
    if levels === nothing
        return levels_all
    end
    lv = levels isa AbstractVector ? collect(levels) : [levels]
    for v in lv
        v in levels_all || error("Level $(v) not found for random effect $(re).")
    end
    return lv
end

function _level_to_individual(dm::DataModel, re::Symbol)
    map = Dict{Any, Int}()
    for (i, ind) in enumerate(dm.individuals)
        g = getfield(ind.re_groups, re)
        if g isa AbstractVector
            for lvl in g
                haskey(map, lvl) || (map[lvl] = i)
            end
        else
            haskey(map, g) || (map[g] = i)
        end
    end
    return map
end

function _ebe_by_level(dm::DataModel, res::FitResult, re::Symbol)
    # Skip fixed levels from constants_re to avoid empty bstars for those levels.
    constants_re = _fit_constants_re(res)
    re_df = get_random_effects(
        dm, res; constants_re = constants_re, flatten = true, include_constants = false)
    df = getproperty(re_df, re)
    re_groups = get_re_groups(dm.model.random.random)
    col = Symbol(getfield(re_groups, re))
    levels = df[!, col]
    value_cols = [c for c in names(df) if Symbol(c) != col]
    out = Dict{Any, Vector{Float64}}()
    for (i, lvl) in enumerate(levels)
        out[lvl] = Float64.(collect(df[i, value_cols]))
    end
    return out, value_cols
end

function _level_values_from_eta(dm::DataModel, re::Symbol, η_vec::Vector{ComponentArray})
    out = Dict{Any, Vector{Float64}}()
    for (i, ind) in enumerate(dm.individuals)
        g = getfield(ind.re_groups, re)
        v = getproperty(η_vec[i], re)
        if g isa AbstractVector
            if length(g) == 1
                lvl = g[1]
                out[lvl] = v isa Number ? [Float64(v)] : Float64.(collect(vec(v)))
            else
                for (gi, lvl) in pairs(g)
                    val = v isa Number ? Float64(v) : v[gi]
                    out[lvl] = val isa Number ? [Float64(val)] : Float64.(collect(vec(val)))
                end
            end
        else
            out[g] = v isa Number ? [Float64(v)] : Float64.(collect(vec(v)))
        end
    end
    return out
end

function _ebe_by_level_mcmc(
        dm::DataModel, res::FitResult, re::Symbol, mcmc_draws::Int, rng::AbstractRNG)
    constants_re = _fit_constants_re(res)
    θ_draws, η_draws, _ = _posterior_drawn_params(
        res, dm, constants_re, NamedTuple(), mcmc_draws, rng)
    sums = Dict{Any, Vector{Float64}}()
    counts = Dict{Any, Int}()
    for d in eachindex(η_draws)
        levels = _level_values_from_eta(dm, re, η_draws[d])
        for (lvl, v) in levels
            if !haskey(sums, lvl)
                sums[lvl] = zeros(Float64, length(v))
                counts[lvl] = 0
            end
            sums[lvl] .+= v
            counts[lvl] += 1
        end
    end
    out = Dict{Any, Vector{Float64}}()
    for (lvl, s) in sums
        out[lvl] = s ./ counts[lvl]
    end
    dim = isempty(out) ? 1 : length(first(values(out)))
    value_cols = flatten_re_names(re, zeros(dim))
    return out, value_cols
end

function _standardize_re(dist, val::Vector{Float64}; flow_samples::Int = 500)
    if dist isa NormalizingPlanarFlow
        n = flow_samples
        if dist isa Distributions.UnivariateDistribution
            samp = vec(rand(dist, n))
            μ = mean(samp)
            σ = std(samp)
            σ == 0 && return nothing
            return [(val[1] - μ) / σ]
        else
            samp = rand(dist, n)
            μv = vec(mean(samp; dims = 2))
            Σm = cov(permutedims(samp))
            L = try
                cholesky(Σm).L
            catch
                return nothing
            end
            return L \ (val .- μv)
        end
    end
    if dist isa Distributions.MvLogNormal
        z = log.(max.(val, 1e-300))
        μ = try
            Float64.(Distributions.mean(dist.normal))
        catch
            return nothing
        end
        Σ = try
            Matrix{Float64}(cov(dist.normal))
        catch
            return nothing
        end
        L = try
            cholesky(Σ).L
        catch
            return nothing
        end
        return L \ (z .- μ)
    end
    if dist isa Distributions.MvLogitNormal
        # ALR: z_i = log(η_i / η_{d+1}), inner d-dim coords
        ηf = max.(Float64.(val), 1e-300)
        ref = ηf[end]
        z = log.(ηf[begin:(end - 1)]) .- log(ref)
        μ = try
            Float64.(Distributions.mean(dist.normal))
        catch
            return nothing
        end
        Σ = try
            Matrix{Float64}(cov(dist.normal))
        catch
            return nothing
        end
        L = try
            cholesky(Σ).L
        catch
            return nothing
        end
        return L \ (z .- μ)
    end
    if dist isa Distributions.MultivariateDistribution && !(dist isa MvNormal)
        @warn "Skipping multivariate RE standardization for non-Normal distribution." dist=typeof(dist)
        return nothing
    end
    μ = try
        Distributions.mean(dist)
    catch
        return nothing
    end
    Σ = try
        dist isa Distributions.MultivariateDistribution ? cov(dist) : var(dist)
    catch
        return nothing
    end
    μv = μ isa Number ? [Float64(μ)] : Float64.(collect(vec(μ)))

    Σm = Σ isa Number ? [Float64(Σ)] : Matrix{Float64}(reshape(collect(Σ), size(Σ)...))
    if length(μv) == 1
        σ = sqrt(Σm[1])
        σ == 0 && return nothing
        return [(val[1] - μv[1]) / σ]
    end
    L = try
        cholesky(Σm).L
    catch
        return nothing
    end
    return L \ (val .- μv)
end

function _marginal_normal(dist, i::Int)
    if dist isa MvNormal
        μ = try
            Distributions.mean(dist)
        catch
            return nothing
        end
        Σ = try
            cov(dist)
        catch
            return nothing
        end
        μv = Float64.(collect(vec(μ)))
        Σm = Matrix{Float64}(Σ)
        i <= length(μv) || return nothing
        return Normal(μv[i], sqrt(Σm[i, i]))
    elseif dist isa Distributions.MvLogNormal
        μ = try
            Distributions.mean(dist.normal)
        catch
            return nothing
        end
        Σ = try
            cov(dist.normal)
        catch
            return nothing
        end
        μv = Float64.(collect(vec(μ)))
        Σm = Matrix{Float64}(Σ)
        i <= length(μv) || return nothing
        return LogNormal(μv[i], sqrt(Σm[i, i]))
    elseif dist isa Distributions.MvLogitNormal
        # Inner (ALR) marginal for i-th ALR coordinate (only valid for i ≤ inner dim d)
        i <= length(dist.normal) || return nothing
        μ = try
            Distributions.mean(dist.normal)
        catch
            return nothing
        end
        Σ = try
            cov(dist.normal)
        catch
            return nothing
        end
        μv = Float64.(collect(vec(μ)))
        Σm = Matrix{Float64}(Σ)
        return Normal(μv[i], sqrt(Σm[i, i]))
    end
    return nothing
end

function _kde_xy(vals; bandwidth = nothing)
    kd = bandwidth === nothing ? kde(vals) : kde(vals; bandwidth = bandwidth)
    return kd.x, kd.density
end

function _interp_linear(x_src::AbstractVector, y_src::AbstractVector, x_tgt::AbstractVector)
    n = length(x_src)
    n == length(y_src) || error("Interpolation source x/y lengths differ.")
    n < 2 && return fill(y_src[1], length(x_tgt))
    y_tgt = similar(x_tgt, Float64)
    j = 1
    x1 = x_src[1]
    x2 = x_src[2]
    y1 = y_src[1]
    y2 = y_src[2]
    for (i, x) in enumerate(x_tgt)
        while x > x2 && j < n - 1
            j += 1
            x1 = x2
            y1 = y2
            x2 = x_src[j + 1]
            y2 = y_src[j + 1]
        end
        if x <= x1
            y_tgt[i] = y1
        elseif x >= x2
            y_tgt[i] = y2
        else
            t = (x - x1) / (x2 - x1)
            y_tgt[i] = y1 + t * (y2 - y1)
        end
    end
    return y_tgt
end

function _pit_value(dist, val::Float64)
    applicable(cdf, dist, val) || return nothing
    return cdf(dist, val)
end
