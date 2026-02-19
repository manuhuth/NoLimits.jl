#using NoLimits
using DataFrames
using Random
using Distributions
using ProgressMeter
using Statistics

function _make_panel(n_ids::Int, n_obs::Int; rng::AbstractRNG)
    ID = repeat(1:n_ids, inner=n_obs)
    tvals = n_obs == 1 ? [0.0] : collect(range(0.0, 1.0; length=n_obs))
    t = repeat(tvals, n_ids)
    z1 = randn(rng, length(t))
    z2 = randn(rng, length(t))
    return DataFrame(ID=ID, t=t, z1=z1, z2=z2, y=0.0)
end

function _flatten_param(x)
    x isa Number && return [Float64(x)]
    x isa AbstractArray && return vec(Float64.(x))
    return Float64[]
end

function _error_stats(θ_hat, θ_true; rel_tol=0.1, abs_tol=0.1)
    names = propertynames(θ_true)
    ok = Bool[]
    abs_errs = Float64[]
    rel_errs = Float64[]
    for n in names
        vh = _flatten_param(getproperty(θ_hat, n))
        vt = _flatten_param(getproperty(θ_true, n))
        for i in eachindex(vt)
            ae = abs(vh[i] - vt[i])
            re = abs(vh[i] - vt[i]) / max(abs(vt[i]), 1e-8)
            push!(abs_errs, ae)
            push!(rel_errs, re)
            push!(ok, (ae <= abs_tol) || (re <= rel_tol))
        end
    end
    return (mean_ok=mean(ok), mean_abs=mean(abs_errs), mean_rel=mean(rel_errs))
end

function _jitter_params(fe, rng::AbstractRNG; scale=0.3)
    θt = get_θ0_transformed(fe)
    θt_j = deepcopy(θt)
    for n in propertynames(θt_j)
        v = getproperty(θt_j, n)
        if v isa Number
            setproperty!(θt_j, n, v + scale * randn(rng))
        elseif v isa AbstractArray
            setproperty!(θt_j, n, v .+ scale .* randn(rng, size(v)))
        end
    end
    inv = get_inverse_transform(fe)
    return inv(θt_j)
end

function _rep_rngs(seed::Integer, n::Int; offset::Integer=0)
    return [MersenneTwister(seed + offset + i) for i in 1:n]
end

function _build_models()
    model1 = @Model begin
        @fixedEffects begin
            a  = RealNumber(50.0, prior=Normal(50.0, 10.0))
            k  = RealNumber(3.0, prior=Normal(3.0, 1.0))
            t0 = RealNumber(0.5, prior=Normal(0.5, 0.2))
            b1 = RealNumber(2.0, prior=Normal(2.0, 0.5))
            σ  = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            z1 = Covariate()
        end

        @formulas begin
            μ = a / (1 + exp(-k * (t - t0))) + b1 * z1
            y ~ Normal(μ, σ)
        end
    end

    model2 = @Model begin
        @fixedEffects begin
            a  = RealNumber(0.2, prior=Normal(0.2, 0.2))
            b1 = RealNumber(0.5, prior=Normal(0.5, 0.2))
            b2 = RealNumber(-0.3, prior=Normal(-0.3, 0.2))
            σ  = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            z1 = Covariate()
        end

        @formulas begin
            μ = exp(a + b1 * z1 + b2 * z1^2)
            y ~ Normal(μ, σ)
        end
    end

    model3 = @Model begin
        @fixedEffects begin
            vmax = RealNumber(10.0, prior=Normal(10.0, 3.0))
            km   = RealNumber(2.0, prior=Normal(2.0, 1.0))
            b1   = RealNumber(1.0, prior=Normal(1.0, 0.5))
            b2   = RealNumber(-0.5, prior=Normal(-0.5, 0.5))
            σ    = RealNumber(0.7, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            z1 = Covariate()
            z2 = Covariate()
        end

        @formulas begin
            μ = vmax * z1 / (km + abs(z1)) + b1 * z2 + b2 * z2^2
            y ~ Normal(μ, σ)
        end
    end

    return (model1, model2, model3)
end
