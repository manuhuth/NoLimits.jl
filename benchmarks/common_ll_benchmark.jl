"""
Baseline + post-fix benchmark for _loglikelihood_individual in common.jl.

Run before and after the performance fixes to measure improvement.
"""

using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using BenchmarkTools
using SciMLBase

import NoLimits: loglikelihood, _loglikelihood_individual, build_ll_cache

# ─── Setup: 10 individuals, 20 observations each, Normal obs ─────────────────

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.5)
        b = RealNumber(0.2)
        σ = RealNumber(0.4, scale=:log)
    end
    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age]; constant_on=:ID)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end
    @formulas begin
        μ = a + b * x.Age + η
        y ~ Normal(μ, σ)
    end
end

n_ind = 10
n_obs = 20
rng = MersenneTwister(42)

df = DataFrame(
    ID    = repeat(1:n_ind, inner=n_obs),
    t     = repeat(range(0.0, 10.0; length=n_obs), n_ind),
    Age   = repeat(rand(rng, n_ind) .* 20 .+ 30, inner=n_obs),
    y     = randn(rng, n_ind * n_obs)
)

dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
θ   = get_θ0_untransformed(model.fixed.fixed)
η_v = [ComponentArray(η = randn(rng)) for _ in 1:n_ind]

# ─── Build cache ─────────────────────────────────────────────────────────────

cache = build_ll_cache(dm)

# ─── Warm-up (trigger compilation) ──────────────────────────────────────────

loglikelihood(dm, θ, η_v; cache=cache)
ForwardDiff.gradient(v -> loglikelihood(dm, ComponentArray(v, getaxes(θ)), η_v; cache=cache), Vector(θ))

# ─── Benchmark 1: Float64 loglikelihood (full sum, cached) ──────────────────

println("\n=== Float64: full loglikelihood (cached, serial) ===")
b1 = @benchmark loglikelihood($dm, $θ, $η_v; cache=$cache)
display(b1)

# ─── Benchmark 2: ForwardDiff gradient of loglikelihood ─────────────────────

println("\n=== ForwardDiff gradient of loglikelihood ===")
θ_vec = Vector(θ)
b2 = @benchmark ForwardDiff.gradient(
    v -> loglikelihood($dm, ComponentArray(v, getaxes($θ)), $η_v; cache=$cache),
    $θ_vec
)
display(b2)

# ─── Benchmark 3: single individual, Float64 ─────────────────────────────────

println("\n=== Single individual, Float64 ===")
b3 = @benchmark _loglikelihood_individual($dm, 1, $θ, $(η_v[1]), $cache)
display(b3)

# ─── Benchmark 4: single individual, ForwardDiff Dual ────────────────────────

println("\n=== Single individual, ForwardDiff (gradient wrt θ) ===")
cfg = ForwardDiff.GradientConfig(nothing, θ_vec, ForwardDiff.Chunk(length(θ_vec)))
θ_dual = ForwardDiff.seed!(copy(ForwardDiff.Dual{Nothing, eltype(θ_vec), length(θ_vec)}.(θ_vec, Ref(ForwardDiff.Partials{length(θ_vec), eltype(θ_vec)}(NTuple{length(θ_vec), eltype(θ_vec)}(i == j ? 1.0 : 0.0 for i in 1:length(θ_vec)) for j in 1:length(θ_vec))[1])), θ_vec, cfg)
θ_ca_dual = ComponentArray(θ_dual, getaxes(θ))
b4 = @benchmark _loglikelihood_individual($dm, 1, $θ_ca_dual, $(η_v[1]), $cache)
display(b4)

println("\nDone. Record these numbers before applying performance fixes.")
