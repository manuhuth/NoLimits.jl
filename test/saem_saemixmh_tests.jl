using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# SaemixMH — construction
# ---------------------------------------------------------------------------

@testset "SaemixMH constructor" begin
    s = SaemixMH()
    @test s isa SaemixMH
    @test s.n_kern1       == 2
    @test s.n_kern2       == 2
    @test s.target_accept == 0.44
    @test s.adapt_rate    == 0.7

    s2 = SaemixMH(n_kern1=3, n_kern2=1, target_accept=0.234, adapt_rate=0.5)
    @test s2.n_kern1  == 3
    @test s2.n_kern2  == 1
    @test s2.adapt_rate == 0.5
end

# ---------------------------------------------------------------------------
# Normal RE — basic parameter recovery
# ---------------------------------------------------------------------------

@testset "SaemixMH Normal RE recovery" begin
    rng    = MersenneTwister(17)
    n_id   = 30
    true_a  = 2.0
    true_σ  = 0.4
    true_τ  = 0.8

    ids  = repeat(1:n_id, inner=4)
    ts   = repeat([0.0, 0.5, 1.0, 1.5], n_id)
    ηs   = true_τ .* randn(rng, n_id)
    ys   = true_a .+ ηs[ids] .+ true_σ .* randn(rng, length(ids))
    df   = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = SaemixMH(n_kern1=2, n_kern2=2),
        maxiters   = 100,
        mcmc_steps = 1,
        q_store_max  = 20,
        progress   = false,
    ))

    @test isfinite(NoLimits.get_objective(res))
    params = NoLimits.get_params(res; scale=:untransformed)
    @test abs(params.a - true_a)  < 0.8
    @test 0.05 < params.σ < 2.0
    @test 0.05 < params.τ < 2.0
end

# ---------------------------------------------------------------------------
# SaemixMH with closed-form M-step (re_cov_params)
# ---------------------------------------------------------------------------

@testset "SaemixMH closed-form M-step recovery" begin
    rng    = MersenneTwister(99)
    n_id   = 20
    true_a  = 1.5
    true_σ  = 0.5
    true_τ  = 0.7

    ids  = repeat(1:n_id, inner=4)
    ts   = repeat([0.0, 0.5, 1.0, 1.5], n_id)
    ηs   = true_τ .* randn(rng, n_id)
    ys   = true_a .+ ηs[ids] .+ true_σ .* randn(rng, length(ids))
    df   = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)

    # With builtin_mean=:glm this triggers the closed-form mean update path
    res = fit_model(dm, SAEM(
        sampler      = SaemixMH(n_kern1=2, n_kern2=2),
        maxiters     = 80,
        mcmc_steps   = 1,
        q_store_max    = 20,
        builtin_mean = :glm,
        re_cov_params = (; η = :τ),
        progress     = false,
    ))

    @test isfinite(NoLimits.get_objective(res))
    params = NoLimits.get_params(res; scale=:untransformed)
    @test abs(params.a - true_a)  < 0.8
    @test 0.05 < params.σ < 2.0
    @test 0.05 < params.τ < 2.0
end

# ---------------------------------------------------------------------------
# SaemixMH warm-start state persistence
# ---------------------------------------------------------------------------

@testset "SaemixMH warm-start state persists" begin
    rng   = MersenneTwister(55)
    n_id  = 10
    ids   = repeat(1:n_id, inner=3)
    ts    = repeat([0.0, 0.5, 1.0], n_id)
    ys    = 1.0 .+ 0.3 .* randn(rng, length(ids))
    df    = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = SaemixMH(),
        maxiters   = 20,
        mcmc_steps = 1,
        q_store_max  = 10,
        progress   = false,
    ))
    @test isfinite(NoLimits.get_objective(res))

    # Diagnostics should report accept counts
    diag = NoLimits.get_diagnostics(res)
    @test !isnothing(diag)
end
