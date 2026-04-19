using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# AdaptiveNoLimitsMH — construction
# ---------------------------------------------------------------------------

@testset "AdaptiveNoLimitsMH constructor" begin
    @test AdaptiveNoLimitsMH() isa AdaptiveNoLimitsMH
    s = AdaptiveNoLimitsMH(adapt_start=10, init_scale=2.0, eps_reg=1e-4)
    @test s.adapt_start == 10
    @test s.init_scale  == 2.0
    @test s.eps_reg     == 1e-4
    @test_throws ArgumentError AdaptiveNoLimitsMH(adapt_start=-1)
    @test_throws ArgumentError AdaptiveNoLimitsMH(init_scale=0.0)
    @test_throws ArgumentError AdaptiveNoLimitsMH(eps_reg=0.0)
end

# ---------------------------------------------------------------------------
# Normal RE — basic recovery
# ---------------------------------------------------------------------------

@testset "AdaptiveNoLimitsMH Normal RE recovery" begin
    rng  = MersenneTwister(42)
    n_id = 20
    true_a  = 1.0
    true_σ  = 0.3
    true_ση = 0.5

    ids = repeat(1:n_id, inner=3)
    ts  = repeat([0.0, 0.5, 1.0], n_id)
    ηs  = true_ση .* randn(rng, n_id)
    ys  = true_a .+ ηs[ids] .+ true_σ .* randn(rng, length(ids))
    df  = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a   = RealNumber(0.5)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = AdaptiveNoLimitsMH(adapt_start=5),
        maxiters=2,
        mcmc_steps = 20,
        progress   = false,
    ))

    params = NoLimits.get_params(res; scale=:untransformed)
    @test abs(params.a  - true_a)  < 1.0
    @test 0.05 < params.σ   < 2.0
    @test 0.05 < params.σ_η < 2.0
end

# ---------------------------------------------------------------------------
# Adaptation state persists across iterations (warm_start=true)
# ---------------------------------------------------------------------------

@testset "AdaptiveNoLimitsMH warm-start state persistence" begin
    rng  = MersenneTwister(7)
    n_id = 10
    ids  = repeat(1:n_id, inner=2)
    ts   = repeat([0.0, 1.0], n_id)
    ys   = 1.0 .+ randn(rng, length(ids)) .* 0.5
    df   = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a   = RealNumber(0.0)
            σ   = RealNumber(0.5, scale=:log)
            σ_η = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = AdaptiveNoLimitsMH(adapt_start=2),
        maxiters=2,
        mcmc_steps = 5,
        progress   = false,
        warm_start = true,
    ))
    @test NoLimits.get_iterations(res) == 15
end

# ---------------------------------------------------------------------------
# MvNormal RE (d=2) — block covariance adaptation
# ---------------------------------------------------------------------------

@testset "AdaptiveNoLimitsMH MvNormal RE (d=2)" begin
    rng    = MersenneTwister(13)
    n_id   = 15
    Ω_true = [1.0 0.4; 0.4 1.0]
    ids    = repeat(1:n_id, inner=3)
    ts     = repeat([0.0, 0.5, 1.0], n_id)
    ηs     = (cholesky(Ω_true).L * randn(rng, 2, n_id))'
    ys     = 1.0 .+ ηs[ids, 1] .+ 0.4 .* randn(rng, length(ids))
    df     = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(0.5, scale=:log)
            Ω = RealPSDMatrix([1.0 0.0; 0.0 1.0]; scale=:cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = AdaptiveNoLimitsMH(adapt_start=5),
        maxiters=2,
        mcmc_steps = 10,
        progress   = false,
    ))
    params = NoLimits.get_params(res; scale=:untransformed)
    @test abs(params.a - 1.0) < 1.5
end

# ---------------------------------------------------------------------------
# LogNormal RE — log-space bijection
# ---------------------------------------------------------------------------

@testset "AdaptiveNoLimitsMH LogNormal RE" begin
    rng  = MersenneTwister(99)
    n_id = 15
    ids  = repeat(1:n_id, inner=3)
    ts   = repeat([0.0, 0.5, 1.0], n_id)
    ηs   = exp.(0.3 .* randn(rng, n_id))
    ys   = ηs[ids] .+ 0.2 .* randn(rng, length(ids))
    df   = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            σ   = RealNumber(0.3, scale=:log)
            σ_η = RealNumber(0.3, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, σ_η); column=:ID)
        end
        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = AdaptiveNoLimitsMH(adapt_start=5),
        maxiters=2,
        mcmc_steps = 10,
        progress   = false,
    ))
    params = NoLimits.get_params(res; scale=:untransformed)
    @test 0.0 < params.σ   < 2.0
    @test 0.0 < params.σ_η < 2.0
end
