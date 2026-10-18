using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random

# VEM (variational EM, Tarek & Afonso 2026). Linear-Gaussian anchor: the dense Gaussian
# family is exact, so VEM must reproduce the (exact) Laplace MLE up to the sample-average
# error; a bounded-support (LogNormal) RE exercises the transport-space family.
function _vem_lin_dm(rng)
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            b = RealNumber(0.0)
            σ = RealNumber(0.5; scale = :log)
            Ω = RealPSDMatrix([0.3 0.0; 0.0 0.3]; scale = :cholesky)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
        end
        @formulas begin
            y ~ Normal((a + η[1]) + (b + η[2]) * t, σ)
        end
    end
    L = cholesky([0.4 0.15; 0.15 0.25]).L
    rows = NamedTuple[]
    for i in 1:60
        η = L * randn(rng, 2)
        for t in 0.0:1.0:5.0
            push!(rows, (; ID = i, t, y = 2.0 + η[1] + (-0.5 + η[2]) * t + 0.3 * randn(rng)))
        end
    end
    return DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
end

@testset "VEM: dense family reproduces the exact MLE (linear Gaussian)" begin
    dm = _vem_lin_dm(Xoshiro(11))
    rl = fit_model(dm, NoLimits.Laplace())
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    θl = get_params(rl; scale = :untransformed)
    θv = get_params(rv; scale = :untransformed)
    @test isapprox(θv.a, θl.a; atol = 0.02)
    @test isapprox(θv.b, θl.b; atol = 0.02)
    @test isapprox(θv.σ, θl.σ; rtol = 0.02)
    @test isapprox(θv.Ω, θl.Ω; atol = 0.03)
    # IS objective close to the exact marginal likelihood
    @test abs(get_objective(rv) - get_objective(rl)) < 1.0
    @test get_converged(rv)
    @test get_result(rv).notes.elbo isa Real
    # random effects for prediction
    p = NoLimits.predict(rv, get_df(dm); re_mode = :ebe)
    @test all(isfinite, p.prediction)
end

@testset "VEM: options and validation" begin
    @test_throws ErrorException NoLimits.VEM(; family = :full)
    @test_throws ErrorException NoLimits.VEM(; n_samples = 0)
    @test_throws ErrorException NoLimits.VEM(; init = :zero)
    dm = _vem_lin_dm(Xoshiro(12))
    rd = fit_model(dm, NoLimits.VEM(; family = :diagonal, n_samples = 8, n_is = 0); rng = Xoshiro(1))
    @test isfinite(get_objective(rd))
    @test get_result(rd).notes.objective_kind === :neg_elbo
    rp = fit_model(dm, NoLimits.VEM(; init = :prior, n_samples = 8); rng = Xoshiro(1))
    @test isfinite(get_objective(rp))
end

@testset "VEM: LogNormal random effect (transport-space family)" begin
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(1.0; scale = :log)
            ω = RealNumber(0.3; scale = :log)
            σ = RealNumber(0.3; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(10 * exp(-k * η * t), σ)
        end
    end
    rng = Xoshiro(5)
    rows = NamedTuple[]
    for i in 1:40
        η = exp(0.4 * randn(rng))
        for t in (0.25, 0.5, 1.0, 2.0)
            push!(rows, (; ID = i, t, y = 10 * exp(-0.7 * η * t) + 0.2 * randn(rng)))
        end
    end
    dm = DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
    rl = fit_model(dm, NoLimits.Laplace())
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(2))
    θl = get_params(rl; scale = :untransformed)
    θv = get_params(rv; scale = :untransformed)
    @test isapprox(θv.k, θl.k; rtol = 0.05)
    @test isapprox(θv.ω, θl.ω; rtol = 0.15)
    @test isapprox(θv.σ, θl.σ; rtol = 0.05)
end
