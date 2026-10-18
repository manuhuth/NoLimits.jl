using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random
using OptimizationOptimJL

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
    # default Optimization.jl LBFGS reproduces the pre-interface Optim.optimize fit
    @test collect(get_params(rv; scale = :transformed)) ≈ [
        1.9644147957675464, -0.5044155641405206, -1.350884846083775,
        -0.4893695877661404, 0.25538065373340313, -0.8110340978938972,
    ] rtol = 1.0e-6
    @test get_iterations(rv) == 25
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

@testset "VEM: Optimization.jl optimizer, optim_kwargs and bounds" begin
    dm = _vem_lin_dm(Xoshiro(11))
    θd = collect(get_params(fit_model(dm, NoLimits.VEM(; refine = false, n_is = 0); rng = Xoshiro(3)); scale = :transformed))
    rb = fit_model(
        dm, NoLimits.VEM(; optimizer = OptimizationOptimJL.BFGS(), refine = false, n_is = 0);
        rng = Xoshiro(3)
    )
    @test get_method(rb).optimizer isa OptimizationOptimJL.BFGS
    @test collect(get_params(rb; scale = :transformed)) ≈ θd atol = 1.0e-3
    # optim_kwargs replace the defaults and reach both stages
    rm = fit_model(dm, NoLimits.VEM(; optim_kwargs = (; maxiters = 3)); rng = Xoshiro(3))
    @test get_iterations(rm) <= 3
    @test get_result(rm).notes.refine_iterations <= 3
    @test !get_converged(rm)
    # transformed-scale upper bound on `a` (MLE ≈ 1.96) is active at the fit
    lo, hi = default_bounds_from_start(dm; margin = 5.0)
    hi.a = 1.5
    rub = fit_model(dm, NoLimits.VEM(; lb = lo, ub = hi, n_is = 0); rng = Xoshiro(3))
    @test get_params(rub; scale = :transformed).a <= 1.5 + 1.0e-8
    @test get_params(rub; scale = :transformed).a > 1.4
    # all randomness (MC draws, IS draws) comes from the fit `rng`
    mc = NoLimits.VEM(; expectation = :mc, n_samples = 8, refine = false, n_is = 50)
    r1 = fit_model(dm, mc; rng = Xoshiro(7))
    r2 = fit_model(dm, mc; rng = Xoshiro(7))
    r3 = fit_model(dm, mc; rng = Xoshiro(8))
    @test collect(get_params(r1; scale = :transformed)) == collect(get_params(r2; scale = :transformed))
    @test get_objective(r1) === get_objective(r2)
    @test collect(get_params(r3; scale = :transformed)) != collect(get_params(r1; scale = :transformed))
end

# Opt-in like enzyme_compat_proxy_tests.jl: Enzyme is not a test dependency.
if get(ENV, "NOLIMITS_TEST_ENZYME", "false") == "true" && VERSION >= v"1.12.5" &&
        Base.find_package("Enzyme") !== nothing
    using Enzyme
    @testset "VEM: Enzyme reverse mode matches ForwardDiff" begin
        dm = _vem_lin_dm(Xoshiro(11))
        ad = NoLimits.Optimization.AutoEnzyme(;
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated
        )
        rf = fit_model(dm, NoLimits.VEM(; n_is = 0); rng = Xoshiro(3))
        re = fit_model(dm, NoLimits.VEM(; n_is = 0, adtype = ad); rng = Xoshiro(3))
        @test get_method(re).adtype === ad
        @test collect(get_params(re; scale = :transformed)) ≈
            collect(get_params(rf; scale = :transformed)) rtol = 1.0e-6
    end
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
