using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random
using OptimizationOptimJL
using Statistics
using ForwardDiff

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

@testset "VEM: planar-flow variational family" begin
    @test_throws ErrorException PlanarFlowFamily(0)
    @test_throws ErrorException PlanarFlowFamily(2; base = :full)
    @test NoLimits.VEM(; family = PlanarFlowFamily(2)).flow == 2
    # identity at the init (û = 0) and exact log-Jacobian of the layer stack
    n, K = 3, 2
    rng = Xoshiro(4)
    θ = zeros(K * (2n + 1))
    for l in 1:K
        w = 0.1 .* randn(rng, n)
        o = (l - 1) * (2n + 1)
        θ[(o + 1):(o + n)] .= w .* (log(ℯ - 1) / sum(abs2, w))
        θ[(o + n + 1):(o + 2n)] .= w
    end
    ξ = randn(rng, n)
    z, ld = NoLimits._vem_flow(θ, 0, ξ, n, K)
    @test z ≈ ξ atol = 1.0e-12
    @test abs(ld) < 1.0e-12
    θr = randn(Xoshiro(5), K * (2n + 1))
    _, ldr = NoLimits._vem_flow(θr, 0, ξ, n, K)
    J = ForwardDiff.jacobian(x -> NoLimits._vem_flow(θr, 0, x, n, K)[1], ξ)
    @test ldr ≈ logabsdet(J)[1] rtol = 1.0e-10
    # Gaussian posterior: the flow family contains the exact q, so the fit matches Laplace
    dm = _vem_lin_dm(Xoshiro(11))
    rl = fit_model(dm, NoLimits.Laplace())
    fam = NoLimits.VEM(; family = PlanarFlowFamily(1))
    rf = fit_model(dm, fam; rng = Xoshiro(3))
    θl = get_params(rl; scale = :untransformed)
    θf = get_params(rf; scale = :untransformed)
    @test isapprox(θf.a, θl.a; atol = 0.03)
    @test isapprox(θf.b, θl.b; atol = 0.03)
    @test isapprox(θf.σ, θl.σ; rtol = 0.03)
    @test isapprox(θf.Ω, θl.Ω; atol = 0.04)
    @test abs(get_objective(rf) - get_objective(rl)) < 1.0
    @test all(q -> all(isfinite, q.mean) && all(isfinite, q.scale), get_notes(rf).variational_params)
    rf2 = fit_model(dm, fam; rng = Xoshiro(3))
    @test collect(get_params(rf2; scale = :transformed)) == collect(get_params(rf; scale = :transformed))
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
    @testset "VEM: Enzyme reverse mode with a planar-flow family" begin
        dm = _vem_lin_dm(Xoshiro(11))
        ad = NoLimits.Optimization.AutoEnzyme(;
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated
        )
        fam = PlanarFlowFamily(1)
        rf = fit_model(dm, NoLimits.VEM(; family = fam, n_is = 0); rng = Xoshiro(3))
        re = fit_model(dm, NoLimits.VEM(; family = fam, n_is = 0, adtype = ad); rng = Xoshiro(3))
        @test collect(get_params(re; scale = :transformed)) ≈
            collect(get_params(rf; scale = :transformed)) rtol = 1.0e-4
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

@testset "VEM: threaded fit equals serial bitwise" begin
    dm = _vem_lin_dm(Xoshiro(11))
    for m in (NoLimits.VEM(), NoLimits.VEM(; expectation = :mc, n_samples = 8, n_is = 50, optim_kwargs = (; maxiters = 10)))
        rt = fit_model(dm, m; rng = Xoshiro(3))
        rs = fit_model(dm, m; rng = Xoshiro(3), serialization = NoLimits.EnsembleSerial())
        @test collect(get_params(rt; scale = :transformed)) == collect(get_params(rs; scale = :transformed))
        @test get_objective(rt) === get_objective(rs)
        @test get_result(rt).notes.elbo === get_result(rs).notes.elbo
        @test get_result(rt).eb_modes == get_result(rs).eb_modes
    end
end

@testset "VEM: Wald UQ, conditional draws from q, CV with constants_re" begin
    dm = _vem_lin_dm(Xoshiro(11))
    rv = fit_model(dm, NoLimits.VEM(); rng = Xoshiro(3))
    @test uq_family(get_method(rv)) === :wald_re
    uq = compute_uq(rv; method = :wald)
    iv = get_uq_intervals(uq)
    θv = get_params(rv; scale = :untransformed)
    @test iv.lower.a < θv.a < iv.upper.a
    @test 0 < iv.upper.a - iv.lower.a < 1

    # draws come from the fitted variational q (b = mean + scale ξ), not a Laplace Gaussian
    bstars, bis, θu, cc, llc, _ = NoLimits._resolve_bstars_for_re(dm, rv, NamedTuple())
    draws = NoLimits._sample_conditional_bstars(dm, bis, bstars, θu, cc, llc, rv, 4000, Xoshiro(1))
    q = get_result(rv).notes.variational_params[1]
    B = reduce(hcat, [d[1] for d in draws])
    @test vec(mean(B; dims = 2)) ≈ q.mean atol = 0.01
    @test cov(B; dims = 2) ≈ q.scale * q.scale' rtol = 0.1
    sre = sample_random_effects(rv; n_samples = 3, rng = Xoshiro(1))
    @test nrow(sre.η) == 3 * 60

    @test NoLimits._cv_method_accepts_constants_re(NoLimits.VEM())
    cv = cross_validate(dm, 2; kind = :observation, rng = Xoshiro(4))
    cres = fit_cv(
        cv, NoLimits.VEM(; n_is = 0, refine = false, optim_kwargs = (; maxiters = 20));
        constants_re = (; η = Dict(1 => [0.0, 0.0])), seen_re_mode = :conditional,
        n_mc_samples = 10, store_results = true, rng = Xoshiro(4)
    )
    @test isfinite(cres.mean_test_loglikelihood)
    for fr in cres.fold_results
        @test haskey(NoLimits.get_fit_kwargs(fr.fit_result).constants_re, :η)
    end
end

@testset "VEM: model errors propagate, discrete random effects are rejected" begin
    model = @Model begin
        @helpers begin
            vem_guard(x) = x > 1.5 ? error("vem_guard: user error") : x
        end
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5; scale = :log)
            ω = RealNumber(0.5; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(vem_guard(a) + η, σ)
        end
    end
    df = DataFrame(ID = repeat(1:8, inner = 3), t = repeat(0.0:2.0, 8), y = 3.0 .+ 0.1 .* randn(Xoshiro(1), 24))
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    # used to be scored -Inf, so the fit silently stalled at a = 1.5
    @test_throws "vem_guard: user error" fit_model(dm, NoLimits.VEM(; n_is = 0); rng = Xoshiro(1))

    dmodel = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Poisson(2.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    ddm = DataModel(dmodel, df; primary_id = :ID, time_col = :t)
    @test_throws "VEM does not support discrete" fit_model(ddm, NoLimits.VEM())
end

# Smoke coverage on shared fixtures (few iterations; not accuracy checks).
@testset "VEM: ODE, Poisson outcome, two RE groupings, Multistart" begin
    m = NoLimits.VEM(; n_is = 50, optim_kwargs = (; maxiters = 5))
    for dm in (fx_ode_dm(), fx_pois_dm(), fx_mg_dm())
        r = fit_model(dm, m; rng = Xoshiro(1))
        @test isfinite(get_objective(r))
        @test all(isfinite, collect(get_params(r; scale = :untransformed)))
        @test length(get_result(r).eb_modes) == length(get_result(r).notes.variational_params)
    end
    ms = NoLimits.Multistart(; dists = (; a = Normal(0.2, 0.3)), n_draws_requested = 3, n_draws_used = 2)
    rm = fit_model(ms, fx_pois_dm(), m; rng = Xoshiro(2))
    @test rm isa NoLimits.MultistartFitResult
    @test length(NoLimits.get_multistart_results(rm)) == 2
    @test isfinite(get_objective(rm))
end
