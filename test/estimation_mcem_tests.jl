using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using SciMLBase
using OptimizationOptimisers
using OptimizationBBO

# One scalar-RE model shared by the option/sampler/constants testsets below
# (they assert fit-option behavior, not model structure). Structure-specific
# testsets (multivariate, ODE, Poisson, covariate-RE, multi-group) use the
# shared fx_* fixtures from fixtures.jl.
const _MCEM_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.5, scale = :log)
    end

    @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0); column = :ID)
    end

    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

const _MCEM_DM2 = DataModel(_MCEM_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]);
    primary_id = :ID, time_col = :t)

const _MCEM_DM3 = DataModel(_MCEM_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0]);
    primary_id = :ID, time_col = :t)

const _MCEM_DM4 = DataModel(_MCEM_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]);
    primary_id = :ID, time_col = :t)

@testset "MCEM default sampler" begin
    method = NoLimits.MCEM()
    @test method.e_step isa NoLimits.MCEM_MCMC
    @test method.e_step.sampler isa NUTS
    @test method.ebe.multistart_n == 50
    @test method.ebe.multistart_k == 1
    @test method.ebe.sampling == :lhs
    @test method.ebe_rescue.sampling == :lhs
end

@testset "MCEM windowed drift test triggers early stop" begin
    # Inf tolerances make every post-window-fill check pass, so the stop point is
    # deterministic: window fill (4) + consecutive (2) - 1 = iteration 5.
    res = fit_model(_MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 30, convergence_window = 4, consecutive_params = 2,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf,
            progress = false))
    @test NoLimits.get_converged(res)
    @test 5 <= res.result.iterations < 30
    diag = res.result.notes.diagnostics
    @test isnan(diag.drift_θ[1])  # window not yet full
    @test isfinite(diag.drift_θ[end])
end

@testset "MCEM no early stop before drift window fills" begin
    res = fit_model(_MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 3, convergence_window = 4, consecutive_params = 1,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf,
            progress = false))
    @test !NoLimits.get_converged(res)
    @test res.result.iterations == 3
end

@testset "MCEM basic (random effects)" begin
    res = fit_model(_MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

# NOTE: testsets shared line-for-line with SAEM (serial-vs-threaded reproducibility,
# convergence stabilization, multiple RE groups, thread caches/RNGs, EBE rescue,
# constants_re) live as parameterized "SAEM/MCEM …" loops in estimation_saem_tests.jl.

@testset "MCEM basic with NUTS" begin
    res = fit_model(_MCEM_DM2,
        NoLimits.MCEM(; sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
end

@testset "MCEM constants_re" begin
    res = fit_model(_MCEM_DM3,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2);
        constants_re = (; η = (; A = 0.0,)))
    @test res isa FitResult
end

@testset "MCEM constants for fixed effects" begin
    res = fit_model(_MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2);
        constants = (a = 0.2,))
    @test res isa FitResult
end

@testset "MCEM RE distribution with constant covariates" begin
    res = fit_model(fx_recov_dm(),
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
end

@testset "MCEM threaded E-step" begin
    res = fit_model(_MCEM_DM4,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2);
        serialization = EnsembleThreads())
    @test res isa FitResult
end

@testset "MCEM multivariate RE" begin
    res = fit_model(fx_mvnp_dm(),
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
end

@testset "MCEM multivariate RE with NUTS" begin
    res = fit_model(fx_mvnp_dm(),
        NoLimits.MCEM(; sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
end

@testset "MCEM optimizer Adam (OptimizationOptimisers)" begin
    method = NoLimits.MCEM(optimizer = OptimizationOptimisers.Adam(0.05),
        optim_kwargs = (; maxiters = 2),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        maxiters = 2)
    res = fit_model(_MCEM_DM2, method)
    @test res isa FitResult
end

@testset "MCEM optimizer BlackBoxOptim (OptimizationBBO)" begin
    lb, ub = default_bounds_from_start(_MCEM_DM2; margin = 1.0)
    method = NoLimits.MCEM(
        optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs = (; iterations = 3),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        maxiters = 2,
        lb = lb, ub = ub)
    res = fit_model(_MCEM_DM2, method)
    @test res isa FitResult
end

@testset "MCEM with ODE model" begin
    res = fit_model(fx_ode_dm(),
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
end

@testset "MCEM non-normal Poisson outcome" begin
    res = fit_model(fx_pois_dm(),
        NoLimits.MCEM(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2))
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end
