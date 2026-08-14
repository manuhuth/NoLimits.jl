using Test
using NoLimits
using OptimizationNLopt
using LikelihoodProfiler
using DataFrames
using Distributions
using ComponentArrays
using Random
using LinearAlgebra
using OptimizationOptimJL
using NoLimits.LineSearches

# Profile intervals must be real numbers that bracket the estimate on both scales.
function _assert_profile_interval(uq)
    for scale in (:natural, :transformed)
        est = get_uq_estimates(uq; scale = scale, as_component = false)
        ints = get_uq_intervals(uq; scale = scale, as_component = false)
        @test ints !== nothing
        @test all(isfinite, ints.lower)
        @test all(isfinite, ints.upper)
        @test all(ints.lower .< est .< ints.upper)
    end
    @test all(get_uq_diagnostics(uq).endpoint_found)
end

# ── Shared models (UQ assertions need specific calculate_se flag patterns, so
# these stay local rather than using the fx_ fixtures; each distinct flag
# pattern is compiled once and shared across the testsets that use it.) ──────

# No-RE, mixed se flags, no priors (Wald MLE).
const _UQ_NORE_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.2, calculate_se = true)
        b = RealNumber(0.1, calculate_se = false)
        σ = RealNumber(0.3, scale = :log, calculate_se = true)
    end
    @formulas begin
        y ~ Normal(a + b * t, σ)
    end
end

# Same shape with priors (MAP / MCMC / VI / mcmc_refit).
const _UQ_NORE_P_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.2, prior = Normal(0.0, 1.0), calculate_se = true)
        b = RealNumber(0.1, prior = Normal(0.0, 1.0), calculate_se = false)
        σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5), calculate_se = true)
    end
    @formulas begin
        y ~ Normal(a + b * t, σ)
    end
end
const _UQ_NORE_P_DM = DataModel(_UQ_NORE_P_MODEL,
    DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.3, 0.1, 0.2]);
    primary_id = :ID, time_col = :t)

# Scalar RE with se on (a, ω) but not σ (Wald / sandwich / MCEM testsets).
const _UQ_RE_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.2, calculate_se = true)
        ω = RealNumber(0.6, scale = :log, calculate_se = true)
        σ = RealNumber(0.3, scale = :log, calculate_se = false)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end
const _UQ_RE_DM = DataModel(_UQ_RE_MODEL,
    DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.25, 0.1, 0.15, 0.3, 0.35]);
    primary_id = :ID, time_col = :t)
# One Laplace fit shared by the Wald and sandwich testsets.
const _UQ_RE_RES_LAP = fit_model(
    _UQ_RE_DM, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

# Single-se no-RE model (profile MLE + mcmc_refit error path).
const _UQ_SE1_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.2, calculate_se = true)
        σ = RealNumber(0.3, scale = :log, calculate_se = false)
    end
    @formulas begin
        y ~ Normal(a, σ)
    end
end

@testset "UQ Wald for MLE" begin
    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.3, 0.1, 0.2, 0.25, 0.35]
    )
    dm = DataModel(_UQ_NORE_MODEL, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(res; method = :wald, n_draws = 30, rng = Random.Xoshiro(1))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :mle
    names = get_uq_parameter_names(uq)
    @test names == [:a, :σ]
    est = get_uq_estimates(uq)
    @test est isa ComponentArray
    ints = get_uq_intervals(uq)
    @test ints !== nothing
    @test hasproperty(ints, :lower)
    @test size(get_uq_vcov(uq)) == (2, 2)
    @test size(get_uq_draws(uq)) == (30, 2)
    Vt = get_uq_vcov(uq; scale = :transformed)
    λmin = minimum(eigvals(Symmetric(0.5 .* (Vt .+ Vt'))))
    @test λmin >= -1e-10
    d = get_uq_diagnostics(uq)
    @test haskey(d, :vcov_projected)
    @test haskey(d, :vcov_min_eig_raw)
    @test haskey(d, :vcov_min_eig_used)
    @test haskey(d, :vcov_n_eigs_clipped)
    @test haskey(d, :hessian_reduced)
    @test haskey(d, :inactive_fixed_effects_held_constant)
    @test d.hessian_reduced
    @test d.inactive_fixed_effects_held_constant
    @test d.vcov_min_eig_used >= -1e-10

    uq_const = compute_uq(
        res; method = :wald, constants = (a = 0.2,), n_draws = 30, rng = Random.Xoshiro(2))
    @test get_uq_parameter_names(uq_const) == [:σ]
end

@testset "UQ Wald for MAP" begin
    res = fit_model(_UQ_NORE_P_DM, NoLimits.MAP(; optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(res; method = :wald, n_draws = 30, rng = Random.Xoshiro(3))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :map
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_vcov(uq)) == (2, 2)
end

@testset "UQ chain for MCMC" begin
    res = fit_model(_UQ_NORE_P_DM,
        NoLimits.MCMC(; turing_kwargs = (n_samples = 18, n_adapt = 2, progress = false)))

    uq = compute_uq(res; method = :chain, mcmc_draws = 15, rng = Random.Xoshiro(4))
    @test get_uq_backend(uq) == :chain
    @test get_uq_source_method(uq) == :mcmc
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_draws(uq)) == (15, 2)
    d = get_uq_diagnostics(uq)
    @test d.requested_draws == 15
    @test d.used_draws == 15
    @test d.available_draws >= d.used_draws
end

@testset "UQ chain for VI" begin
    res = fit_model(
        _UQ_NORE_P_DM, NoLimits.VI(; turing_kwargs = (max_iter = 15, progress = false));
        rng = Random.Xoshiro(401))

    uq = compute_uq(res; method = :chain, mcmc_draws = 35, rng = Random.Xoshiro(402))
    @test get_uq_backend(uq) == :chain
    @test get_uq_source_method(uq) == :vi
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_draws(uq)) == (35, 2)
    d = get_uq_diagnostics(uq)
    @test d.requested_draws == 35
    @test d.used_draws == 35
    @test d.source == :vi_posterior

    uq_auto = compute_uq(res; method = :auto, n_draws = 22, rng = Random.Xoshiro(403))
    @test get_uq_backend(uq_auto) == :chain
    @test size(get_uq_draws(uq_auto)) == (22, 2)

    uq_const = compute_uq(res; method = :chain, constants = (a = 0.2,),
        mcmc_draws = 20, rng = Random.Xoshiro(404))
    @test get_uq_parameter_names(uq_const) == [:σ]
    @test size(get_uq_draws(uq_const)) == (20, 1)
end

@testset "UQ errors when no active calculate_se parameters" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se = false)
            σ = RealNumber(0.3, scale = :log, calculate_se = false)
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.2, 0.3]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    @test_throws ErrorException compute_uq(res; method = :wald, n_draws = 50)
end

@testset "UQ Wald for Laplace" begin
    res = _UQ_RE_RES_LAP

    uq = compute_uq(res; method = :wald, n_draws = 30, rng = Random.Xoshiro(5))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :laplace
    @test get_uq_parameter_names(uq) == [:a, :ω]
    @test size(get_uq_vcov(uq)) == (2, 2)
    @test size(get_uq_draws(uq)) == (30, 2)
    d = get_uq_diagnostics(uq)
    @test haskey(d, :hessian_reduced)
    @test haskey(d, :inactive_fixed_effects_held_constant)
    @test d.hessian_reduced
    @test d.inactive_fixed_effects_held_constant
end

function _uq_psd_re_model(scale::Symbol)
    if scale === :lie
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.2, calculate_se = false)
                σ = RealNumber(0.5, scale = :log, calculate_se = false)
                Ω = RealLiePSDMatrix(Matrix{Float64}(I, 2, 2); calculate_se = true)
            end
            @randomEffects begin
                η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η[1] + η[2] * t, σ)
            end
        end
    end
    return @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se = false)
            σ = RealNumber(0.5, scale = :log, calculate_se = false)
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale = scale, calculate_se = true)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2] * t, σ)
        end
    end
end

# 12 subjects x 3 observations, random intercept AND random slope: with only `η[1]` in the
# formula the second random effect never reached the likelihood, so two of the three Ω
# coordinates were structurally unidentified and their Wald variance was ~1e12 - draws then
# overflowed on the natural scale and `get_uq_vcov` was a coin flip on where the optimizer
# landed (#159). Both effects must enter the observation for the covariance to be estimable.
function _uq_psd_re_df()
    ids = [Symbol("S", i) for i in 1:12]
    y = [0.10, 0.21, 0.32, 0.02, -0.08, -0.15, 0.15, 0.19, 0.27, -0.05, 0.04, 0.11,
        0.22, 0.30, 0.41, -0.12, -0.06, 0.01, 0.08, 0.17, 0.25, 0.31, 0.36, 0.44,
        -0.02, 0.06, 0.14, 0.18, 0.24, 0.29, 0.05, 0.12, 0.20, -0.09, -0.01, 0.07]
    return DataFrame(ID = repeat(ids; inner = 3), t = repeat([0.0, 1.0, 2.0], 12), y = y)
end

@testset "UQ Wald for Laplace with PSD fixed covariance (cholesky/expm/lie)" begin
    # n_coords is the transformed free-coordinate count: n(n+1)/2 = 3 for a 2x2 in
    # every PSD scale (:cholesky stores the lower triangle, :expm the upper, :lie [λ; α]).
    for (scale, n_coords, seed) in ((:cholesky, 3, 31), (:expm, 3, 32), (:lie, 3, 33))
        model = _uq_psd_re_model(scale)
        dm = DataModel(model, _uq_psd_re_df(); primary_id = :ID, time_col = :t)
        # The Wald Hessian is only finite at a properly converged estimate - at a degenerate
        # point it is NaN and `compute_uq` says so rather than returning NaN silently. The step
        # cap is what keeps the fit out of that region.
        res = fit_model(dm,
            NoLimits.Laplace(;
                optimizer = OptimizationOptimJL.LBFGS(
                    linesearch = LineSearches.BackTracking(maxstep = 1.0))))

        uq = compute_uq(res; method = :wald, pseudo_inverse = true,
            n_draws = 40, rng = Random.Xoshiro(seed))
        @test get_uq_backend(uq) == :wald
        @test get_uq_source_method(uq) == :laplace
        names = get_uq_parameter_names(uq)
        @test length(names) == n_coords
        @test all(s -> startswith(String(s), "Ω_"), names)
        V = get_uq_vcov(uq)
        @test size(V) == (n_coords, n_coords)
        @test all(isfinite, V)
        @test isapprox(V, V'; rtol = 1e-10, atol = 1e-10)
        @test size(get_uq_draws(uq)) == (40, n_coords)
    end
end

@testset "Wald natural-scale draw filter rejects unsquarable rows" begin
    # Row observed in the #159 reproduction: every entry is finite, so the old `isfinite`
    # filter kept it, and 4.19e159 squared is Inf - which made get_uq_vcov() all-Inf.
    row = [5.152933056233719e158, 1.4685718986792252e159, 4.185389947927327e159]
    @test all(isfinite, row)
    @test !NoLimits._wald_usable_draw_row(row, 40)
    @test NoLimits._wald_usable_draw_row([0.0134, -4.48e-7, 1.49e-11], 40)
    @test !NoLimits._wald_usable_draw_row([Inf, 0.0, 0.0], 40)
    @test !NoLimits._wald_usable_draw_row([NaN, 0.0, 0.0], 40)
end

@testset "UQ profile for MLE" begin
    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.25, 0.1, 0.15, 0.3, 0.35]
    )
    dm = DataModel(_UQ_SE1_MODEL, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(res;
        method = :profile,
        profile_method = :LIN_EXTRAPOL,
        profile_scan_width = 1.0,
        profile_max_iter = 80,
        rng = Random.Xoshiro(9))
    @test get_uq_backend(uq) == :profile
    @test get_uq_source_method(uq) == :mle
    @test get_uq_parameter_names(uq) == [:a]
    # A dead profiler backend returns NaN bounds and still passes `ints !== nothing`,
    # which is how the LikelihoodProfiler 0.x -> 1.x API break went unnoticed (#139).
    _assert_profile_interval(uq)
    d = get_uq_diagnostics(uq)
    @test haskey(d, :profile_method)
    @test d.profile_method == :LIN_EXTRAPOL
    @test all(isnothing, d.errors)
    @test all(==(:Identifiable), d.left_status)
    @test all(==(:Identifiable), d.right_status)

    # The 0.x-only CICO tolerances are ignored, and say so instead of pretending to act.
    @test_logs (:warn, r"profile_scan_tol") NoLimits._warn_removed_profile_kw(
        :profile_scan_tol, 1e-2)
    @test_logs (:warn, r"profile_loss_tol") NoLimits._warn_removed_profile_kw(
        :profile_loss_tol, 1e-2)
    @test_logs NoLimits._warn_removed_profile_kw(:profile_scan_tol, nothing)
end

@testset "UQ profile for Laplace" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se = true)
            ω = RealNumber(0.6, scale = :log, calculate_se = false)
            σ = RealNumber(0.3, scale = :log, calculate_se = false)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.25, 0.1, 0.15, 0.3, 0.35]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(res;
        method = :profile,
        profile_method = :LIN_EXTRAPOL,
        profile_scan_width = 0.8,
        profile_max_iter = 80,
        rng = Random.Xoshiro(10))
    @test get_uq_backend(uq) == :profile
    @test get_uq_source_method(uq) == :laplace
    @test get_uq_parameter_names(uq) == [:a]
    _assert_profile_interval(uq)
end

@testset "UQ mcmc_refit for MLE" begin
    res = fit_model(_UQ_NORE_P_DM, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(res;
        method = :mcmc_refit,
        mcmc_turing_kwargs = (n_samples = 12, n_adapt = 2, progress = false),
        mcmc_draws = 9,
        rng = Random.Xoshiro(11))
    @test get_uq_backend(uq) == :mcmc_refit
    @test get_uq_source_method(uq) == :mle
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_draws(uq)) == (9, 2)
    d = get_uq_diagnostics(uq)
    @test d.requested_draws == 9
    @test d.used_draws == 9
    @test d.available_draws >= d.used_draws
    @test haskey(d, :sampled_fixed_names)
    @test :b ∉ d.sampled_fixed_names
end

@testset "UQ mcmc_refit errors without priors on sampled fixed effects" begin
    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.2, 0.3]
    )
    dm = DataModel(_UQ_SE1_MODEL, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test_throws ErrorException compute_uq(res;
        method = :mcmc_refit,
        mcmc_turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false))
end

@testset "UQ Wald sandwich for MLE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se = true)
            σ = RealNumber(0.3, scale = :log, calculate_se = true)
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.2, 0.25, 0.1, 0.15, 0.3, 0.35, 0.18, 0.22]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(
        res; method = :wald, vcov = :sandwich, n_draws = 30, rng = Random.Xoshiro(12))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :mle
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.vcov == :sandwich
end

@testset "UQ Wald sandwich for Laplace" begin
    uq = compute_uq(_UQ_RE_RES_LAP;
        method = :wald, vcov = :sandwich, n_draws = 30, rng = Random.Xoshiro(13))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :laplace
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.vcov == :sandwich
end

@testset "UQ Wald for MCEM via Laplace approximation" begin
    res = fit_model(_UQ_RE_DM,
        NoLimits.MCEM(;
            maxiters = 2,
            sample_schedule = 2,
            turing_kwargs = (n_adapt = 2, progress = false),
            optim_kwargs = (maxiters = 2,)))

    uq = compute_uq(res; method = :wald, n_draws = 40, rng = Random.Xoshiro(21))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :mcem
    @test get_uq_parameter_names(uq) == [:a, :ω]
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.approximation_method == :laplace
    @test_throws ErrorException compute_uq(res; method = :wald, re_approx = :invalid)
end

# ── :logit scale UQ helpers ──────────────────────────────────────────────────

@testset "_flat_transform_kinds_for_free — :logit uniform" begin
    fe = @fixedEffects begin
        p = RealNumber(0.3; scale = :logit, calculate_se = true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:p])
    @test kinds == [:logit]
end

@testset "_flat_transform_kinds_for_free — :logit vector" begin
    fe = @fixedEffects begin
        v = RealVector(
            [0.2, 0.5, 0.8]; scale = [:logit, :logit, :logit], calculate_se = true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:v])
    @test kinds == [:logit, :logit, :logit]
end

@testset "_flat_transform_kinds_for_free — :elementwise mixed" begin
    fe = @fixedEffects begin
        v = RealVector(
            [0.4, 2.0, -1.0]; scale = [:logit, :log, :identity], calculate_se = true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:v])
    @test kinds == [:logit, :log, :identity]
end

@testset "_flat_transform_kinds_for_free — mixed params" begin
    fe = @fixedEffects begin
        p = RealNumber(0.3; scale = :logit, calculate_se = true)
        σ = RealNumber(1.0; scale = :log, calculate_se = true)
        a = RealNumber(0.5; scale = :identity, calculate_se = true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:p, :σ, :a])
    @test kinds == [:logit, :log, :identity]
end

@testset "_flat_transform_kinds_for_free — :elementwise two params" begin
    fe = @fixedEffects begin
        v = RealVector(
            [0.3, 2.0, -1.0]; scale = [:logit, :log, :identity], calculate_se = true)
        σ = RealNumber(0.5; scale = :log, calculate_se = true)
    end
    kinds = NoLimits._flat_transform_kinds_for_free(fe, [:v, :σ])
    @test kinds == [:logit, :log, :identity, :log]
end

@testset "_wald_closed_form_kind — :logit on natural scale" begin
    # coord_transforms[j] == :logit → :logitnormal
    coord_transforms = [:logit]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :logitnormal
end

@testset "_wald_closed_form_kind — :log on natural scale" begin
    coord_transforms = [:log]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :lognormal
end

@testset "_wald_closed_form_kind — :identity on natural scale" begin
    coord_transforms = [:identity]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :normal
end

@testset "_wald_closed_form_kind — any on transformed scale" begin
    # On transformed scale, always :normal regardless of coord transform kind
    coord_transforms = [:logit]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :transformed, 1, vcov_t, coord_transforms)
    @test kind == :normal
end

@testset "_wald_closed_form_kind — :cholesky on natural scale → :none" begin
    coord_transforms = [:cholesky]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, coord_transforms)
    @test kind == :none
end

@testset "_wald_closed_form_kind — not wald backend → :none" begin
    coord_transforms = [:logit]
    vcov_t = ones(1, 1)
    kind = NoLimits._wald_closed_form_kind(:chain, :natural, 1, vcov_t, coord_transforms)
    @test kind == :none
end

# Note: _wald_density_xy takes v = variance (not sigma). σ = sqrt(v) inside.

@testset "_wald_density_xy — :logitnormal basic properties" begin
    # μ=0 (logit(0.5)=0), v=1.0 → σ=1: LogitNormal(0, 1)
    # Density should be on (0,1), symmetric around 0.5
    μ = 0.0
    v = 1.0  # variance → σ = 1.0
    result = NoLimits._wald_density_xy(:logitnormal, μ, v)
    @test result !== nothing
    x, y = result
    @test length(x) == length(y)
    # All x in (0, 1)
    @test all(0.0 .< x .< 1.0)
    # All density values non-negative
    @test all(y .>= 0.0)
    # Density integrates close to 1 (trapezoid rule check)
    area = sum((x[2:end] .- x[1:(end - 1)]) .* (y[2:end] .+ y[1:(end - 1)]) ./ 2)
    @test isapprox(area, 1.0; atol = 0.01)
end

@testset "_wald_density_xy — :logitnormal matches Distributions.LogitNormal" begin
    μ = 1.0
    σ = 0.5
    v = σ^2  # variance → pass v to _wald_density_xy
    result = NoLimits._wald_density_xy(:logitnormal, μ, v)
    @test result !== nothing
    x, y = result
    dist = LogitNormal(μ, σ)  # σ = sqrt(v)
    # Check density at a few interior points
    for (xi, yi) in zip(x[50:10:(end - 50)], y[50:10:(end - 50)])
        @test isapprox(yi, pdf(dist, xi); rtol = 1e-8)
    end
end

@testset "_wald_density_xy — :logitnormal zero variance returns nothing" begin
    result = NoLimits._wald_density_xy(:logitnormal, 0.0, 0.0)
    @test result === nothing
end

@testset "_wald_density_xy — :lognormal still works (regression)" begin
    result = NoLimits._wald_density_xy(:lognormal, 0.0, 1.0)  # v=1.0 → σ=1.0
    @test result !== nothing
    x, y = result
    @test all(x .> 0.0)
    @test all(y .>= 0.0)
end

@testset "_wald_density_xy — :normal still works (regression)" begin
    result = NoLimits._wald_density_xy(:normal, 0.0, 1.0)  # v=1.0 → σ=1.0
    @test result !== nothing
    x, y = result
    @test all(isfinite.(x))
    @test all(y .>= 0.0)
end

# #173: pinv's default rank tolerance mapped a near-zero singular value to 0, so a weakly
# identified direction was reported as near-zero variance - falsely precise. It must be large.
@testset "Wald bread keeps weakly identified directions large" begin
    H = [1.0 1.0; 1.0 1.0+1e-10]
    B = NoLimits._wald_pinv(H, [:a, :b])
    @test diag(B)[2] > 1e8
    @test isapprox(B, inv(H); rtol = 1e-3)
end
