using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using SciMLBase
using OptimizationOptimisers
using OptimizationBBO
using LinearAlgebra

const SAEM_FAST = (maxiters = 2, t0 = 1, kappa = 0.6, mcmc_steps = 1, q_store_max = 2)

# Part 1 of 2 (part 2: estimation_saem2_tests.jl, split for CI shard balance —
# the halves run in different shards; the HMM/LTS fixtures moved with part 2).

# ── shared DataModels (built once, reused across most testsets) ───────────────
# Small: 2 individuals — used by most basic SAEM testsets
const _SAEM_DF_S = DataFrame(
    ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]
)
const _SAEM_DM_S = DataModel(
    @Model(
        begin
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
    ),
    _SAEM_DF_S; primary_id = :ID, time_col = :t
)

# Medium: 4 individuals — used by threaded/minibatch/optimizer testsets
const _SAEM_DF_M = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C, :D, :D], t = repeat([0.0, 1.0], 4),
    y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
)
const _SAEM_DM_M = DataModel(
    @Model(
        begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.1)
                σ = RealNumber(0.4, scale = :log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, 1.0); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    ),
    _SAEM_DF_M; primary_id = :ID, time_col = :t
)

# Diagonal-MvNormal RE with parameterized means/variances — shared by the
# builtin_stats diagonal testsets (closed-form fit, variance-target unit test,
# autodetect).
const _SAEM_DIAG_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        μ1 = RealNumber(0.1)
        μ2 = RealNumber(0.2)
        ω1 = RealNumber(0.5, scale = :log)
        ω2 = RealNumber(0.4, scale = :log)
        σ = RealNumber(0.3, scale = :log)
    end
    @randomEffects begin
        η = RandomEffect(
            MvNormal([μ1, μ2], LinearAlgebra.Diagonal([ω1, ω2])); column = :ID
        )
    end
    @formulas begin
        y ~ Normal(η[1], σ)
    end
end
const _SAEM_DIAG_DM = DataModel(
    _SAEM_DIAG_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.15, 0.2, 0.25, 0.05, 0.1]
    );
    primary_id = :ID, time_col = :t
)

# Two Normal outcomes with separate residual σs — shared by the separate-σ
# builtin_stats testsets (plain, missing-data regression, glm+builtin_stats).
const _SAEM_SEP_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        a = RealNumber(0.1)
        b = RealNumber(0.2)
        σ1 = RealNumber(0.4, scale = :log)
        σ2 = RealNumber(0.3, scale = :log)
        τ = RealNumber(0.3, scale = :log)
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, τ); column = :ID)
    end
    @formulas begin
        y1 ~ Normal(a + η, σ1)
        y2 ~ Normal(b + η, σ2)
    end
end
const _SAEM_SEP_DM = DataModel(
    _SAEM_SEP_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    );
    primary_id = :ID, time_col = :t
)
const _SAEM_SEP_DM_MISSING = DataModel(
    _SAEM_SEP_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = Union{Missing, Float64}[0.1, missing, 0.0, -0.1],
        y2 = Union{Missing, Float64}[missing, 0.25, 0.05, missing]
    );
    primary_id = :ID, time_col = :t
)

@testset "SAEM default sampler" begin
    method = NoLimits.SAEM()
    @test method.saem.sampler isa SaemixMH
    @test method.saem.ebe_multistart_n == 50
    @test method.saem.ebe_multistart_k == 1
    @test method.saem.ebe_multistart_sampling == :lhs
    @test method.saem.ebe_rescue.sampling == :lhs
end

@testset "SAEM closed-form M-step flag metadata" begin
    dm = _SAEM_DM_S

    res_numeric = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            builtin_stats = :none,
            maxiters = 2,
            progress = false,
            q_store_max = 2
        )
    )
    @test !NoLimits.get_closed_form_mstep_used(res_numeric)
    notes_numeric = NoLimits.get_notes(res_numeric)
    @test notes_numeric.closed_form_mstep_used === false
    @test notes_numeric.closed_form_mstep_mode == :numeric_only
    @test isempty(notes_numeric.closed_form_mstep_sources)

    res_builtin = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            builtin_stats = :closed_form,
            maxiters = 2,
            progress = false,
            q_store_max = 2
        )
    )
    @test NoLimits.get_closed_form_mstep_used(res_builtin)
    notes_builtin = NoLimits.get_notes(res_builtin)
    @test notes_builtin.closed_form_mstep_used === true
    @test notes_builtin.closed_form_mstep_mode == :hybrid
    @test :builtin_stats in notes_builtin.closed_form_mstep_sources

    suffstats = (dm, batch_infos, b_current, θ, constants_re) -> begin
        s = 0.0
        for b in b_current
            s += sum(b)
        end
        return (; s)
    end
    mstep_closed_form = (s, dm) -> get_θ0_untransformed(dm.model.fixed.fixed)
    res_custom = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            builtin_stats = :none,
            suffstats = suffstats,
            mstep_closed_form = mstep_closed_form,
            maxiters = 2,
            progress = false,
            q_store_max = 2
        )
    )
    @test NoLimits.get_closed_form_mstep_used(res_custom)
    notes_custom = NoLimits.get_notes(res_custom)
    @test notes_custom.closed_form_mstep_used === true
    @test notes_custom.closed_form_mstep_mode == :closed_form_only
    @test :custom_mstep_closed_form in notes_custom.closed_form_mstep_sources

    @test_logs match_mode = :any (:info, r"SAEM: numerically optimized parameters") begin
        fit_model(
            dm,
            NoLimits.SAEM(;
                sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
                builtin_stats = :auto,
                maxiters = 2,
                progress = false,
                q_store_max = 2
            )
        )
    end
end

# ── custom suffstats/q_from_stats (merged from estimation_saem_suffstats_tests.jl) ──

@testset "SAEM sufficient stats (linear Gaussian)" begin
    dm = fx_re_dm()

    function suffstats(dm, batch_infos, b_current, θ, constants_re)
        # simple quadratic stats for demo
        s1 = 0.0
        s2 = 0.0
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            s1 += sum(b)
            s2 += sum(b .* b)
        end
        return (; s1, s2)
    end

    q_from_stats = (s, θ, dm) -> -0.5 * (s.s1^2 + s.s2^2)

    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            suffstats = suffstats,
            q_from_stats = q_from_stats
        )
    )
    @test res isa FitResult
end

@testset "SAEM sufficient stats (nonlinear Gaussian)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            c = RealNumber(0.1)
            σ = RealNumber(0.5, scale = :log)
            τ = RealNumber(0.4, scale = :log)
        end

        @covariates begin
            t = Covariate()
            x = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            μ = exp(a + c * x + η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        x = [0.1, 0.2, 0.15, 0.3],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    function suffstats(dm, batch_infos, b_current, θ, constants_re)
        s1 = 0.0
        s2 = 0.0
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            s1 += sum(b)
            s2 += sum(b .* b)
        end
        return (; s1, s2)
    end

    q_from_stats = (s, θ, dm) -> -0.5 * (s.s1^2 + s.s2^2)

    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            suffstats = suffstats,
            q_from_stats = q_from_stats
        )
    )
    @test res isa FitResult
end

# MvLogNormal / MvLogitNormal models with parameterized mean + PSD covariance,
# shared by the default-sampler and builtin-stats testsets below. (Kept in the
# merge: saem_mh_kernel covers these families only with AdaptiveNoLimitsMH and
# literal means, so neither the default SaemixMH fit nor the parameterized-mean
# closed-form update is exercised elsewhere.)
const _SFX_MVLN_MODEL = @Model begin
    @fixedEffects begin
        μ = RealVector([0.0, 0.0])
        Ω = RealPSDMatrix(Matrix(I, 2, 2); scale = :cholesky)
        σ = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(MvLogNormal(μ, Ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(η[1], σ)
    end
end

const _SFX_MVLIT_MODEL = @Model begin
    @fixedEffects begin
        μ = RealVector([0.0, 0.0])
        Ω = RealPSDMatrix(Matrix(I, 2, 2); scale = :cholesky)
        σ = RealNumber(0.1, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(MvLogitNormal(μ, Ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(η[1], σ)
    end
end

@testset "SAEM default sampler + RealPSDMatrix: MvLogNormal and MvLogitNormal RE" begin
    n_id = 8
    ids = repeat(1:n_id, inner = 3)
    ts = repeat([0.0, 0.5, 1.0], n_id)
    Omega_true = [1.0 0.4; 0.4 1.0]

    # MvLogNormal with default SaemixMH
    etas_ln = exp.(rand(MvNormal([0.0, 0.0], Omega_true), n_id))
    df_ln = DataFrame(ID = ids, t = ts, y = etas_ln[1, ids] .+ 0.3 .* randn(length(ids)))
    dm_ln = DataModel(_SFX_MVLN_MODEL, df_ln; primary_id = :ID, time_col = :t)
    res_ln = fit_model(dm_ln, NoLimits.SAEM(maxiters = 3, progress = false))
    @test res_ln isa FitResult
    @test isfinite(NoLimits.get_params(res_ln; scale = :untransformed).σ)

    # MvLogitNormal with default SaemixMH
    etas_lit = rand(MvLogitNormal([0.0, 0.0], Omega_true), n_id)
    df_lit = DataFrame(ID = ids, t = ts, y = etas_lit[1, ids] .+ 0.05 .* randn(length(ids)))
    dm_lit = DataModel(_SFX_MVLIT_MODEL, df_lit; primary_id = :ID, time_col = :t)
    res_lit = fit_model(dm_lit, NoLimits.SAEM(maxiters = 3, progress = false))
    @test res_lit isa FitResult
    @test isfinite(NoLimits.get_params(res_lit; scale = :untransformed).σ)
end

@testset "SAEM builtin stats MvLogNormal and MvLogitNormal RE" begin
    # MvLogNormal: samples in (0,∞)^d, M-step transforms with log
    df_ln = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [1.2, 1.3, 0.8, 0.9]
    )
    dm_ln = DataModel(_SFX_MVLN_MODEL, df_ln; primary_id = :ID, time_col = :t)
    res_ln = fit_model(
        dm_ln,
        NoLimits.SAEM(;
            sampler = AdaptiveNoLimitsMH(adapt_start = 2), maxiters = 3, mcmc_steps = 5, progress = false
        )
    )
    @test res_ln isa FitResult
    @test isfinite(NoLimits.get_params(res_ln; scale = :untransformed).σ)

    # MvLogitNormal: samples in (0,1)^d, M-step transforms with logit
    df_lit = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.4, 0.45, 0.55, 0.5]
    )
    dm_lit = DataModel(_SFX_MVLIT_MODEL, df_lit; primary_id = :ID, time_col = :t)
    res_lit = fit_model(
        dm_lit,
        NoLimits.SAEM(;
            sampler = AdaptiveNoLimitsMH(adapt_start = 2), maxiters = 3, mcmc_steps = 5, progress = false
        )
    )
    @test res_lit isa FitResult
    @test isfinite(NoLimits.get_params(res_lit; scale = :untransformed).σ)
end

@testset "SAEM basic (random effects)" begin
    dm = _SAEM_DM_S
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...
        )
    )
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

@testset "SAEM/MCEM serial vs threaded is reproducible" begin
    Threads.nthreads() < 2 && return

    dm = _SAEM_DM_S
    tk = (n_samples = 2, n_adapt = 2, progress = false, verbose = false)
    for (label, method) in (
            (
                "SAEM",
                NoLimits.SAEM(;
                    sampler = MH(), turing_kwargs = tk, maxiters = 2,
                    mcmc_steps = 1, q_store_max = 2, progress = false
                ),
            ),
            (
                "MCEM",
                NoLimits.MCEM(;
                    sampler = MH(), turing_kwargs = tk, maxiters = 2,
                    progress = false
                ),
            ),
        )
        @testset "$label" begin
            res_serial = fit_model(
                dm, method; serialization = EnsembleSerial(), rng = MersenneTwister(123)
            )
            res_threads = fit_model(
                dm, method; serialization = EnsembleThreads(), rng = MersenneTwister(123)
            )
            @test res_serial.summary.objective == res_threads.summary.objective
            @test collect(NoLimits.get_params(res_serial, scale = :untransformed)) ==
                collect(NoLimits.get_params(res_threads, scale = :untransformed))
        end
    end
end

@testset "SAEM basic with NUTS" begin
    dm = _SAEM_DM_S
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...
        )
    )
    @test res isa FitResult
end

@testset "_half_window_test" begin
    # Flat noise: drift stays within the MC-noise floor, so the test passes.
    pass, d, s = NoLimits._half_window_test([1.0, 1.01, 0.99, 1.0], 1.0e-4, 1.0e-3)
    @test pass
    @test d ≈ 0.01
    @test s ≈ 1.005
    # Clear trend fails: drift far exceeds both tolerance and noise floor.
    pass, d, _ = NoLimits._half_window_test(collect(1.0:0.1:1.7), 1.0e-4, 1.0e-3)
    @test !pass
    @test d ≈ 0.4
    # Non-finite value: hard fail with NaN drift.
    pass, d, _ = NoLimits._half_window_test([1.0, Inf, 1.0, 1.0], 1.0e-4, 1.0e-3)
    @test !pass
    @test isnan(d)
    # atol == rtol == 0 disables the test regardless of the noise floor.
    pass, _, _ = NoLimits._half_window_test([1.0, 1.01, 0.99, 1.0], 0.0, 0.0)
    @test !pass
    # Inf tolerance always passes; odd window ignores the middle element.
    pass, d, _ = NoLimits._half_window_test([0.0, 0.0, 100.0, 1.0, 1.0], Inf, 0.0)
    @test pass
    @test d ≈ 1.0
    # Vector windows: a single noiseless trending coordinate blocks the pass.
    pass, d, s = NoLimits._half_window_test(
        [[1.0, 10.0], [1.0, 10.0], [1.0, 12.0], [1.0, 12.0]], 1.0e-4, 1.0e-3
    )
    @test !pass
    @test d ≈ 2.0
    @test s ≈ 10.0
    # Scale floor of 1 for small-magnitude trajectories.
    _, _, s = NoLimits._half_window_test([0.1, 0.1, 0.2, 0.2], 1.0e-4, 1.0e-3)
    @test s == 1.0
end

@testset "SAEM/MCEM convergence requires both parameter and Q stabilization" begin
    dm = _SAEM_DM_S
    tk = (n_samples = 2, n_adapt = 2, progress = false)
    @testset "SAEM" begin
        # θ drift passes as soon as the window fills (Inf tolerance) but Q never does
        # (zero tolerance): without the Q gate this would stop at iteration 4.
        res = fit_model(
            dm,
            NoLimits.SAEM(;
                sampler = MH(), turing_kwargs = tk, sa_burnin_iters = 0,
                t0 = 0, q_store_max = 2, maxiters = 8, convergence_window = 4,
                consecutive_params = 1, atol_theta = Inf, rtol_theta = Inf,
                atol_Q = 0.0, rtol_Q = 0.0
            )
        )
        @test res.result.iterations == 8
        @test !NoLimits.get_converged(res)
    end
    @testset "MCEM" begin
        # Same discrimination for MCEM: θ passes once its window fills (iteration 4)
        # but Q never does, so the fit must run to maxiters.
        res = fit_model(
            dm,
            NoLimits.MCEM(;
                sampler = MH(), turing_kwargs = tk, maxiters = 8,
                convergence_window = 4, consecutive_params = 1,
                atol_theta = Inf, rtol_theta = Inf, atol_Q = 0.0, rtol_Q = 0.0
            )
        )
        @test res.result.iterations == 8
        @test !NoLimits.get_converged(res)
    end
end

@testset "SAEM windowed drift test triggers early stop" begin
    # Inf tolerances make every post-window-fill check pass, so the stop point is
    # deterministic: stabilization (t0=5) + window fill (10) + consecutive (2) - 1.
    res = fit_model(
        _SAEM_DM_S,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            sa_burnin_iters = 0, t0 = 5, maxiters = 100, mcmc_steps = 1,
            q_store_max = 2, convergence_window = 10, consecutive_params = 2,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf
        )
    )
    @test NoLimits.get_converged(res)
    @test 16 <= res.result.iterations < 100
    diag = NoLimits.get_notes(res).diagnostics
    @test isnan(diag.drift_θ[1])  # window not yet full
    @test isfinite(diag.drift_θ[end])
    @test isfinite(diag.drift_Q[end])
end

@testset "SAEM no early stop before drift window fills" begin
    res = fit_model(
        _SAEM_DM_S,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            sa_burnin_iters = 0, t0 = 0, maxiters = 6, mcmc_steps = 1,
            q_store_max = 2, convergence_window = 10, consecutive_params = 1,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf
        )
    )
    @test !NoLimits.get_converged(res)
    @test res.result.iterations == 6
end

@testset "SAEM/MCEM multiple RE groups" begin
    dm = fx_mg_dm()
    tk = (n_samples = 2, n_adapt = 2, progress = false)
    for (label, method) in (
            ("SAEM", NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, SAEM_FAST...)),
            ("MCEM", NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 2)),
        )
        @testset "$label" begin
            res = fit_model(dm, method)
            @test res isa FitResult
            re = NoLimits.get_random_effects(dm, res)
            @test !isempty(re)
        end
    end
end

@testset "SAEM constants_re" begin
    res = fit_model(
        _SAEM_DM_S,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...
        );
        constants_re = (; η = (; A = 0.0))
    )
    @test res isa FitResult
end

@testset "SAEM threaded updates" begin
    dm = _SAEM_DM_M
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...
        );
        serialization = EnsembleThreads()
    )
    @test res isa FitResult
end

@testset "SAEM minibatch updates" begin
    dm = _SAEM_DM_M
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            update_schedule = 1, SAEM_FAST...
        )
    )
    @test res isa FitResult
end

@testset "SAEM optimizer Adam (OptimizationOptimisers)" begin
    dm = _SAEM_DM_S
    method = NoLimits.SAEM(;
        optimizer = OptimizationOptimisers.Adam(0.05),
        optim_kwargs = (; maxiters = 2),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        SAEM_FAST...
    )
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "SAEM optimizer BlackBoxOptim (OptimizationBBO)" begin
    dm = _SAEM_DM_S
    lb, ub = default_bounds_from_start(dm; margin = 1.0)
    method = NoLimits.SAEM(;
        optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs = (; iterations = 3),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        SAEM_FAST...,
        lb = lb, ub = ub
    )
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "SAEM constants for fixed effects" begin
    dm = _SAEM_DM_S
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...
        );
        constants = (a = 0.2,)
    )
    @test res isa FitResult
end

@testset "SAEM RE distribution with constant covariates" begin
    # fx_recov: η ~ Normal(b * Age, 0.5) with Age a ConstantCovariate.
    res = fit_model(
        fx_recov_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (scalar RE)" begin
    # fx_re_model: η ~ Normal(0, ω) with ω a fixed effect → re_cov_params = (; η = :ω).
    res = fit_model(
        fx_re_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :ω)
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multivariate RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale = :log)
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale = :cholesky)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :Ω)
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multivariate diagonal + means)" begin
    res = fit_model(
        _SAEM_DIAG_DM,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = (:ω1, :ω2)),
            re_mean_params = (; η = (:μ1, :μ2))
        )
    )
    @test res isa FitResult
    θ = NoLimits.get_params(res; scale = :untransformed)
end

@testset "SAEM builtin_stats uses variance for MvNormal diagonal targets" begin
    dm = _SAEM_DIAG_DM
    θ = NoLimits.get_θ0_untransformed(_SAEM_DIAG_MODEL.fixed.fixed)
    stats = (;
        re = (;
            η = (
                family = :mvnormal, mean = [0.0, 0.0],
                second = [0.25 0.0; 0.0 0.04], n = 10,
            ),
        ),
        outcome = NamedTuple(),
        hmm = NamedTuple(),
    )

    updates = NoLimits._saem_builtin_updates_from_smoothed_stats(
        dm,
        θ,
        stats,
        NamedTuple(),
        NamedTuple(),
        (; η = (:ω1, :ω2)),
        NamedTuple()
    )
    @test isapprox(updates.ω1, 0.25; atol = 1.0e-12)
    @test isapprox(updates.ω2, 0.04; atol = 1.0e-12)
end

@testset "SAEM builtin_stats gaussian_re respects fixed-effect lower bounds for RE means" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ = RealNumber(0.2, lower = 0.0)
            ω = RealNumber(0.5, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ, ω); column = :ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [-2.1, -1.8, -2.0, -1.9, -2.2, -1.7]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :ω),
            re_mean_params = (; η = :μ)
        )
    )
    @test res isa FitResult
    θ = NoLimits.get_params(res; scale = :untransformed)
end

@testset "SAEM builtin_stats gaussian_re (multiple RE dists)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale = :log)
            τ_id = RealNumber(0.3, scale = :log)
            τ_site = RealNumber(0.2, scale = :log)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, τ_id); column = :ID)
            η_site = RandomEffect(Normal(0.0, τ_site); column = :SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        SITE = [:X, :X, :Y, :Y],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η_id = :τ_id, η_site = :τ_site)
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multiple normal outcomes)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale = :log)
            τ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ)
            y2 ~ Normal(b + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :τ)
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multiple normal outcomes, separate σ)" begin
    res = fit_model(
        _SAEM_SEP_DM,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = (; y1 = :σ1, y2 = :σ2),
            re_cov_params = (; η = :τ)
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats skips missing normal outcomes (regression)" begin
    res = fit_model(
        _SAEM_SEP_DM_MISSING,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :closed_form,
            resid_var_param = (; y1 = :σ1, y2 = :σ2),
            re_cov_params = (; η = :τ)
        )
    )
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test :builtin_stats in notes.closed_form_mstep_sources
    θ = NoLimits.get_params(res; scale = :untransformed)
end

@testset "SAEM builtin_stats gaussian_re falls back for non-Normal outcomes" begin
    # fx_bern_model: y ~ Bernoulli(logistic(a + η)), η ~ Normal(0, ω).
    res = fit_model(
        fx_bern_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            re_cov_params = (; η = :ω)
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects gaussian_re (scalar RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale = :log)
            τ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(a, τ); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(
        dm, NoLimits.get_names(model.fixed.fixed)
    )
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    @test auto_cfg.re_mean_params == (; η = :a)
    @test auto_cfg.resid_var_param == :σ

    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :auto
        )
    )
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects gaussian_re (MvNormal diagonal + means)" begin
    dm = _SAEM_DIAG_DM
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(
        dm, NoLimits.get_names(_SAEM_DIAG_MODEL.fixed.fixed)
    )
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = (:ω1, :ω2))
    @test auto_cfg.re_mean_params == (; η = (:μ1, :μ2))
    @test auto_cfg.resid_var_param == :σ

    res = fit_model(
        dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :auto
        )
    )
    @test res isa FitResult
end

# NOTE: "auto detects MvNormal symbol mean with fixed diagonal expression" lives in
# estimation_saem_autodetect_tests.jl (identical model + assertions; was duplicated here).
