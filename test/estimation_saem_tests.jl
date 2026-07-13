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

# ── shared DataModels (built once, reused across most testsets) ───────────────
# Small: 2 individuals — used by most basic SAEM testsets
const _SAEM_DF_S = DataFrame(
    ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1])
const _SAEM_DM_S = DataModel(
    @Model(begin
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
    end),
    _SAEM_DF_S; primary_id = :ID, time_col = :t)

# Medium: 4 individuals — used by threaded/minibatch/optimizer testsets
const _SAEM_DF_M = DataFrame(
    ID = [:A, :A, :B, :B, :C, :C, :D, :D], t = repeat([0.0, 1.0], 4),
    y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1])
const _SAEM_DM_M = DataModel(
    @Model(begin
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
    end),
    _SAEM_DF_M; primary_id = :ID, time_col = :t)

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
            MvNormal([μ1, μ2], LinearAlgebra.Diagonal([ω1, ω2])); column = :ID)
    end
    @formulas begin
        y ~ Normal(η[1], σ)
    end
end
const _SAEM_DIAG_DM = DataModel(_SAEM_DIAG_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.10, 0.15, 0.20, 0.25, 0.05, 0.10]);
    primary_id = :ID, time_col = :t)

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
const _SAEM_SEP_DM = DataModel(_SAEM_SEP_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]);
    primary_id = :ID, time_col = :t)
const _SAEM_SEP_DM_MISSING = DataModel(_SAEM_SEP_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = Union{Missing, Float64}[0.1, missing, 0.0, -0.1],
        y2 = Union{Missing, Float64}[missing, 0.25, 0.05, missing]);
    primary_id = :ID, time_col = :t)

function _saem_test_create_trans_pi0(eta_hmm, eta_initial)
    n_states = length(eta_initial) + 1
    @assert length(eta_hmm) == n_states * (n_states - 1)

    T = promote_type(eltype(eta_hmm), eltype(eta_initial))
    A = Matrix{T}(undef, n_states, n_states)
    pi0 = Vector{T}(undef, n_states)

    m0 = zero(T)
    @inbounds for j in 1:(n_states - 1)
        eta = eta_initial[j]
        m0 = ifelse(eta > m0, eta, m0)
    end

    denom0 = exp(-m0)
    @inbounds for s in 2:n_states
        val = exp(eta_initial[s - 1] - m0)
        pi0[s] = val
        denom0 += val
    end
    invden0 = inv(denom0)
    pi0[1] = exp(-m0) * invden0
    @inbounds for s in 2:n_states
        pi0[s] *= invden0
    end

    row_logits = Vector{T}(undef, n_states - 1)
    idx = 1
    @inbounds for i in 1:n_states
        m = zero(T)
        for k in 1:(n_states - 1)
            eta = eta_hmm[idx]
            row_logits[k] = eta
            m = ifelse(eta > m, eta, m)
            idx += 1
        end

        denom = exp(-m)
        k = 1
        for s in 1:n_states
            if s == i
                continue
            end
            val = exp(row_logits[k] - m)
            A[i, s] = val
            denom += val
            k += 1
        end

        invden = inv(denom)
        A[i, i] = exp(-m) * invden
        for s in 1:n_states
            if s == i
                continue
            end
            A[i, s] *= invden
        end
    end

    return A, pi0
end

# ── Shared HMM/LTS fixtures (built once so both testsets share the same Julia type) ──

const _SAEM_HMM2_MODEL = @Model begin
    @helpers begin
        create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(
            eta_hmm, eta_initial)
    end
    @covariates begin
        t = Covariate()
    end
    @fixedEffects begin
        mean_transitions = RealVector([0.0, 0.0])
        mean_initial = RealNumber(0.0)
        omega_hmm = RealVector([0.3, 0.3], scale = fill(:log, 2))
        omega_initial = RealNumber(0.3, scale = :log)
        eta_theta = RealVector(fill(0.6, 4), scale = fill(:logit, 4))
    end
    @randomEffects begin
        eta_hmm = RandomEffect(
            MvNormal(mean_transitions, Diagonal(omega_hmm)); column = :ID)
        eta_initial = RandomEffect(Normal(mean_initial, omega_initial); column = :ID)
    end
    @formulas begin
        trans_pi0 = create_trans_pi0(eta_hmm, [eta_initial])
        theta_mat = reshape(eta_theta, 2, 2)
        emissions = ntuple(s -> ntuple(j -> Bernoulli(theta_mat[s, j]), 2), 2)
        y ~ MVDiscreteTimeDiscreteStatesHMM(
            trans_pi0[1], emissions, Categorical(trans_pi0[2]))
    end
end
const _SAEM_HMM2_DM = DataModel(_SAEM_HMM2_MODEL,
    DataFrame(ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0],
        y = [[1, 0], [0, 1], [1, 1], [0, 0]]);
    primary_id = :ID, time_col = :t)
const _SAEM_HMM2_DM_MISSING = DataModel(_SAEM_HMM2_MODEL,
    DataFrame(ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0],
        y = Any[[1, 0], missing, [1, 1], [0, 0]]);
    primary_id = :ID, time_col = :t)

let _n_s = 7, _n_o = 9, _n_tr = 42  # 7 states × 6 transitions each
    const global _SAEM_LTS_MODEL = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(
                eta_hmm, eta_initial)
        end
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            mean_initial = RealVector(fill(0.0, _n_s - 1))
            mean_transitions = RealVector(fill(0.0, _n_tr))
            omega_hmm = RealVector(fill(0.3, _n_tr), scale = fill(:log, _n_tr))
            omega_initial = RealVector(fill(0.3, _n_s - 1), scale = fill(:log, _n_s - 1))
            eta_theta = RealVector(
                fill(0.5, _n_s * _n_o), scale = fill(:logit, _n_s * _n_o))
        end
        @randomEffects begin
            eta_hmm = RandomEffect(
                MvNormal(mean_transitions, Diagonal(omega_hmm)); column = :ID)
            eta_initial = RandomEffect(
                MvNormal(mean_initial, Diagonal(omega_initial)); column = :ID)
        end
        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, eta_initial)
            theta_mat = reshape(eta_theta, 7, 9)
            emissions = ntuple(s -> ntuple(j -> Bernoulli(theta_mat[s, j]), 9), 7)
            y ~ MVDiscreteTimeDiscreteStatesHMM(
                trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end
    const global _SAEM_LTS_DF = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0],
        y = [[1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1, 0]]
    )
    const global _SAEM_LTS_DM = DataModel(
        _SAEM_LTS_MODEL, _SAEM_LTS_DF; primary_id = :ID, time_col = :t)
end

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

    res_numeric = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            builtin_stats = :none,
            maxiters = 2,
            progress = false,
            q_store_max = 2
        ))
    @test !NoLimits.get_closed_form_mstep_used(res_numeric)
    notes_numeric = NoLimits.get_notes(res_numeric)
    @test notes_numeric.closed_form_mstep_used === false
    @test notes_numeric.closed_form_mstep_mode == :numeric_only
    @test isempty(notes_numeric.closed_form_mstep_sources)

    res_builtin = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            builtin_stats = :closed_form,
            maxiters = 2,
            progress = false,
            q_store_max = 2
        ))
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
    res_custom = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            builtin_stats = :none,
            suffstats = suffstats,
            mstep_closed_form = mstep_closed_form,
            maxiters = 2,
            progress = false,
            q_store_max = 2
        ))
    @test NoLimits.get_closed_form_mstep_used(res_custom)
    notes_custom = NoLimits.get_notes(res_custom)
    @test notes_custom.closed_form_mstep_used === true
    @test notes_custom.closed_form_mstep_mode == :closed_form_only
    @test :custom_mstep_closed_form in notes_custom.closed_form_mstep_sources

    @test_logs match_mode=:any (:info, r"SAEM: numerically optimized parameters") begin
        fit_model(dm,
            NoLimits.SAEM(;
                sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
                builtin_stats = :auto,
                maxiters = 2,
                progress = false,
                q_store_max = 2
            ))
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

    res = fit_model(dm,
        NoLimits.SAEM(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            suffstats = suffstats,
            q_from_stats = q_from_stats))
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

    res = fit_model(dm,
        NoLimits.SAEM(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            suffstats = suffstats,
            q_from_stats = q_from_stats))
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
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [1.2, 1.3, 0.8, 0.9])
    dm_ln = DataModel(_SFX_MVLN_MODEL, df_ln; primary_id = :ID, time_col = :t)
    res_ln = fit_model(dm_ln,
        NoLimits.SAEM(;
            sampler = AdaptiveNoLimitsMH(adapt_start = 2), maxiters = 3, mcmc_steps = 5, progress = false))
    @test res_ln isa FitResult
    @test isfinite(NoLimits.get_params(res_ln; scale = :untransformed).σ)

    # MvLogitNormal: samples in (0,1)^d, M-step transforms with logit
    df_lit = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.4, 0.45, 0.55, 0.5])
    dm_lit = DataModel(_SFX_MVLIT_MODEL, df_lit; primary_id = :ID, time_col = :t)
    res_lit = fit_model(dm_lit,
        NoLimits.SAEM(;
            sampler = AdaptiveNoLimitsMH(adapt_start = 2), maxiters = 3, mcmc_steps = 5, progress = false))
    @test res_lit isa FitResult
    @test isfinite(NoLimits.get_params(res_lit; scale = :untransformed).σ)
end

@testset "SAEM basic (random effects)" begin
    dm = _SAEM_DM_S
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...))
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

@testset "SAEM/MCEM serial vs threaded is reproducible" begin
    Threads.nthreads() < 2 && return

    dm = _SAEM_DM_S
    tk = (n_samples = 2, n_adapt = 2, progress = false, verbose = false)
    for (label, method) in (
        ("SAEM",
            NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, maxiters = 2,
                mcmc_steps = 1, q_store_max = 2, progress = false)),
        ("MCEM",
            NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 2,
                progress = false)))
        @testset "$label" begin
            res_serial = fit_model(
                dm, method; serialization = EnsembleSerial(), rng = MersenneTwister(123))
            res_threads = fit_model(
                dm, method; serialization = EnsembleThreads(), rng = MersenneTwister(123))
            @test res_serial.summary.objective == res_threads.summary.objective
            @test collect(NoLimits.get_params(res_serial, scale = :untransformed)) ==
                  collect(NoLimits.get_params(res_threads, scale = :untransformed))
        end
    end
end

@testset "SAEM basic with NUTS" begin
    dm = _SAEM_DM_S
    res = fit_model(dm,
        NoLimits.SAEM(; sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...))
    @test res isa FitResult
end

@testset "_half_window_test" begin
    # Flat noise: drift stays within the MC-noise floor, so the test passes.
    pass, d, s = NoLimits._half_window_test([1.0, 1.01, 0.99, 1.0], 1e-4, 1e-3)
    @test pass
    @test d ≈ 0.01
    @test s ≈ 1.005
    # Clear trend fails: drift far exceeds both tolerance and noise floor.
    pass, d, _ = NoLimits._half_window_test(collect(1.0:0.1:1.7), 1e-4, 1e-3)
    @test !pass
    @test d ≈ 0.4
    # Non-finite value: hard fail with NaN drift.
    pass, d, _ = NoLimits._half_window_test([1.0, Inf, 1.0, 1.0], 1e-4, 1e-3)
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
        [[1.0, 10.0], [1.0, 10.0], [1.0, 12.0], [1.0, 12.0]], 1e-4, 1e-3)
    @test !pass
    @test d ≈ 2.0
    @test s ≈ 10.0
    # Scale floor of 1 for small-magnitude trajectories.
    _, _, s = NoLimits._half_window_test([0.1, 0.1, 0.2, 0.2], 1e-4, 1e-3)
    @test s == 1.0
end

@testset "SAEM/MCEM convergence requires both parameter and Q stabilization" begin
    dm = _SAEM_DM_S
    tk = (n_samples = 2, n_adapt = 2, progress = false)
    @testset "SAEM" begin
        # θ drift passes as soon as the window fills (Inf tolerance) but Q never does
        # (zero tolerance): without the Q gate this would stop at iteration 4.
        res = fit_model(dm,
            NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, sa_burnin_iters = 0,
                t0 = 0, q_store_max = 2, maxiters = 8, convergence_window = 4,
                consecutive_params = 1, atol_theta = Inf, rtol_theta = Inf,
                atol_Q = 0.0, rtol_Q = 0.0))
        @test res.result.iterations == 8
        @test !NoLimits.get_converged(res)
    end
    @testset "MCEM" begin
        res = fit_model(dm,
            NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 2,
                consecutive_params = 1, atol_theta = Inf, rtol_theta = Inf,
                atol_Q = 0.0, rtol_Q = 0.0))
        # If stopping used only parameter tolerance, this would stop after 1 iteration.
        @test res.result.iterations == 2
    end
end

@testset "SAEM windowed drift test triggers early stop" begin
    # Inf tolerances make every post-window-fill check pass, so the stop point is
    # deterministic: stabilization (t0=5) + window fill (10) + consecutive (2) - 1.
    res = fit_model(_SAEM_DM_S,
        NoLimits.SAEM(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            sa_burnin_iters = 0, t0 = 5, maxiters = 100, mcmc_steps = 1,
            q_store_max = 2, convergence_window = 10, consecutive_params = 2,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf))
    @test NoLimits.get_converged(res)
    @test 16 <= res.result.iterations < 100
    diag = NoLimits.get_notes(res).diagnostics
    @test isnan(diag.drift_θ[1])  # window not yet full
    @test isfinite(diag.drift_θ[end])
    @test isfinite(diag.drift_Q[end])
end

@testset "SAEM no early stop before drift window fills" begin
    res = fit_model(_SAEM_DM_S,
        NoLimits.SAEM(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            sa_burnin_iters = 0, t0 = 0, maxiters = 6, mcmc_steps = 1,
            q_store_max = 2, convergence_window = 10, consecutive_params = 1,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf))
    @test !NoLimits.get_converged(res)
    @test res.result.iterations == 6
end

@testset "SAEM/MCEM multiple RE groups" begin
    dm = fx_mg_dm()
    tk = (n_samples = 2, n_adapt = 2, progress = false)
    for (label, method) in (
        ("SAEM", NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, SAEM_FAST...)),
        ("MCEM", NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 2)))
        @testset "$label" begin
            res = fit_model(dm, method)
            @test res isa FitResult
            re = NoLimits.get_random_effects(dm, res)
            @test !isempty(re)
        end
    end
end

@testset "SAEM constants_re" begin
    res = fit_model(_SAEM_DM_S,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...);
        constants_re = (; η = (; A = 0.0,)))
    @test res isa FitResult
end

@testset "SAEM threaded updates" begin
    dm = _SAEM_DM_M
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...);
        serialization = EnsembleThreads())
    @test res isa FitResult
end

@testset "SAEM minibatch updates" begin
    dm = _SAEM_DM_M
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            update_schedule = 1, SAEM_FAST...))
    @test res isa FitResult
end

@testset "SAEM optimizer Adam (OptimizationOptimisers)" begin
    dm = _SAEM_DM_S
    method = NoLimits.SAEM(; optimizer = OptimizationOptimisers.Adam(0.05),
        optim_kwargs = (; maxiters = 2),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        SAEM_FAST...)
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
        lb = lb, ub = ub)
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "SAEM constants for fixed effects" begin
    dm = _SAEM_DM_S
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...);
        constants = (a = 0.2,))
    @test res isa FitResult
end

@testset "SAEM RE distribution with constant covariates" begin
    # fx_recov: η ~ Normal(b * Age, 0.5) with Age a ConstantCovariate.
    res = fit_model(fx_recov_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (scalar RE)" begin
    # fx_re_model: η ~ Normal(0, ω) with ω a fixed effect → re_cov_params = (; η = :ω).
    res = fit_model(fx_re_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :ω)))
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
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :Ω)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multivariate diagonal + means)" begin
    res = fit_model(_SAEM_DIAG_DM,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = (:ω1, :ω2)),
            re_mean_params = (; η = (:μ1, :μ2))))
    @test res isa FitResult
    θ = NoLimits.get_params(res; scale = :untransformed)
end

@testset "SAEM builtin_stats uses variance for MvNormal diagonal targets" begin
    dm = _SAEM_DIAG_DM
    θ = NoLimits.get_θ0_untransformed(_SAEM_DIAG_MODEL.fixed.fixed)
    stats = (;
        re = (;
            η = (family = :mvnormal, mean = [0.0, 0.0],
                second = [0.25 0.0; 0.0 0.04], n = 10)),
        outcome = NamedTuple(),
        hmm = NamedTuple()
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
    @test isapprox(updates.ω1, 0.25; atol = 1e-12)
    @test isapprox(updates.ω2, 0.04; atol = 1e-12)
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
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :ω),
            re_mean_params = (; η = :μ)))
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
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η_id = :τ_id, η_site = :τ_site)))
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
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats gaussian_re (multiple normal outcomes, separate σ)" begin
    res = fit_model(_SAEM_SEP_DM,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            resid_var_param = (; y1 = :σ1, y2 = :σ2),
            re_cov_params = (; η = :τ)))
    @test res isa FitResult
end

@testset "SAEM builtin_stats skips missing normal outcomes (regression)" begin
    res = fit_model(_SAEM_SEP_DM_MISSING,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :closed_form,
            resid_var_param = (; y1 = :σ1, y2 = :σ2),
            re_cov_params = (; η = :τ)))
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test :builtin_stats in notes.closed_form_mstep_sources
    θ = NoLimits.get_params(res; scale = :untransformed)
end

@testset "SAEM builtin_stats gaussian_re falls back for non-Normal outcomes" begin
    # fx_bern_model: y ~ Bernoulli(logistic(a + η)), η ~ Normal(0, ω).
    res = fit_model(fx_bern_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_stats = :closed_form,
            re_cov_params = (; η = :ω)))
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
        dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    @test auto_cfg.re_mean_params == (; η = :a)
    @test auto_cfg.resid_var_param == :σ

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :auto))
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects gaussian_re (MvNormal diagonal + means)" begin
    dm = _SAEM_DIAG_DM
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(
        dm, NoLimits.get_names(_SAEM_DIAG_MODEL.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = (:ω1, :ω2))
    @test auto_cfg.re_mean_params == (; η = (:μ1, :μ2))
    @test auto_cfg.resid_var_param == :σ

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :auto))
    @test res isa FitResult
end

# NOTE: "auto detects MvNormal symbol mean with fixed diagonal expression" lives in
# estimation_saem_autodetect_tests.jl (identical model + assertions; was duplicated here).

@testset "SAEM builtin_stats auto detects LogNormal/Exponential RE + outcomes" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μη = RealNumber(0.1)
            ση = RealNumber(0.5, scale = :log)
            θη = RealNumber(1.2, scale = :log)
            μy = RealNumber(0.0)
            σy = RealNumber(0.4, scale = :log)
            θy = RealNumber(1.1, scale = :log)
        end

        @randomEffects begin
            η_ln = RandomEffect(LogNormal(μη, ση); column = :ID)
            η_exp = RandomEffect(Exponential(θη); column = :SITE)
        end

        @formulas begin
            y_ln ~ LogNormal(μy, σy)
            y_exp ~ Exponential(θy)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        SITE = [:S1, :S1, :S2, :S2],
        t = [0.0, 1.0, 0.0, 1.0],
        y_ln = [1.2, 1.1, 0.9, 1.0],
        y_exp = [0.8, 1.0, 0.7, 1.3]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(
        dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η_ln = :ση, η_exp = :θη)
    @test auto_cfg.re_mean_params == (; η_ln = :μη)
    @test auto_cfg.resid_var_param == (; y_ln = (; μ = :μy, σ = :σy), y_exp = :θy)

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_stats = :auto))
    @test res isa FitResult
end

@testset "SAEM builtin_stats auto detects Bernoulli/Poisson outcome params when direct symbols" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p = RealNumber(0.6)
            λ = RealNumber(1.5)
            τ = RealNumber(0.4, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            yb ~ Bernoulli(p)
            yp ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        yb = [0, 1, 1, 0],
        yp = [1, 2, 0, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(
        dm, NoLimits.get_names(model.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.resid_var_param == (; yb = :p, yp = :λ)
end

@testset "SAEM builtin_stats classifies HMM closed-form eligibility" begin
    dm_partial = _SAEM_HMM2_DM
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(
        dm_partial, NoLimits.get_names(_SAEM_HMM2_MODEL.fixed.fixed))
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; eta_hmm = :omega_hmm, eta_initial = :omega_initial)
    @test auto_cfg.re_mean_params ==
          (; eta_hmm = :mean_transitions, eta_initial = :mean_initial)
    @test auto_cfg.resid_var_param == NamedTuple()
    @test auto_cfg.hmm_emission_params.y.target == :eta_theta
    @test auto_cfg.hmm_outcomes == (:y,)
    @test auto_cfg.hmm_transition_closed_form == :covered_by_re_or_external
    @test auto_cfg.hmm_emission_closed_form == :eligible

    elig_partial = NoLimits._saem_builtin_closed_form_eligibility(
        dm_partial,
        auto_cfg.re_cov_params,
        auto_cfg.re_mean_params,
        auto_cfg.resid_var_param,
        auto_cfg.hmm_emission_params
    )
    @test elig_partial.re_block_eligible
    @test !elig_partial.outcome_block_eligible
    @test elig_partial.hmm_emission_block_eligible
    @test elig_partial.has_any_closed_form_block
    @test elig_partial.hmm_outcomes == (:y,)
    @test elig_partial.hmm_transition_closed_form == :covered_by_re_or_external
    @test elig_partial.hmm_emission_closed_form == :eligible
    @test !(:hmm_latent_state_suffstats_not_available_builtin in elig_partial.reasons)

    res_partial = fit_model(dm_partial,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            mcmc_steps = 1,
            q_store_max = 2,
            maxiters = 2,
            progress = false,
            builtin_stats = :auto);
        store_eb_modes = false)
    notes_partial = NoLimits.get_notes(res_partial)
    @test notes_partial.builtin_stats_mode_effective == :closed_form
    @test notes_partial.builtin_stats_closed_form_eligibility.re_block_eligible
    @test !notes_partial.builtin_stats_closed_form_eligibility.outcome_block_eligible
    @test notes_partial.builtin_stats_closed_form_eligibility.hmm_emission_block_eligible
    @test notes_partial.builtin_stats_closed_form_eligibility.hmm_outcomes == (:y,)
    @test NoLimits.get_closed_form_mstep_used(res_partial)
    @test notes_partial.closed_form_mstep_mode == :closed_form_only
    @test :builtin_stats in notes_partial.closed_form_mstep_sources

    model_ineligible = @Model begin
        @helpers begin
            create_trans_pi0(eta_hmm, eta_initial) = _saem_test_create_trans_pi0(
                eta_hmm, eta_initial)
        end

        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            eta_theta = RealVector(fill(0.6, 4), scale = fill(:logit, 4))
        end

        @randomEffects begin
            eta_hmm = RandomEffect(MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])); column = :ID)
            eta_initial = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            trans_pi0 = create_trans_pi0(eta_hmm, [eta_initial])
            theta_mat = reshape(eta_theta, 2, 2)
            emissions = ntuple(
                s -> ntuple(j -> Bernoulli(clamp(0.8 * theta_mat[s, j] + 0.1, 0.0, 1.0)), 2),
                2)
            y ~ MVDiscreteTimeDiscreteStatesHMM(
                trans_pi0[1], emissions, Categorical(trans_pi0[2]))
        end
    end

    dm_ineligible = DataModel(
        model_ineligible, get_df(_SAEM_HMM2_DM); primary_id = :ID, time_col = :t)
    auto_cfg_none = NoLimits._saem_autodetect_gaussian_re(
        dm_ineligible, NoLimits.get_names(model_ineligible.fixed.fixed))
    @test auto_cfg_none === nothing

    res_ineligible = fit_model(dm_ineligible,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            mcmc_steps = 1,
            q_store_max = 2,
            maxiters = 2,
            progress = false,
            builtin_stats = :auto);
        store_eb_modes = false)
    notes_ineligible = NoLimits.get_notes(res_ineligible)
    @test notes_ineligible.builtin_stats_mode_effective == :none
    @test !notes_ineligible.builtin_stats_closed_form_eligibility.has_any_closed_form_block
    @test notes_ineligible.builtin_stats_closed_form_eligibility.hmm_outcomes == (:y,)
    @test !NoLimits.get_closed_form_mstep_used(res_ineligible)
    @test notes_ineligible.closed_form_mstep_mode == :numeric_only
end

function _saem_exact_discrete_hmm_gamma(dists, ys)
    K = dists[1].n_states
    n_time = length(dists)
    gamma = zeros(Float64, K, n_time)
    total = 0.0
    first_probs = probabilities_hidden_states(dists[1])
    ranges = ntuple(_ -> 1:K, n_time)

    for path in Iterators.product(ranges...)
        w = first_probs[path[1]]
        for t in 1:n_time
            y = ys[t]
            if !ismissing(y)
                w *= pdf(dists[t].emission_dists[path[t]], y)
            end
            if t < n_time
                w *= dists[t + 1].transition_matrix[path[t], path[t + 1]]
            end
        end
        total += w
        for t in 1:n_time
            gamma[path[t], t] += w
        end
    end

    gamma ./= total
    return gamma
end

@testset "SAEM discrete HMM emission stats use recursive sequence posteriors" begin
    P = [0.85 0.15; 0.25 0.75]
    init = Categorical([0.6, 0.4])
    dists = [
        DiscreteTimeDiscreteStatesHMM(P, (Bernoulli(0.9), Bernoulli(0.2)), init),
        DiscreteTimeDiscreteStatesHMM(P, (Bernoulli(0.9), Bernoulli(0.2)), init),
        DiscreteTimeDiscreteStatesHMM(P, (Bernoulli(0.9), Bernoulli(0.2)), init)
    ]
    ys = Union{Missing, Int}[1, missing, 0]

    gamma, ok = NoLimits._saem_hmm_smoothed_gamma(dists, ys, Float64)
    expected = _saem_exact_discrete_hmm_gamma(dists, ys)

    @test ok
    @test isapprox(gamma, expected; atol = 1e-12)
    @test !isapprox(gamma[:, 1], posterior_hidden_states(dists[1], 1); atol = 1e-6)
end

@testset "SAEM builtin_stats HMM emission handles fully missing rows (regression)" begin
    res = fit_model(_SAEM_HMM2_DM_MISSING,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            mcmc_steps = 1,
            q_store_max = 2,
            maxiters = 2,
            progress = false,
            builtin_stats = :auto);
        store_eb_modes = false)
    @test res isa FitResult
    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test :builtin_stats in notes.closed_form_mstep_sources
end

@testset "SAEM builtin_stats auto detects lts_random_no_cv full closed-form coverage" begin
    dm = _SAEM_LTS_DM
    fixed_names = NoLimits.get_names(_SAEM_LTS_MODEL.fixed.fixed)
    auto_cfg = NoLimits._saem_autodetect_gaussian_re(dm, fixed_names)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; eta_hmm = :omega_hmm, eta_initial = :omega_initial)
    @test auto_cfg.re_mean_params ==
          (; eta_hmm = :mean_transitions, eta_initial = :mean_initial)
    @test auto_cfg.resid_var_param == NamedTuple()
    @test auto_cfg.hmm_emission_params.y.target == :eta_theta
    @test auto_cfg.hmm_outcomes == (:y,)
    @test auto_cfg.hmm_transition_closed_form == :covered_by_re_or_external
    @test auto_cfg.hmm_emission_closed_form == :eligible

    elig = NoLimits._saem_builtin_closed_form_eligibility(
        dm,
        auto_cfg.re_cov_params,
        auto_cfg.re_mean_params,
        auto_cfg.resid_var_param,
        auto_cfg.hmm_emission_params
    )
    @test elig.re_block_eligible
    @test !elig.outcome_block_eligible
    @test elig.hmm_emission_block_eligible
    @test elig.has_any_closed_form_block
    @test elig.hmm_outcomes == (:y,)
    @test elig.hmm_transition_closed_form == :covered_by_re_or_external
    @test elig.hmm_emission_closed_form == :eligible

    closed_form_targets = Symbol[]
    for v in values(auto_cfg.re_cov_params)
        NoLimits._saem_collect_target_symbols!(closed_form_targets, v)
    end
    for v in values(auto_cfg.re_mean_params)
        NoLimits._saem_collect_target_symbols!(closed_form_targets, v)
    end
    for col in keys(auto_cfg.hmm_emission_params)
        info = getfield(auto_cfg.hmm_emission_params, col)
        if hasproperty(info, :target) && info.target isa Symbol
            push!(closed_form_targets, info.target)
        end
    end
    @test Set(closed_form_targets) == Set(fixed_names)
end

@testset "SAEM builtin_stats runs lts_random_no_cv style model in closed-form-only mode" begin
    res = fit_model(_SAEM_LTS_DM,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            mcmc_steps = 1,
            q_store_max = 2,
            maxiters = 2,
            progress = false,
            builtin_stats = :auto);
        store_eb_modes = false)

    notes = NoLimits.get_notes(res)
    @test notes.builtin_stats_mode_effective == :closed_form
    @test NoLimits.get_closed_form_mstep_used(res)
    @test notes.closed_form_mstep_mode == :closed_form_only
    @test :builtin_stats in notes.closed_form_mstep_sources
end

@testset "SAEM builtin_mean glm (Bernoulli + Poisson)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            τ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y1 ~ Bernoulli(p)
            λ = exp(b + η)
            y2 ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0, 1, 1, 0],
        y2 = [1, 2, 0, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_mean = :glm))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm (Normal + Exponential)" begin
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
            y2 ~ Exponential(exp(b + η))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.8, 1.2, 0.5, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_mean = :glm))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm (ODE Normal)" begin
    res = fit_model(fx_ode_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            q_store_max = 2,
            maxiters = 2,
            builtin_mean = :glm))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm + builtin_stats (Normal)" begin
    res = fit_model(fx_re_dm(),
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_mean = :glm,
            builtin_stats = :closed_form,
            resid_var_param = :σ,
            re_cov_params = (; η = :ω)))
    @test res isa FitResult
end

@testset "SAEM builtin_mean glm + builtin_stats (Normal outcomes, separate σ)" begin
    res = fit_model(_SAEM_SEP_DM,
        NoLimits.SAEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            SAEM_FAST...,
            builtin_mean = :glm,
            builtin_stats = :closed_form,
            resid_var_param = (; y1 = :σ1, y2 = :σ2),
            re_cov_params = (; η = :τ)))
    @test res isa FitResult
end

@testset "SAEM/MCEM threaded helper cache preserves ODE options" begin
    # fx_ode_model already has saveat_mode = :saveat in its solver config.
    dm = fx_ode_dm()
    ll_cache = build_ll_cache(dm; ode_kwargs = (abstol = 1e-8, reltol = 1e-7))
    for (label, thread_caches) in (("SAEM", NoLimits._saem_thread_caches),
        ("MCEM", NoLimits._mcem_thread_caches))
        @testset "$label" begin
            threaded = thread_caches(dm, ll_cache, 2)
            @test length(threaded) == 2
            @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
            @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
        end
    end
end

@testset "SAEM/MCEM thread RNGs are reproducible from passed rng" begin
    for (label, thread_rngs) in (("SAEM", NoLimits._saem_thread_rngs),
        ("MCEM", NoLimits._mcem_thread_rngs))
        @testset "$label" begin
            r1 = thread_rngs(MersenneTwister(42), 3)
            r2 = thread_rngs(MersenneTwister(42), 3)
            r3 = thread_rngs(MersenneTwister(99), 3)
            s1 = [rand(r, Float64) for r in r1]
            s2 = [rand(r, Float64) for r in r2]
            s3 = [rand(r, Float64) for r in r3]
            @test s1 == s2
            @test s1 != s3
        end
    end
end

@testset "SAEM/MCEM final EBE rescue options are configurable" begin
    rescue_kwargs = (; ebe_rescue_on_high_grad = false, ebe_rescue_multistart_n = 2,
        ebe_rescue_multistart_k = 2, ebe_rescue_max_rounds = 7,
        ebe_rescue_grad_tol = 1e-5)
    saem = NoLimits.SAEM(; rescue_kwargs...)
    mcem = NoLimits.MCEM(; rescue_kwargs...)
    for (label, rescue) in (("SAEM", saem.saem.ebe_rescue), ("MCEM", mcem.ebe_rescue))
        @testset "$label" begin
            @test rescue.enabled == false
            @test rescue.multistart_n == 2
            @test rescue.multistart_k == 2
            @test rescue.max_rounds == 7
            @test rescue.grad_tol == 1e-5
        end
    end
end

@testset "SAEM/MCEM get_random_effects recomputes EB modes with rescue options" begin
    tk = (n_samples = 2, n_adapt = 2, progress = false)
    rescue_kwargs = (; ebe_rescue_on_high_grad = true, ebe_rescue_multistart_n = 2,
        ebe_rescue_multistart_k = 2, ebe_rescue_max_rounds = 2,
        ebe_rescue_grad_tol = 1e-7)
    for (label, method) in (
        ("SAEM",
            NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, q_store_max = 2,
                maxiters = 2, mcmc_steps = 1, t0 = 1, rescue_kwargs...)),
        ("MCEM",
            NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 2,
                rescue_kwargs...)))
        @testset "$label" begin
            res = fit_model(_SAEM_DM_S, method; store_eb_modes = false)
            re = NoLimits.get_random_effects(_SAEM_DM_S, res)
            @test haskey(re, :η)
            @test nrow(re.η) == 2  # 2 unique IDs in _SAEM_DM_S
        end
    end
end

@testset "SAEM/MCEM constants_re: fixed-RE individuals contribute observations to Q" begin
    # When constants_re pins one individual's RE, the other individual's observations must
    # still influence the M-step objective: the fit completes with a finite objective and
    # logf for the constant-RE batch (empty b) stays finite.
    tk = (n_samples = 3, n_adapt = 2, progress = false)
    for (label, method, rng) in (
        ("SAEM",
            NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, maxiters = 3,
                t0 = 1, kappa = 0.6, mcmc_steps = 1, q_store_max = 2),
            Xoshiro(42)),
        ("MCEM",
            NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 3),
            Xoshiro(7)))
        @testset "$label" begin
            # Fit with RE for :A pinned to 0.0 — :B should still contribute its observations.
            res = fit_model(
                _SAEM_DM_S, method; constants_re = (; η = (; A = 0.0,)), rng = rng)
            @test res isa NoLimits.FitResult
            @test isfinite(NoLimits.get_objective(res))

            # Manually evaluate logf for the constant-RE batch with an empty b — must be finite.
            θu = NoLimits.get_params(res; scale = :untransformed)
            _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(
                _SAEM_DM_S, (; η = (; A = 0.0,)))
            ll_cache = NoLimits.build_ll_cache(_SAEM_DM_S)
            for (bi, info) in enumerate(batch_infos)
                if info.n_b == 0
                    logf = NoLimits._laplace_logf_batch(
                        _SAEM_DM_S, info, θu, Float64[], const_cache, ll_cache)
                    @test isfinite(logf)
                end
            end
        end
    end
end

@testset "SAEM/MCEM constants_re: all RE constant — fit runs and objective is finite" begin
    # When all RE levels are constant the entire model reduces to a fixed-effects evaluation
    # on the M-step.  The fit should complete without error and yield a finite objective.
    tk = (n_samples = 3, n_adapt = 2, progress = false)
    for (label, method, rng) in (
        ("SAEM",
            NoLimits.SAEM(; sampler = MH(), turing_kwargs = tk, maxiters = 3,
                t0 = 1, kappa = 0.6, mcmc_steps = 1, q_store_max = 2),
            Xoshiro(1)),
        ("MCEM",
            NoLimits.MCEM(; sampler = MH(), turing_kwargs = tk, maxiters = 3),
            Xoshiro(8)))
        @testset "$label" begin
            res = fit_model(
                _SAEM_DM_S, method; constants_re = (; η = (; A = 0.0, B = 0.0)), rng = rng)
            @test res isa NoLimits.FitResult
            @test isfinite(NoLimits.get_objective(res))
        end
    end
end
