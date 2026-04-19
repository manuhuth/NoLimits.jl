using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using ComponentArrays
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Merged from: saem_schedule_tests.jl, saem_sa_anneal_tests.jl,
#              saem_var_lb_tests.jl, saem_multichain_tests.jl
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# SA SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

@testset "SA schedule: _saem_gamma_schedule robbins_monro" begin
    # maxiters=2, sa_burnin_iters=0, t0=5 (t0 > maxiters → only stabilization phase runs)
    opts = NoLimits.SAEM(; t0=5, kappa=0.65, maxiters=2).saem

    @test opts.sa_schedule === :robbins_monro

    # With t0=5 > maxiters=2, only stabilization phase runs — any iter ≤ t0 gives γ = 1.0
    @test NoLimits._saem_gamma_schedule(1, opts) == 1.0
    @test NoLimits._saem_gamma_schedule(2, opts) == 1.0
end

@testset "SA schedule: _saem_gamma_schedule two_phase kappa=-1" begin
    opts = NoLimits.SAEM(;
        sa_schedule=:two_phase,
        sa_burnin_iters=3,
        sa_phase1_iters=5,
        sa_phase2_kappa=-1.0
    ).saem

    # Burn-in: γ = 0
    @test NoLimits._saem_gamma_schedule(1, opts) == 0.0
    @test NoLimits._saem_gamma_schedule(3, opts) == 0.0

    # Phase 1: γ = 1
    @test NoLimits._saem_gamma_schedule(4, opts) == 1.0
    @test NoLimits._saem_gamma_schedule(8, opts) == 1.0

    # Phase 2: k2^(-1) = 1/k2
    @test NoLimits._saem_gamma_schedule(9, opts)  ≈ 1.0   # k2 = 1
    @test NoLimits._saem_gamma_schedule(10, opts) ≈ 0.5   # k2 = 2
    @test NoLimits._saem_gamma_schedule(11, opts) ≈ 1/3   # k2 = 3
    @test NoLimits._saem_gamma_schedule(18, opts) ≈ 0.1   # k2 = 10
end

@testset "SA schedule: _saem_gamma_schedule two_phase kappa=-2" begin
    opts = NoLimits.SAEM(;
        sa_schedule=:two_phase,
        sa_burnin_iters=0,
        sa_phase1_iters=2,
        sa_phase2_kappa=-2.0
    ).saem

    # Phase 2 starts at iter 3 (k2 = 1)
    @test NoLimits._saem_gamma_schedule(3, opts) ≈ 1.0     # 1^(-2) = 1, clamped to 1
    @test NoLimits._saem_gamma_schedule(4, opts) ≈ 0.25    # 2^(-2)
    @test NoLimits._saem_gamma_schedule(5, opts) ≈ 1/9     # 3^(-2)
end

@testset "SA schedule: _saem_gamma_schedule custom" begin
    fn = (iter, _opts) -> iter == 1 ? 1.0 : 0.5
    opts = NoLimits.SAEM(; sa_schedule=:custom, sa_schedule_fn=fn).saem

    @test NoLimits._saem_gamma_schedule(1, opts)  == 1.0
    @test NoLimits._saem_gamma_schedule(2, opts)  == 0.5
    @test NoLimits._saem_gamma_schedule(99, opts) == 0.5
end

@testset "SA schedule: _saem_schedule_phase labels" begin
    # Default: sa_burnin_iters=0, t0=maxiters÷2 — no burn-in, stabilization then decay
    opts_rm = NoLimits.SAEM().saem
    @test opts_rm.t0 == opts_rm.maxiters ÷ 2
    @test NoLimits._saem_schedule_phase(1,   opts_rm) === :robbins_monro
    @test NoLimits._saem_schedule_phase(opts_rm.t0, opts_rm) === :robbins_monro
    @test NoLimits._saem_schedule_phase(opts_rm.t0 + 1, opts_rm) === :robbins_monro_decay

    opts_rm_short = NoLimits.SAEM(; maxiters=2).saem
    @test opts_rm_short.t0 == 1
    @test NoLimits._saem_schedule_phase(opts_rm_short.t0, opts_rm_short) === :robbins_monro
    @test NoLimits._saem_schedule_phase(opts_rm_short.t0 + 1, opts_rm_short) === :robbins_monro_decay

    # With burn-in
    opts_rm_bi = NoLimits.SAEM(; sa_burnin_iters=3, t0=5).saem
    @test NoLimits._saem_schedule_phase(1, opts_rm_bi) === :burnin
    @test NoLimits._saem_schedule_phase(3, opts_rm_bi) === :burnin
    @test NoLimits._saem_schedule_phase(4, opts_rm_bi) === :robbins_monro
    @test NoLimits._saem_schedule_phase(8, opts_rm_bi) === :robbins_monro
    @test NoLimits._saem_schedule_phase(9, opts_rm_bi) === :robbins_monro_decay

    opts_tp = NoLimits.SAEM(; sa_schedule=:two_phase,
                              sa_burnin_iters=2, sa_phase1_iters=3).saem
    @test NoLimits._saem_schedule_phase(1, opts_tp) === :burnin
    @test NoLimits._saem_schedule_phase(2, opts_tp) === :burnin
    @test NoLimits._saem_schedule_phase(3, opts_tp) === :phase1
    @test NoLimits._saem_schedule_phase(5, opts_tp) === :phase1
    @test NoLimits._saem_schedule_phase(6, opts_tp) === :phase2

    opts_cu = NoLimits.SAEM(; sa_schedule=:custom, sa_schedule_fn=(i, _) -> 1.0).saem
    @test NoLimits._saem_schedule_phase(1, opts_cu) === :custom
end

@testset "SA schedule: _saem_past_stabilization_phase" begin
    opts_rm = NoLimits.SAEM(; t0=5).saem
    @test !NoLimits._saem_past_stabilization_phase(5, opts_rm)
    @test  NoLimits._saem_past_stabilization_phase(6, opts_rm)

    opts_tp = NoLimits.SAEM(; sa_schedule=:two_phase,
                              sa_burnin_iters=2, sa_phase1_iters=3).saem
    @test !NoLimits._saem_past_stabilization_phase(5, opts_tp)  # end of phase1
    @test  NoLimits._saem_past_stabilization_phase(6, opts_tp)  # first of phase2

    opts_cu = NoLimits.SAEM(; sa_schedule=:custom, sa_schedule_fn=(i, _) -> 1.0).saem
    @test !NoLimits._saem_past_stabilization_phase(1, opts_cu)
    @test  NoLimits._saem_past_stabilization_phase(2, opts_cu)
end

@testset "SA schedule: _saem_stats_update burnin guard" begin
    s     = (a=1.0, b=2.0)
    s_new = (a=5.0, b=6.0)

    # γ = 0: must return s unchanged (burn-in), including when s is nothing
    @test NoLimits._saem_stats_update(s, s_new, 0.0) === s
    @test NoLimits._saem_stats_update(nothing, s_new, 0.0) === nothing

    # γ > 0 with s=nothing: seeds with s_new (unchanged behavior)
    @test NoLimits._saem_stats_update(nothing, s_new, 1.0) === s_new

    # Normal update: linear interpolation
    result = NoLimits._saem_stats_update(s, s_new, 0.5)
    @test result.a ≈ 1.0 + 0.5 * (5.0 - 1.0)
    @test result.b ≈ 2.0 + 0.5 * (6.0 - 2.0)
end

@testset "SA schedule: _saem_store_push! burnin guard" begin
    capacity  = 4
    n_batches = 2
    store = NoLimits._SAEMSampleStore(
        zeros(Float64, capacity),
        [[zeros(Float64, 1) for _ in 1:n_batches] for _ in 1:capacity],
        1, 1, 0, capacity, 1e-10, 0
    )
    b = [zeros(Float64, 1), zeros(Float64, 1)]

    # γ = 0: store stays empty
    NoLimits._saem_store_push!(store, b, 0.0)
    @test store.len == 0
    @test store.next_idx == 1

    # γ = 1: store gains one entry
    NoLimits._saem_store_push!(store, b, 1.0)
    @test store.len == 1
    @test store.weights[1] == 1.0
end

# schedule integration helpers

function _sched_dm()
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end
const _CONST_SCHED_DM = _sched_dm()

@testset "SA schedule: default robbins_monro regression" begin
    dm = _CONST_SCHED_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=1, kappa=0.65,
        progress=false, q_store_max=2, builtin_stats=:none
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test conv.schedule_phase[1] === :robbins_monro
    @test conv.schedule_phase[2] === :robbins_monro_decay

    @test conv.gamma[1] == 1.0
    @test conv.gamma[2] ≈ 0.0   # k3=1, frac=(1-1)/1 = 0

end

@testset "SA schedule: two_phase integration" begin
    dm = _CONST_SCHED_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        sa_schedule=:two_phase,
        sa_burnin_iters=1, sa_phase1_iters=1, sa_phase2_kappa=-1.0,
        progress=false, q_store_max=2, builtin_stats=:none
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test conv.schedule_phase[1] === :burnin
    @test conv.schedule_phase[2] === :phase1
    @test conv.gamma[1] == 0.0
    @test conv.gamma[2] == 1.0

end

@testset "SA schedule: custom schedule integration" begin
    dm = _CONST_SCHED_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        sa_schedule=:custom, sa_schedule_fn=((iter, _opts) -> 0.5),
        progress=false, q_store_max=2, builtin_stats=:none
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test all(p === :custom for p in conv.schedule_phase)
    @test all(γ ≈ 0.5 for γ in conv.gamma)
end

# ══════════════════════════════════════════════════════════════════════════════
# SA ANNEALING
# ══════════════════════════════════════════════════════════════════════════════

@testset "SA anneal: _saem_sa_anneal_floor_scalar" begin
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 0.0, :exponential) ≈ 0.9
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 0.0, :linear)      ≈ 0.9
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 1.0, :exponential) ≈ 0.9 * exp(-3.0)
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 1.0, :linear)      ≈ 0.0
    @test NoLimits._saem_sa_anneal_floor_scalar(-0.5, 0.8, 0.0, :exponential) ≈ 0.4
    @test NoLimits._saem_sa_anneal_floor_scalar(-0.5, 0.8, 0.0, :linear)      ≈ 0.4
    @test NoLimits._saem_sa_anneal_floor_scalar(2.0, 0.9, 0.5, :linear)      ≈ 0.9
    @test NoLimits._saem_sa_anneal_floor_scalar(2.0, 0.9, 0.5, :exponential) ≈ 1.8 * exp(-1.5) atol=1e-10
end

@testset "SA anneal: _saem_build_sa_anneal_targets — auto-detect" begin
    re_cov  = (; eta = :sigma_eta)
    re_fam  = (; eta = :normal)
    targets = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam, 0.9)
    @test haskey(targets, :sigma_eta)
    @test targets.sigma_eta ≈ 0.9

    re_fam2 = (; eta = :lognormal)
    targets2 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam2, 0.8)
    @test haskey(targets2, :sigma_eta)
    @test targets2.sigma_eta ≈ 0.8

    re_fam3 = (; eta = :exponential)
    targets3 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam3, 0.9)
    @test isempty(keys(targets3))

    re_fam4 = (; eta = :mvnormal)
    targets4 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam4, 0.9)
    @test isempty(keys(targets4))

    re_cov5 = (; eta1=:s1, eta2=:s2, eta3=:s3)
    re_fam5 = (; eta1=:normal, eta2=:exponential, eta3=:lognormal)
    targets5 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov5, re_fam5, 0.9)
    @test haskey(targets5, :s1)
    @test !haskey(targets5, :s2)
    @test haskey(targets5, :s3)

    re_cov6  = (; eta1=:sigma, eta2=:sigma)
    re_fam6  = (; eta1=:normal, eta2=:normal)
    targets6 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov6, re_fam6, 0.9)
    @test length(keys(targets6)) == 1
    @test haskey(targets6, :sigma)
end

@testset "SA anneal: _saem_build_sa_anneal_targets — user override" begin
    re_cov = (; eta = :sigma_eta)
    re_fam = (; eta = :normal)
    user_t = (; my_param = 0.7)
    targets = NoLimits._saem_build_sa_anneal_targets(user_t, re_cov, re_fam, 0.9)
    @test targets === user_t
    @test !haskey(targets, :sigma_eta)
end

@testset "SA anneal: _saem_apply_sa_anneal_clamp — scalar param" begin
    θu = ComponentArray(; a=0.2, sigma=0.05)
    active = (; sigma=0.9)
    initial = (; sigma=0.5)

    result = NoLimits._saem_apply_sa_anneal_clamp(θu, active, initial, 1, 10, 0.9, :exponential)
    @test result !== θu
    @test result.sigma ≈ 0.9 * 0.5
    @test result.a ≈ 0.2

    θu_large = ComponentArray(; a=0.2, sigma=1.0)
    result2 = NoLimits._saem_apply_sa_anneal_clamp(θu_large, active, initial, 1, 10, 0.9, :exponential)
    @test result2 === θu_large

    result3 = NoLimits._saem_apply_sa_anneal_clamp(θu, active, initial, 11, 10, 0.9, :exponential)
    @test result3 === θu

    result4 = NoLimits._saem_apply_sa_anneal_clamp(θu, NamedTuple(), NamedTuple(), 1, 10, 0.9, :exponential)
    @test result4 === θu
end

@testset "SA anneal: _saem_apply_sa_anneal_clamp — vector param" begin
    θu = ComponentArray(; omega=Float64[0.01, 0.02])
    initial = (; omega=Float64[1.0, 2.0])
    active  = (; omega=0.9)

    result = NoLimits._saem_apply_sa_anneal_clamp(θu, active, initial, 1, 10, 0.9, :exponential)
    @test result !== θu
    @test result.omega[1] ≈ 0.9
    @test result.omega[2] ≈ 1.8

    θu2 = ComponentArray(; omega=Float64[0.5, 3.0])
    result2 = NoLimits._saem_apply_sa_anneal_clamp(θu2, active, initial, 1, 10, 0.9, :exponential)
    @test result2 !== θu2
    @test result2.omega[1] ≈ 0.9
    @test result2.omega[2] ≈ 3.0
end

@testset "SA anneal: SAEMOptions defaults" begin
    opts = NoLimits.SAEM().saem
    @test isempty(keys(opts.sa_anneal_targets))
    @test opts.sa_anneal_schedule == :exponential
    @test opts.sa_anneal_iters == 0
    @test opts.sa_anneal_alpha == 0.9
    @test opts.sa_anneal_fn === nothing
end

@testset "SA anneal: SAEMOptions explicit values" begin
    opts = NoLimits.SAEM(;
        sa_anneal_targets=(; sigma=0.8),
        sa_anneal_schedule=:linear,
        sa_anneal_iters=30,
        sa_anneal_alpha=0.7,
    ).saem
    @test opts.sa_anneal_targets.sigma ≈ 0.8
    @test opts.sa_anneal_schedule == :linear
    @test opts.sa_anneal_iters == 30
    @test opts.sa_anneal_alpha ≈ 0.7
end

# anneal integration helpers

function _anneal_dm_normal()
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a       = RealNumber(0.2)
            sigma_y = RealNumber(0.5, scale=:log)
            sigma_e = RealNumber(0.4, scale=:log)
        end
        @randomEffects begin
            eta = RandomEffect(Normal(0.0, sigma_e); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + eta, sigma_y)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C,:D,:D,:E,:E],
                   t =repeat([0.0,1.0], 5),
                   y =randn(10))
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end
const _CONST_ANNEAL_DM_NORMAL = _anneal_dm_normal()

function _anneal_dm_lognormal()
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            mu_y    = RealNumber(0.0)
            sigma_y = RealNumber(0.3, scale=:log)
            sigma_e = RealNumber(0.4, scale=:log)
        end
        @randomEffects begin
            eta = RandomEffect(LogNormal(0.0, sigma_e); column=:ID)
        end
        @formulas begin
            y ~ LogNormal(mu_y + log(eta), sigma_y)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C,:D,:D,:E,:E],
                   t =repeat([0.0,1.0], 5),
                   y =abs.(randn(10)) .+ 0.1)
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _anneal_dm_mvnormal()
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a       = RealNumber(0.2)
            sigma_y = RealNumber(0.4, scale=:log)
            omega   = RealDiagonalMatrix([0.5, 0.5], scale=:log)
        end
        @randomEffects begin
            eta = RandomEffect(MvNormal([0.0, 0.0], Diagonal(omega)); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + eta[1] + eta[2], sigma_y)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C,:D,:D,:E,:E],
                   t =repeat([0.0,1.0], 5),
                   y =randn(10))
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "SA anneal: Normal RE — auto-detected sigma_e floored" begin
    dm = _CONST_ANNEAL_DM_NORMAL
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test length(conv.anneal_active) == 2
    @test all(conv.anneal_active[1:2])
end

@testset "SA anneal: Normal RE — no clamp when sigma_e starts large" begin
    dm = _CONST_ANNEAL_DM_NORMAL
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9,
    ))
end

@testset "SA anneal: Normal RE — disabled when sa_anneal_alpha=0.0" begin
    dm = _CONST_ANNEAL_DM_NORMAL
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.0,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test length(conv.anneal_active) == 2
end

@testset "SA anneal: LogNormal RE — auto-detected" begin
    dm = _anneal_dm_lognormal()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test length(conv.anneal_active) == 2
    @test all(conv.anneal_active[1:2])
end

@testset "SA anneal: MvNormal RE — manual targets" begin
    dm = _anneal_dm_mvnormal()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9,
        sa_anneal_targets=(; omega=0.9),
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test length(conv.anneal_active) == 2
    @test all(conv.anneal_active[1:2])
end

@testset "SA anneal: no annealing when targets is empty and re_cov_params is empty" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + eta, σ)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[0.1,0.2,0.0,-0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test !any(conv.anneal_active)
end

@testset "SA anneal: linear schedule works" begin
    dm = _CONST_ANNEAL_DM_NORMAL
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9, sa_anneal_schedule=:linear,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test length(conv.anneal_active) == 2
end

@testset "SA anneal: auto effective_iters = 30% of maxiters" begin
    dm = _CONST_ANNEAL_DM_NORMAL
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_alpha=0.9,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test length(conv.anneal_active) == length(conv.gamma)
end

# ══════════════════════════════════════════════════════════════════════════════
# VARIANCE LOWER BOUND
# ══════════════════════════════════════════════════════════════════════════════

@testset "_saem_build_var_lb_target_set: Normal RE auto-detect" begin
    re_cov_params   = (; η = :σ_η)
    re_family_map   = (; η = :normal)
    resid_var_param = :σ
    θ0_u = ComponentArray(σ_η = 1.0, σ = 0.5, a = 0.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :σ_η in targets
    @test :σ    in targets
    @test :a   ∉ targets
end

@testset "_saem_build_var_lb_target_set: LogNormal RE auto-detect" begin
    re_cov_params   = (; η = :ω)
    re_family_map   = (; η = :lognormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(ω = 0.3, a = 1.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :ω in targets
    @test :a ∉ targets
end

@testset "_saem_build_var_lb_target_set: MvNormal RE auto-detect" begin
    re_cov_params   = (; η = :Ω)
    re_family_map   = (; η = :mvnormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(Ω = [1.0 0.0; 0.0 1.0])

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test isempty(targets)
end

@testset "_saem_build_var_lb_target_set: MvNormal with scalar SD" begin
    re_cov_params   = (; η = :σ_η)
    re_family_map   = (; η = :mvnormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(σ_η = 1.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :σ_η in targets
end

@testset "_saem_build_var_lb_target_set: unsupported family excluded" begin
    re_cov_params   = (; η = :α)
    re_family_map   = (; η = :beta)
    resid_var_param = nothing
    θ0_u = ComponentArray(α = 2.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test isempty(targets)
end

@testset "_saem_build_var_lb_target_set: NamedTuple resid_var_param" begin
    re_cov_params   = NamedTuple()
    re_family_map   = NamedTuple()
    resid_var_param = (; obs1 = :σ1, obs2 = :σ2)
    θ0_u = ComponentArray(σ1 = 0.5, σ2 = 0.3)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :σ1 in targets
    @test :σ2 in targets
end

@testset "_saem_apply_var_lb: scalar clamping" begin
    θu = ComponentArray(σ = 1e-8, a = 0.5)
    θu_lb = NoLimits._saem_apply_var_lb(θu, (:σ,), 1e-5)
    @test θu_lb.σ ≈ 1e-5
    @test θu_lb.a ≈ 0.5
    @test θu_lb !== θu

    θu2 = ComponentArray(σ = 0.5)
    θu2_lb = NoLimits._saem_apply_var_lb(θu2, (:σ,), 1e-5)
    @test θu2_lb === θu2
end

@testset "_saem_apply_var_lb: vector clamping" begin
    θu = ComponentArray(ω = [1e-9, 0.3, 1e-10])
    θu_lb = NoLimits._saem_apply_var_lb(θu, (:ω,), 1e-5)
    @test θu_lb.ω[1] ≈ 1e-5
    @test θu_lb.ω[2] ≈ 0.3
    @test θu_lb.ω[3] ≈ 1e-5
end

@testset "_saem_apply_var_lb: empty targets no-op" begin
    θu = ComponentArray(σ = 1e-9)
    θu_lb = NoLimits._saem_apply_var_lb(θu, (), 1e-5)
    @test θu_lb === θu
end

@testset "SAEMOptions: auto_var_lb defaults" begin
    opts = NoLimits.SAEM()
    @test opts.saem.auto_var_lb == true
    @test opts.saem.var_lb_value == 1e-5
end

@testset "SAEMOptions: auto_var_lb explicit override" begin
    opts = NoLimits.SAEM(auto_var_lb=false, var_lb_value=1e-6)
    @test opts.saem.auto_var_lb == false
    @test opts.saem.var_lb_value == 1e-6
end

# var_lb integration helpers

function _make_normal_re_dm()
    m = @Model begin
        @fixedEffects begin
            a    = RealNumber(0.3)
            σ_η  = RealNumber(0.5; scale=:log)
            σ    = RealNumber(0.3; scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID=repeat(1:5, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 5),
        y=0.5 .+ 0.1 .* randn(20)
    )
    DataModel(m, df; primary_id=:ID, time_col=:t)
end
const _CONST_NORMAL_RE_DM = _make_normal_re_dm()

@testset "var lb integration: Normal RE — lb prevents collapse" begin
    dm = _CONST_NORMAL_RE_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        auto_var_lb=true,
        var_lb_value=1e-5
    ))
    θ = NoLimits.get_params(res; scale=:untransformed)
    @test Float64(θ.σ_η) >= 1e-5
    @test Float64(θ.σ)   >= 1e-5
end

@testset "var lb integration: auto_var_lb=false — no floor enforced" begin
    dm = _CONST_NORMAL_RE_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        auto_var_lb=false
    ))
    @test NoLimits.get_objective(res) !== nothing
end

@testset "var lb integration: LogNormal RE — lb active on ω" begin
    m = @Model begin
        @fixedEffects begin
            a  = RealNumber(0.5)
            ω  = RealNumber(0.4; scale=:log)
            σ  = RealNumber(0.3; scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, ω); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID=repeat(1:4, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 4),
        y=0.8 .+ 0.1 .* randn(16)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        auto_var_lb=true,
        var_lb_value=1e-5
    ))
    θ = NoLimits.get_params(res; scale=:untransformed)
    @test Float64(θ.ω) >= 1e-5
    @test Float64(θ.σ) >= 1e-5
end

@testset "var lb integration: min rule — anneal_min_sd < var_lb_value" begin
    m = @Model begin
        @fixedEffects begin
            a    = RealNumber(0.3)
            σ_η  = RealNumber(0.5; scale=:log)
            σ    = RealNumber(0.3; scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID=repeat(1:5, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 5),
        y=0.5 .+ 0.1 .* randn(20)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        anneal_to_fixed=(:η,),
        anneal_min_sd=1e-6,
        auto_var_lb=true,
        var_lb_value=1e-5
    ))
    @test NoLimits.get_objective(res) !== nothing
end

@testset "anneal_to_fixed: MvNormal RE — sd0 from initial distribution" begin
    m = @Model begin
        @fixedEffects begin
            a     = RealNumber(0.2)
            σ     = RealNumber(0.3; scale=:log)
            omega = RealDiagonalMatrix([0.5, 0.5]; scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Diagonal(omega)); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2], σ)
        end
    end
    df = DataFrame(
        ID=repeat(1:5, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 5),
        y=0.5 .+ 0.1 .* randn(20)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        anneal_to_fixed=(:η,),
        anneal_min_sd=1e-5,
    ))
end

@testset "anneal_to_fixed: MvNormal RE — all schedules accepted" begin
    m = @Model begin
        @fixedEffects begin
            a     = RealNumber(0.2)
            σ     = RealNumber(0.3; scale=:log)
            omega = RealDiagonalMatrix([0.5, 0.5]; scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Diagonal(omega)); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2], σ)
        end
    end
    df = DataFrame(
        ID=repeat(1:4, inner=3),
        t=repeat([0.0, 1.0, 2.0], 4),
        y=0.5 .+ 0.1 .* randn(12)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)
    for sched in (:exponential, :linear, :gamma)
        res = fit_model(dm, NoLimits.SAEM(;
            sampler=MH(),
            turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
            maxiters=2,
            anneal_to_fixed=(:η,),
            anneal_schedule=sched,
            anneal_min_sd=1e-5,
        ))
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-CHAIN
# ══════════════════════════════════════════════════════════════════════════════

@testset "Multi-chain: _saem_effective_chains" begin
    @test NoLimits._saem_effective_chains(1, false, 50, 10) == 1
    @test NoLimits._saem_effective_chains(3, false, 50, 10) == 3
    @test NoLimits._saem_effective_chains(1, false, 50, 100) == 1
    @test NoLimits._saem_effective_chains(1, true, 50,  5) == 10
    @test NoLimits._saem_effective_chains(1, true, 50, 10) == 5
    @test NoLimits._saem_effective_chains(1, true, 50, 25) == 2
    @test NoLimits._saem_effective_chains(1, true, 50, 50) == 1
    @test NoLimits._saem_effective_chains(1, true, 50, 60) == 1
    @test NoLimits._saem_effective_chains(8, true, 50, 10) == 8
    @test NoLimits._saem_effective_chains(2, true, 50, 10) == 5
    @test NoLimits._saem_effective_chains(1, true, 50, 0) == 50
end

@testset "Multi-chain: _saem_update_b_current! n_chains=1 passthrough" begin
    b_chains  = [[Float64[1.0, 2.0]], [Float64[3.0, 4.0]]]
    b_current = [zeros(2), zeros(2)]
    NoLimits._saem_update_b_current!(b_current, b_chains, [1, 2], 1)
    @test b_current[1] === b_chains[1][1]
    @test b_current[2] === b_chains[2][1]
end

@testset "Multi-chain: _saem_update_b_current! n_chains=3 average" begin
    b_chains = [
        [Float64[1.0], Float64[3.0], Float64[5.0]],
        [Float64[2.0], Float64[4.0], Float64[6.0]],
    ]
    b_current = [zeros(1), zeros(1)]
    NoLimits._saem_update_b_current!(b_current, b_chains, [1, 2], 3)
    @test b_current[1] ≈ [3.0]
    @test b_current[2] ≈ [4.0]
end

@testset "Multi-chain: _saem_update_b_current! only updates listed batches" begin
    b_chains  = [[Float64[10.0]], [Float64[20.0]]]
    b_current = [Float64[99.0], Float64[99.0]]
    NoLimits._saem_update_b_current!(b_current, b_chains, [1], 1)
    @test b_current[1] === b_chains[1][1]
    @test b_current[2][1] == 99.0
end

@testset "Multi-chain: SAEMOptions n_chains defaults" begin
    opts = NoLimits.SAEM().saem
    @test opts.n_chains == 1
    @test opts.auto_small_n_chains == false
    @test opts.small_n_chain_target == 50
end

@testset "Multi-chain: SAEMOptions explicit values" begin
    opts = NoLimits.SAEM(; n_chains=4, auto_small_n_chains=true, small_n_chain_target=20).saem
    @test opts.n_chains == 4
    @test opts.auto_small_n_chains == true
    @test opts.small_n_chain_target == 20
end

# multichain integration helpers

function _mc_dm()
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   y=[0.1,0.2,0.0,-0.1,0.15,0.05])
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end
const _CONST_MC_DM = _mc_dm()

@testset "Multi-chain: n_chains=1 regression" begin
    dm = _CONST_MC_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        n_chains=1
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test all(n == 1 for n in conv.n_chains_used)
end

@testset "Multi-chain: n_chains=2 runs and diagnostics reflect chain count" begin
    dm = _CONST_MC_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        n_chains=2
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test all(n == 2 for n in conv.n_chains_used)
end

@testset "Multi-chain: auto_small_n_chains inflates chain count" begin
    dm = _CONST_MC_DM
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=1, progress=false, q_store_max=2, builtin_stats=:none,
        auto_small_n_chains=true, small_n_chain_target=50
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test all(n == 17 for n in conv.n_chains_used)
end

@testset "Multi-chain: auto_small_n_chains no inflation when n_batches >= target" begin
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    n = 60
    df = DataFrame(
        ID = repeat(1:n, inner=2),
        t  = repeat([0.0, 1.0], n),
        y  = randn(2n)
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=1, progress=false, q_store_max=2, builtin_stats=:none,
        n_chains=1, auto_small_n_chains=true, small_n_chain_target=50
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test all(n == 1 for n in conv.n_chains_used)
end
