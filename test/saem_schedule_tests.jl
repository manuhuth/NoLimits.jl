using Test
using NoLimits
using DataFrames
using Distributions
using Turing

# ── unit tests for schedule helpers ──────────────────────────────────────────

@testset "SA schedule: _saem_gamma_schedule robbins_monro" begin
    # maxiters=20, sa_burnin_iters=0, t0=5 → phase3_total=15
    opts = NoLimits.SAEM(; t0=5, kappa=0.65, maxiters=20).saem

    @test opts.sa_schedule === :robbins_monro

    # Burn-in phase (sa_burnin_iters=0, so no burn-in iterations here)
    # Stabilization: iter 1..5 → γ = 1.0
    @test NoLimits._saem_gamma_schedule(1, opts) == 1.0
    @test NoLimits._saem_gamma_schedule(5, opts) == 1.0

    # Decay phase: iter 6..20, phase3_total=15, k3 = iter - 5
    # γ = ((15 - k3) / 15)^0.65
    @test NoLimits._saem_gamma_schedule(6,  opts) ≈ (14.0/15.0)^0.65   # k3=1
    @test NoLimits._saem_gamma_schedule(11, opts) ≈ ( 9.0/15.0)^0.65   # k3=6
    @test NoLimits._saem_gamma_schedule(20, opts) ≈  0.0                # k3=15

    # γ = 0 exactly at maxiters
    @test NoLimits._saem_gamma_schedule(opts.maxiters, opts) == 0.0
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
    @test NoLimits._saem_schedule_phase(1,   opts_rm) === :robbins_monro        # stabilization
    @test NoLimits._saem_schedule_phase(opts_rm.t0, opts_rm) === :robbins_monro        # last stabilization
    @test NoLimits._saem_schedule_phase(opts_rm.t0 + 1, opts_rm) === :robbins_monro_decay  # first decay

    opts_rm_short = NoLimits.SAEM(; maxiters=10).saem
    @test opts_rm_short.t0 == 5
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

# ── integration tests ─────────────────────────────────────────────────────────

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

@testset "SA schedule: default robbins_monro regression" begin
    # maxiters=5, t0=2, sa_burnin_iters=0 → stabilization iter 1-2, decay iter 3-5
    # phase3_total = 5 - 0 - 2 = 3; γ at iter k: ((3 - (k-2)) / 3)^0.65
    dm = _sched_dm()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
        maxiters=5, t0=2, kappa=0.65,
        progress=false, q_store_max=4, builtin_stats=:none
    ))
    conv   = NoLimits.get_diagnostics(res).convergence
    method = NoLimits.get_method(res)

    # Phase labels
    @test conv.schedule_phase[1] === :robbins_monro
    @test conv.schedule_phase[2] === :robbins_monro
    @test conv.schedule_phase[3] === :robbins_monro_decay
    @test conv.schedule_phase[5] === :robbins_monro_decay

    # γ values
    @test conv.gamma[1] == 1.0
    @test conv.gamma[2] == 1.0
    @test conv.gamma[3] ≈ (2.0/3.0)^0.65   # k3=1, frac=(3-1)/3
    @test conv.gamma[4] ≈ (1.0/3.0)^0.65   # k3=2, frac=(3-2)/3
    @test conv.gamma[5] ≈  0.0              # k3=3, frac=(3-3)/3 = 0

    @test isfinite(NoLimits.get_objective(res))
end

@testset "SA schedule: two_phase integration" begin
    dm = _sched_dm()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
        maxiters=8,
        sa_schedule=:two_phase,
        sa_burnin_iters=2, sa_phase1_iters=3, sa_phase2_kappa=-1.0,
        progress=false, q_store_max=4, builtin_stats=:none
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test conv.schedule_phase[1] === :burnin
    @test conv.schedule_phase[2] === :burnin
    @test conv.schedule_phase[3] === :phase1
    @test conv.schedule_phase[4] === :phase1
    @test conv.schedule_phase[5] === :phase1
    @test conv.schedule_phase[6] === :phase2

    # γ values
    @test conv.gamma[1] == 0.0
    @test conv.gamma[2] == 0.0
    @test conv.gamma[3] == 1.0
    @test conv.gamma[4] == 1.0
    @test conv.gamma[5] == 1.0
    @test conv.gamma[6] ≈ 1.0    # k2=1 → 1^(-1) = 1
    length(conv.gamma) >= 7 && @test conv.gamma[7] ≈ 0.5  # k2=2

    @test isfinite(NoLimits.get_objective(res))
end

@testset "SA schedule: custom schedule integration" begin
    dm = _sched_dm()
    constant_fn = (iter, _opts) -> 0.5
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=3, n_adapt=0, progress=false),
        maxiters=4,
        sa_schedule=:custom, sa_schedule_fn=constant_fn,
        progress=false, q_store_max=4, builtin_stats=:none
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test all(p === :custom for p in conv.schedule_phase)
    @test all(γ ≈ 0.5 for γ in conv.gamma)
    @test isfinite(NoLimits.get_objective(res))
end
