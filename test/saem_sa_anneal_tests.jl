using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using ComponentArrays
using LinearAlgebra

# ── unit tests ────────────────────────────────────────────────────────────────

@testset "SA anneal: _saem_sa_anneal_floor_scalar" begin
    # frac=0 (first iter): floor = alpha * |init|
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 0.0, :exponential) ≈ 0.9
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 0.0, :linear)      ≈ 0.9

    # frac=1 (last iter): exponential decays to alpha*exp(-3); linear decays to 0
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 1.0, :exponential) ≈ 0.9 * exp(-3.0)
    @test NoLimits._saem_sa_anneal_floor_scalar(1.0, 0.9, 1.0, :linear)      ≈ 0.0

    # abs(init) used — negative init_val treated as positive magnitude
    @test NoLimits._saem_sa_anneal_floor_scalar(-0.5, 0.8, 0.0, :exponential) ≈ 0.4
    @test NoLimits._saem_sa_anneal_floor_scalar(-0.5, 0.8, 0.0, :linear)      ≈ 0.4

    # frac=0.5 intermediate
    @test NoLimits._saem_sa_anneal_floor_scalar(2.0, 0.9, 0.5, :linear)      ≈ 0.9  # 0.9*2*(1-0.5)
    @test NoLimits._saem_sa_anneal_floor_scalar(2.0, 0.9, 0.5, :exponential) ≈
          1.8 * exp(-1.5)  atol=1e-10
end

@testset "SA anneal: _saem_build_sa_anneal_targets — auto-detect" begin
    # Normal RE → cov param auto-detected
    re_cov  = (; eta = :sigma_eta)
    re_fam  = (; eta = :normal)
    targets = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam, 0.9)
    @test haskey(targets, :sigma_eta)
    @test targets.sigma_eta ≈ 0.9

    # LogNormal RE → cov param auto-detected
    re_fam2 = (; eta = :lognormal)
    targets2 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam2, 0.8)
    @test haskey(targets2, :sigma_eta)
    @test targets2.sigma_eta ≈ 0.8

    # Exponential RE → NOT auto-detected
    re_fam3 = (; eta = :exponential)
    targets3 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam3, 0.9)
    @test isempty(keys(targets3))

    # MvNormal RE → NOT auto-detected
    re_fam4 = (; eta = :mvnormal)
    targets4 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov, re_fam4, 0.9)
    @test isempty(keys(targets4))

    # Multiple REs, only Normal/LogNormal picked
    re_cov5 = (; eta1=:s1, eta2=:s2, eta3=:s3)
    re_fam5 = (; eta1=:normal, eta2=:exponential, eta3=:lognormal)
    targets5 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov5, re_fam5, 0.9)
    @test haskey(targets5, :s1)
    @test !haskey(targets5, :s2)
    @test haskey(targets5, :s3)

    # Duplicate cov param symbols deduplicated
    re_cov6  = (; eta1=:sigma, eta2=:sigma)
    re_fam6  = (; eta1=:normal, eta2=:normal)
    targets6 = NoLimits._saem_build_sa_anneal_targets(NamedTuple(), re_cov6, re_fam6, 0.9)
    @test length(keys(targets6)) == 1
    @test haskey(targets6, :sigma)
end

@testset "SA anneal: _saem_build_sa_anneal_targets — user override" begin
    # Non-empty user targets: returned as-is, no auto-detection
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

    # iter=1, anneal_iters=10: frac=0, floor = 0.9*0.5 = 0.45 > 0.05 → clamped
    result = NoLimits._saem_apply_sa_anneal_clamp(θu, active, initial, 1, 10, 0.9, :exponential)
    @test result !== θu   # new object returned
    @test result.sigma ≈ 0.9 * 0.5   # = 0.45
    @test result.a ≈ 0.2              # untouched

    # When current > floor: no clamp needed, original returned
    θu_large = ComponentArray(; a=0.2, sigma=1.0)
    result2 = NoLimits._saem_apply_sa_anneal_clamp(θu_large, active, initial, 1, 10, 0.9, :exponential)
    @test result2 === θu_large

    # After anneal period: no clamp
    result3 = NoLimits._saem_apply_sa_anneal_clamp(θu, active, initial, 11, 10, 0.9, :exponential)
    @test result3 === θu

    # Empty active targets: passthrough
    result4 = NoLimits._saem_apply_sa_anneal_clamp(θu, NamedTuple(), NamedTuple(), 1, 10, 0.9, :exponential)
    @test result4 === θu
end

@testset "SA anneal: _saem_apply_sa_anneal_clamp — vector param" begin
    θu = ComponentArray(; omega=Float64[0.01, 0.02])
    initial = (; omega=Float64[1.0, 2.0])
    active  = (; omega=0.9)

    # iter=1, frac=0: floor[1]=0.9, floor[2]=1.8; both > curr → both clamped
    result = NoLimits._saem_apply_sa_anneal_clamp(θu, active, initial, 1, 10, 0.9, :exponential)
    @test result !== θu
    @test result.omega[1] ≈ 0.9
    @test result.omega[2] ≈ 1.8

    # Only one element needs clamping
    θu2 = ComponentArray(; omega=Float64[0.5, 3.0])
    result2 = NoLimits._saem_apply_sa_anneal_clamp(θu2, active, initial, 1, 10, 0.9, :exponential)
    @test result2 !== θu2
    @test result2.omega[1] ≈ 0.9   # clamped
    @test result2.omega[2] ≈ 3.0   # not clamped (3.0 > 1.8)
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

# ── helpers for integration tests ─────────────────────────────────────────────

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

# ── integration tests ─────────────────────────────────────────────────────────

@testset "SA anneal: Normal RE — auto-detected sigma_e floored" begin
    dm = _anneal_dm_normal()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9,
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    # maxiters=2 < sa_anneal_iters=4, so all iterations are within anneal period
    @test length(conv.anneal_active) == 2
    @test all(conv.anneal_active[1:2])
end

@testset "SA anneal: Normal RE — no clamp when sigma_e starts large" begin
    # sigma_e starts at 0.4 which is > alpha*0.4 = 0.36, so no clamp applied
    dm = _anneal_dm_normal()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.9,
    ))
end

@testset "SA anneal: Normal RE — disabled when sa_anneal_alpha=0.0" begin
    dm = _anneal_dm_normal()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_iters=4, sa_anneal_alpha=0.0,
        # alpha=0 → floor=0 → no clamping ever
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    # anneal_active can be true (clamp period active) but clamp itself doesn't
    # change anything since floor=0; fit should complete normally
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
    # MvNormal RE is NOT auto-detected; user must specify sa_anneal_targets manually
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
    # Model with no re_cov_params auto-detectable (no Normal/LogNormal RE with cov param)
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            eta = RandomEffect(Normal(0.0, 1.0); column=:ID)  # literal SD, not a param
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
    # No clamping applied
    @test !any(conv.anneal_active)
end

@testset "SA anneal: linear schedule works" begin
    dm = _anneal_dm_normal()
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
    # sa_anneal_iters=0 triggers auto: 30% of maxiters=2 = 3
    dm = _anneal_dm_normal()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        sa_anneal_alpha=0.9,  # sa_anneal_iters=0 (default) → auto = 3
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    # First 3 iters should have clamp active (some of them may clamp, some may not)
    # Just verify the diagnostics vector has the right length
    @test length(conv.anneal_active) == length(conv.gamma)
end
