using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using ComponentArrays
using LinearAlgebra

# ── unit tests ────────────────────────────────────────────────────────────────

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
    # MvNormal cov param is a matrix — should be skipped
    re_cov_params   = (; η = :Ω)
    re_family_map   = (; η = :mvnormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(Ω = [1.0 0.0; 0.0 1.0])

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test isempty(targets)  # matrix params skipped
end

@testset "_saem_build_var_lb_target_set: MvNormal with scalar SD" begin
    # When the MvNormal cov param is a scalar (diagonal case), it should be included
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
    @test θu_lb.a ≈ 0.5  # untouched
    @test θu_lb !== θu   # new object returned

    # No clamping needed
    θu2 = ComponentArray(σ = 0.5)
    θu2_lb = NoLimits._saem_apply_var_lb(θu2, (:σ,), 1e-5)
    @test θu2_lb === θu2  # same object — no copy
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

# ── integration tests ─────────────────────────────────────────────────────────

# Helper: build a minimal DataModel with Normal RE
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

@testset "var lb integration: Normal RE — lb prevents collapse" begin
    dm = _make_normal_re_dm()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2,
        auto_var_lb=true,
        var_lb_value=1e-5
    ))
    θ = NoLimits.get_params(res; scale=:untransformed)
    # σ_η and σ must stay ≥ 1e-5 after clamping
    @test Float64(θ.σ_η) >= 1e-5
    @test Float64(θ.σ)   >= 1e-5
end

@testset "var lb integration: auto_var_lb=false — no floor enforced" begin
    # Just checks the run completes without error; no assertion on parameter magnitude
    dm = _make_normal_re_dm()
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
    # When anneal_min_sd < var_lb_value, effective_var_lb = anneal_min_sd.
    # anneal_to_fixed requires a literal SD in the RE distribution.
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
