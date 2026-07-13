using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using ComponentArrays
using LinearAlgebra
using Random

# ── unit tests ────────────────────────────────────────────────────────────────

@testset "_saem_build_var_lb_target_set: Normal RE auto-detect" begin
    re_cov_params = (; η = :σ_η)
    re_family_map = (; η = :normal)
    resid_var_param = :σ
    θ0_u = ComponentArray(σ_η = 1.0, σ = 0.5, a = 0.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :σ_η in targets
    @test :σ in targets
    @test :a ∉ targets
end

@testset "_saem_build_var_lb_target_set: LogNormal RE auto-detect" begin
    re_cov_params = (; η = :ω)
    re_family_map = (; η = :lognormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(ω = 0.3, a = 1.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :ω in targets
    @test :a ∉ targets
end

@testset "_saem_build_var_lb_target_set: MvNormal RE auto-detect" begin
    # MvNormal cov param is a matrix — should be skipped
    re_cov_params = (; η = :Ω)
    re_family_map = (; η = :mvnormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(Ω = [1.0 0.0; 0.0 1.0])

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test isempty(targets)  # matrix params skipped
end

@testset "_saem_build_var_lb_target_set: MvNormal with scalar SD" begin
    # When the MvNormal cov param is a scalar (diagonal case), it should be included
    re_cov_params = (; η = :σ_η)
    re_family_map = (; η = :mvnormal)
    resid_var_param = nothing
    θ0_u = ComponentArray(σ_η = 1.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test :σ_η in targets
end

@testset "_saem_build_var_lb_target_set: unsupported family excluded" begin
    re_cov_params = (; η = :α)
    re_family_map = (; η = :beta)
    resid_var_param = nothing
    θ0_u = ComponentArray(α = 2.0)

    targets = NoLimits._saem_build_var_lb_target_set(
        re_cov_params, re_family_map, resid_var_param, θ0_u)
    @test isempty(targets)
end

@testset "_saem_build_var_lb_target_set: NamedTuple resid_var_param" begin
    re_cov_params = NamedTuple()
    re_family_map = NamedTuple()
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
    opts = NoLimits.SAEM(auto_var_lb = false, var_lb_value = 1e-6)
    @test opts.saem.auto_var_lb == false
    @test opts.saem.var_lb_value == 1e-6
end

# ── integration tests ─────────────────────────────────────────────────────────

# Helper: build a minimal DataModel with Normal RE
function _make_normal_re_dm()
    m = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ_η = RealNumber(0.5; scale = :log)
            σ = RealNumber(0.3; scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = repeat(1:5, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 5),
        y = 0.5 .+ 0.1 .* randn(20)
    )
    DataModel(m, df; primary_id = :ID, time_col = :t)
end

@testset "var lb integration: Normal RE — lb prevents collapse" begin
    dm = _make_normal_re_dm()
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2,
            auto_var_lb = true,
            var_lb_value = 1e-5
        ))
    θ = NoLimits.get_params(res; scale = :untransformed)
    # σ_η and σ must stay ≥ 1e-5 after clamping
    @test Float64(θ.σ_η) >= 1e-5
    @test Float64(θ.σ) >= 1e-5
end

@testset "var lb integration: auto_var_lb=false — no floor enforced" begin
    # Just checks the run completes without error; no assertion on parameter magnitude
    dm = _make_normal_re_dm()
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2,
            auto_var_lb = false
        ))
    @test NoLimits.get_objective(res) !== nothing
end

@testset "var lb integration: LogNormal RE — lb active on ω" begin
    m = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            ω = RealNumber(0.4; scale = :log)
            σ = RealNumber(0.3; scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, ω); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = repeat(1:4, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 4),
        y = 0.8 .+ 0.1 .* randn(16)
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2,
            auto_var_lb = true,
            var_lb_value = 1e-5
        ))
    θ = NoLimits.get_params(res; scale = :untransformed)
    @test Float64(θ.ω) >= 1e-5
    @test Float64(θ.σ) >= 1e-5
end

@testset "var lb integration: min rule — anneal_min_sd < var_lb_value" begin
    # When anneal_min_sd < var_lb_value, effective_var_lb = anneal_min_sd.
    # anneal_to_fixed requires a literal SD in the RE distribution.
    m = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ_η = RealNumber(0.5; scale = :log)
            σ = RealNumber(0.3; scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = repeat(1:5, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 5),
        y = 0.5 .+ 0.1 .* randn(20)
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2,
            anneal_to_fixed = (:η,),
            anneal_min_sd = 1e-6,
            auto_var_lb = true,
            var_lb_value = 1e-5
        ))
    @test NoLimits.get_objective(res) !== nothing
end

@testset "var lb prevents collapse without explicit lower= bound (regression)" begin
    # Identical observations → true inter-individual RE variance ≈ 0. Without the fix,
    # the closed-form M-step writes σ = 0 into iter_constants → log(0) = -Inf in
    # θ_const_t → Q = Inf and the algorithm is stuck. With the fix, auto_var_lb clamps
    # iter_constants before θ_const_t is built, so σ stays ≥ var_lb_value.
    m = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            ω = RealNumber(0.4; scale = :log)    # deliberately NO lower= bound
            σ = RealNumber(0.3; scale = :log)    # deliberately NO lower= bound
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, ω); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = repeat(1:8, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 8),
        y = fill(0.5, 32)   # identical observations → RE variance collapses to 0 immediately
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 3, n_adapt = 2, progress = false),
            maxiters = 20,
            t0 = 5,
            auto_var_lb = true,
            var_lb_value = 1e-5
        ); rng = Random.Xoshiro(0))
    θ = NoLimits.get_params(res; scale = :untransformed)
    # exp(log(1e-5)) < 1e-5 by 2 ulps: allow the log-scale transform round-trip.
    @test Float64(θ.ω) >= 1e-5 * (1 - 1e-12)
    @test Float64(θ.σ) >= 1e-5 * (1 - 1e-12)
    @test isfinite(NoLimits.get_objective(res))
end

# MvNormal-diagonal RE model shared by the two anneal_to_fixed testsets below.
const _VARLB_MVDIAG_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2)
        σ = RealNumber(0.3; scale = :log)
        omega = RealDiagonalMatrix([0.5, 0.5]; scale = :log)
    end
    @randomEffects begin
        η = RandomEffect(MvNormal([0.0, 0.0], Diagonal(omega)); column = :ID)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + η[1] + η[2], σ)
    end
end

@testset "anneal_to_fixed: MvNormal RE — sd0 from initial distribution" begin
    df = DataFrame(
        ID = repeat(1:5, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 5),
        y = 0.5 .+ 0.1 .* randn(20)
    )
    dm = DataModel(_VARLB_MVDIAG_MODEL, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2,
            anneal_to_fixed = (:η,),
            anneal_min_sd = 1e-5
        ))
end

@testset "anneal_to_fixed: MvNormal RE — all schedules accepted" begin
    df = DataFrame(
        ID = repeat(1:4, inner = 3),
        t = repeat([0.0, 1.0, 2.0], 4),
        y = 0.5 .+ 0.1 .* randn(12)
    )
    dm = DataModel(_VARLB_MVDIAG_MODEL, df; primary_id = :ID, time_col = :t)
    for sched in (:exponential, :linear, :gamma)
        res = fit_model(dm,
            NoLimits.SAEM(;
                sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
                maxiters = 2,
                anneal_to_fixed = (:η,),
                anneal_schedule = sched,
                anneal_min_sd = 1e-5
            ))
    end
end

# ── anneal_to_fixed EBE tests (merged from saem_anneal_ebe_tests.jl) ──────────
# anneal_to_fixed REs must be pinned to their distribution mean in the final EBE
# step, rather than being freely estimated per individual.

@testset "anneal_to_fixed EBE: scalar Normal — values equal distribution mean" begin
    m = @Model begin
        @fixedEffects begin
            mu_a = RealNumber(1.0)
            mu_b = RealNumber(0.5)
            sigma = RealNumber(0.3; scale = :log)
        end
        @randomEffects begin
            a = RandomEffect(Normal(mu_a, 1.0); column = :ID)
            b = RandomEffect(Normal(mu_b, 1.0); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + b * t, sigma)
        end
    end

    df = DataFrame(
        ID = repeat(1:8, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 8),
        y = 1.0 .+ 0.5 .* repeat([0.0, 1.0, 2.0, 3.0], 8) .+
            0.1 .* randn(MersenneTwister(1), 32)
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 3, n_adapt = 2, progress = false),
            maxiters = 10,
            anneal_to_fixed = (:a, :b)
        ))

    re = NoLimits.get_random_effects(res)
    θu = NoLimits.get_params(res; scale = :untransformed)

    # All individuals should have the same value for annealed REs
    @test length(unique(re.a.a_1)) == 1
    @test length(unique(re.b.b_1)) == 1

    # That value should equal the distribution mean = mu_a / mu_b at estimated θu
    @test re.a.a_1[1]≈θu.mu_a atol=1e-8
    @test re.b.b_1[1]≈θu.mu_b atol=1e-8
end

@testset "anneal_to_fixed EBE: non-annealed RE still varies freely" begin
    m = @Model begin
        @fixedEffects begin
            mu_a = RealNumber(1.0)
            sigma = RealNumber(0.3; scale = :log)
        end
        @randomEffects begin
            a = RandomEffect(Normal(mu_a, 1.0); column = :ID)
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η * t, sigma)
        end
    end

    df = DataFrame(
        ID = repeat(1:6, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 6),
        y = 1.0 .+ 0.3 .* repeat([0.0, 1.0, 2.0, 3.0], 6) .+
            0.1 .* randn(MersenneTwister(42), 24)
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 3, n_adapt = 2, progress = false),
            maxiters = 10,
            anneal_to_fixed = (:a,)
        ))

    re = NoLimits.get_random_effects(res)
    θu = NoLimits.get_params(res; scale = :untransformed)

    # Annealed RE: all identical and equal to distribution mean
    @test length(unique(re.a.a_1)) == 1
    @test re.a.a_1[1]≈θu.mu_a atol=1e-8

    # Non-annealed RE: column present for all individuals
    @test :η_1 in propertynames(re.η)
    @test nrow(re.η) == 6
end

@testset "anneal_to_fixed EBE: MvNormal RE — value equals distribution mean" begin
    m = @Model begin
        @fixedEffects begin
            mu = RealNumber(0.5)
            sigma = RealNumber(0.3; scale = :log)
            omega = RealDiagonalMatrix([0.5, 0.5]; scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([mu, mu], Diagonal(omega)); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(η[1] + η[2] * t, sigma)
        end
    end

    df = DataFrame(
        ID = repeat(1:5, inner = 4),
        t = repeat([0.0, 1.0, 2.0, 3.0], 5),
        y = 0.5 .+ 0.5 .* repeat([0.0, 1.0, 2.0, 3.0], 5) .+
            0.1 .* randn(MersenneTwister(7), 20)
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 3, n_adapt = 2, progress = false),
            maxiters = 10,
            anneal_to_fixed = (:η,)
        ))

    re = NoLimits.get_random_effects(res)
    θu = NoLimits.get_params(res; scale = :untransformed)

    # All individuals should have the same η values
    @test length(unique(re.η.η_1)) == 1
    @test length(unique(re.η.η_2)) == 1

    # Values equal the MvNormal mean [mu, mu]
    @test re.η.η_1[1]≈θu.mu atol=1e-8
    @test re.η.η_2[1]≈θu.mu atol=1e-8
end

@testset "anneal_to_fixed EBE: notes stores anneal_to_fixed for serialization" begin
    m = @Model begin
        @fixedEffects begin
            mu_a = RealNumber(0.3)
            sigma = RealNumber(0.3; scale = :log)
        end
        @randomEffects begin
            a = RandomEffect(Normal(mu_a, 1.0); column = :ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, sigma)
        end
    end

    df = DataFrame(
        ID = repeat(1:4, inner = 3),
        t = repeat([0.0, 1.0, 2.0], 4),
        y = 0.3 .+ 0.1 .* randn(MersenneTwister(5), 12)
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)

    res = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 5,
            anneal_to_fixed = (:a,)
        ))

    notes = NoLimits.get_notes(res)
    @test hasproperty(notes, :anneal_to_fixed)
    @test :a in notes.anneal_to_fixed
end
