# Loaded by enzyme_smoke_tests.jl only when the opt-in gate passes.
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using Random
using Lux
using OrdinaryDiffEq
using Enzyme

Enzyme.API.strictAliasing!(false)

@testset "Enzyme smoke: joint logf gradients (SoftTree model, fwd + rev)" begin
    # SoftTree model with η feeding the tree input — exercises the positional
    # parameter rebuild, BLAS-free eval, transforms, and the laplace batch path.
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.4, scale = :log)
            ω = RealNumber(0.5, scale = :log)
            Γ = SoftTreeParameters(
                2, 2; function_name = :ST1, seed = 0, calculate_se = false)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            μ = ST1([x.Age, η], Γ)[1]
            y ~ Normal(μ, σ)
        end
    end
    rng = Xoshiro(1)
    df = DataFrame(
        ID = repeat(1:4, inner = 3),
        t = repeat(collect(0.0:1.0:2.0), outer = 4),
        Age = repeat([0.3, -0.2, 0.1, 0.5], inner = 3),
        y = randn(rng, 12) .* 0.3 .+ 0.5
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NoLimits.build_ll_cache(dm)
    info = batch_infos[1]
    θu = get_θ0_untransformed(dm.model.fixed.fixed)
    b = randn(Xoshiro(42), max(info.n_b, 1)) .* 0.2

    f = let dm = dm, info = info, θu = θu, const_cache = const_cache, ll_cache = ll_cache
        bv -> NoLimits._laplace_logf_batch_impl(dm, info, θu, bv, const_cache, ll_cache)
    end
    g_fd = ForwardDiff.gradient(f, b)
    @test all(isfinite, g_fd)

    g_fwd = collect(Enzyme.gradient(
        set_runtime_activity(Enzyme.Forward), Const(f), copy(b))[1])
    @test isapprox(g_fwd, g_fd; rtol = 1e-6, atol = 1e-10)

    g_rev = collect(Enzyme.gradient(
        set_runtime_activity(Enzyme.Reverse), Const(f), copy(b))[1])
    @test isapprox(g_rev, g_fd; rtol = 1e-6, atol = 1e-10)
end

@testset "Enzyme smoke: diagonal closed-form ODE marginal loglik (fwd + rev)" begin
    # First real-Enzyme validation of the ODE path: a diagonal linear ODE solved in
    # closed form has NO solver/adjoint, so Enzyme differentiates pure arithmetic and
    # matches ForwardDiff to ~machine precision. (:linear/hybrid closed-form and the
    # numerical solve are NOT covered here — see the closed-form Enzyme notes.)
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(0.8, scale = :log)
            a = RealNumber(1.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -k * x1
        end
        @initialDE begin
            x1 = a
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
    model = set_solver_config(model; saveat_mode = :saveat, closed_form = :auto)
    df = DataFrame(ID = repeat(1:2, inner = 3), t = repeat([0.0, 0.5, 1.0], outer = 2),
        y = abs.(randn(Xoshiro(1), 6)) .+ 0.5)
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test get_closed_form_plan(dm).mode === :diagonal
    cache = NoLimits.build_ll_cache(dm)
    θu = get_θ0_untransformed(dm.model.fixed.fixed)
    ax = getaxes(θu)
    x0 = collect(θu)
    f = let dm = dm, cache = cache, ax = ax
        xv -> NoLimits.loglikelihood(dm, ComponentArray(xv, ax), ComponentArray();
            cache = cache, serialization = NoLimits.EnsembleSerial())
    end
    g_fd = ForwardDiff.gradient(f, x0)
    @test all(isfinite, g_fd)
    g_fwd = collect(Enzyme.gradient(
        set_runtime_activity(Enzyme.Forward), Const(f), copy(x0))[1])
    @test isapprox(g_fwd, g_fd; rtol = 1e-6, atol = 1e-9)
    g_rev = collect(Enzyme.gradient(
        set_runtime_activity(Enzyme.Reverse), Const(f), copy(x0))[1])
    @test isapprox(g_rev, g_fd; rtol = 1e-6, atol = 1e-9)
end
