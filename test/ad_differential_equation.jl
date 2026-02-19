using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using Zygote
using ComponentArrays

@testset "DifferentialEquation AD" begin
    # AD through out-of-place RHS with compiled context.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + s(t)
        D(x2) ~ -b * x2 + c
        s(t) = sin(t)
    end
    compile = get_de_compiler(de)
    f = get_de_f(de)
    p = (; fixed_effects = ComponentArray(a=2.0, b=3.0, c=1.0),
          random_effects = ComponentArray(),
          constant_covariates = NamedTuple(),
          varying_covariates = NamedTuple(),
          helpers = NamedTuple(),
          model_funs = NamedTuple(),
          preDE = NamedTuple())
    pc = compile(p)

    f_u(u) = sum(f(u, pc, 0.5))
    u0 = [1.0, 2.0]
    val_fwd, grad_fwd = value_and_gradient(f_u, AutoForwardDiff(), u0)
    grad_zyg = Zygote.gradient(f_u, u0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    hess = ForwardDiff.hessian(f_u, u0)
    @test size(hess, 1) == length(u0)
    @test size(hess, 2) == length(u0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

    hess_mixed = hessian_fwd_over_zygote(f_u, u0)
    @test size(hess_mixed, 1) == length(u0)
    @test size(hess_mixed, 2) == length(u0)
    @test isapprox(hess_mixed, hess; rtol=1e-6, atol=1e-8)
end

@testset "DifferentialEquation AD (params, transformed)" begin
    # AD through parameters on transformed scale using out-of-place RHS.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + s(t)
        D(x2) ~ -b * x2 + c
        s(t) = sin(t)
    end
    compile = get_de_compiler(de)
    f = get_de_f(de)
    fe = @fixedEffects begin
        a = RealNumber(2.0, scale=:log, lower=1e-12)
        b = RealNumber(3.0, scale=:log, lower=1e-12)
        c = RealNumber(1.0, scale=:identity)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    p0 = (; fixed_effects = inverse_transform(θ0),
          random_effects = ComponentArray(),
          constant_covariates = NamedTuple(),
          varying_covariates = NamedTuple(),
          helpers = NamedTuple(),
          model_funs = NamedTuple(),
          preDE = NamedTuple())
    pc = compile(p0)
    u0 = [1.0, 2.0]

    fθ(θ) = begin
        fe_un = inverse_transform(θ)
        p = (; fixed_effects = fe_un,
              random_effects = ComponentArray(),
              constant_covariates = NamedTuple(),
              varying_covariates = NamedTuple(),
              helpers = NamedTuple(),
              model_funs = NamedTuple(),
              preDE = NamedTuple())
        pcθ = compile(p)
        sum(f(u0, pcθ, 0.5))
    end

    val_fwd, grad_fwd = value_and_gradient(fθ, AutoForwardDiff(), θ0)
    grad_zyg = Zygote.gradient(fθ, θ0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    hess = ForwardDiff.hessian(fθ, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

    hess_mixed = hessian_fwd_over_zygote(fθ, θ0)
    @test size(hess_mixed, 1) == length(θ0)
    @test size(hess_mixed, 2) == length(θ0)
    @test isapprox(hess_mixed, hess; rtol=1e-6, atol=1e-8)
end

@testset "DifferentialEquation AD (in-place)" begin
    # ForwardDiff through f! (in-place) with fixed context.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1
        D(x2) ~ -b * x2 + c
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    p = (; fixed_effects = ComponentArray(a=2.0, b=3.0, c=1.0),
          random_effects = ComponentArray(),
          constant_covariates = NamedTuple(),
          varying_covariates = NamedTuple(),
          helpers = NamedTuple(),
          model_funs = NamedTuple(),
          preDE = NamedTuple())
    pc = compile(p)
    u0 = [1.0, 2.0]

    g(u) = begin
        du = similar(u)
        f!(du, u, pc, 0.0)
        return sum(du)
    end

    val_fwd, grad_fwd = value_and_gradient(g, AutoForwardDiff(), u0)
    @test length(grad_fwd) == length(u0)
end

@testset "DifferentialEquation AD (macros + preDE)" begin
    # Full macro path with helpers/preDE and AD on out-of-place RHS.
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    fe = @fixedEffects begin
        a = RealNumber(2.0, scale=:log, lower=1e-12)
        b = RealNumber(3.0, scale=:log, lower=1e-12)
    end
    prede = @preDifferentialEquation begin
        pre = a + b
    end
    de = @DifferentialEquation begin
        D(x1) ~ sat(x1) + pre
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    helpers = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    u0 = [0.5]

    fθ(θ) = begin
        fe_un = inverse_transform(θ)
        pre = get_prede_builder(prede)(fe_un, ComponentArray(), NamedTuple(), NamedTuple(), helpers)
        p = (; fixed_effects = fe_un,
              random_effects = ComponentArray(),
              constant_covariates = NamedTuple(),
              varying_covariates = NamedTuple(),
              helpers = helpers,
              model_funs = NamedTuple(),
              preDE = pre)
        pc = get_de_compiler(de)(p)
        sum(get_de_f(de)(u0, pc, 0.0))
    end

    val_fwd, grad_fwd = value_and_gradient(fθ, AutoForwardDiff(), θ0)
    grad_zyg = Zygote.gradient(fθ, θ0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    hess = ForwardDiff.hessian(fθ, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

    hess_mixed = hessian_fwd_over_zygote(fθ, θ0)
    @test size(hess_mixed, 1) == length(θ0)
    @test size(hess_mixed, 2) == length(θ0)
    @test isapprox(hess_mixed, hess; rtol=1e-6, atol=1e-8)
end
