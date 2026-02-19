using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ComponentArrays
using OrdinaryDiffEq

@testset "ODE solve AD (transformed params, richer)" begin
    de = @DifferentialEquation begin
        s(t) = sat(x1) + pre
        D(x1) ~ -a * x1 + b * s(t)
        D(x2) ~ -b * x2 + a * x1
    end
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    fe = @fixedEffects begin
        a = RealNumber(0.7, scale=:log, lower=1e-12)
        b = RealNumber(0.4, scale=:log, lower=1e-12)
    end
    prede = @preDifferentialEquation begin
        pre = a + b
    end
    compile = get_de_compiler(de)
    de_f! = get_de_f!(de)
    de_accessors = get_de_accessors_builder(de)
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    helper_functions = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    u0 = [0.2, 0.1]
    tspan = (0.0, 0.6)

    fθ_fd(θ) = begin
        fe_un = inverse_transform(θ)
        pre = get_prede_builder(prede)(fe_un, ComponentArray(), NamedTuple(), NamedTuple(), helper_functions)
        p = (; fixed_effects = fe_un,
              random_effects = ComponentArray(),
              constant_covariates = NamedTuple(),
              varying_covariates = NamedTuple(),
              helpers = helper_functions,
              model_funs = NamedTuple(),
              preDE = pre)
        pc = compile(p)
        prob = OrdinaryDiffEq.ODEProblem(de_f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
        acc = de_accessors(sol, pc)
        return acc.x1(0.4) + acc.x2(0.4) + acc.s(0.4)
    end

    val_fwd, grad_fwd = value_and_gradient(fθ_fd, AutoForwardDiff(), θ0)
    hess = ForwardDiff.hessian(fθ_fd, θ0)
    @test size(hess) == (length(θ0), length(θ0))
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)
end

@testset "ODE solve AD (random effects, richer)" begin
    de = @DifferentialEquation begin
        D(x1) ~ -(a + η1) * x1 + w1(t)
        D(x2) ~ -(a + η2) * x2 + x1
    end
    fe = @fixedEffects begin
        a = RealNumber(0.9, scale=:log, lower=1e-12)
    end
    compile = get_de_compiler(de)
    de_f! = get_de_f!(de)
    de_accessors = get_de_accessors_builder(de)
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    u0 = [0.3, 0.1]
    tspan = (0.0, 0.4)

    fη_fd(ηv) = begin
        fe_un = inverse_transform(θ0)
        η = ComponentArray(η1=ηv[1], η2=ηv[2])
        p = (; fixed_effects = fe_un,
              random_effects = η,
              constant_covariates = NamedTuple(),
              varying_covariates = (w1 = t -> 0.2 * t,),
              helpers = NamedTuple(),
              model_funs = NamedTuple(),
              preDE = NamedTuple())
        pc = compile(p)
        prob = OrdinaryDiffEq.ODEProblem(de_f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
        acc = de_accessors(sol, pc)
        return acc.x1(0.3) + acc.x2(0.3)
    end

    η0 = [0.1, -0.05]
    val_fwd, grad_fwd = value_and_gradient(fη_fd, AutoForwardDiff(), η0)
    hess = ForwardDiff.hessian(fη_fd, η0)
    @test size(hess) == (length(η0), length(η0))
end
