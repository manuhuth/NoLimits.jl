using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity

@testset "ODE solve AD (transformed params)" begin
    de = @DifferentialEquation begin
        D(x1) ~ -a * x1
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    fe = @fixedEffects begin
        a = RealNumber(1.0, scale=:log, lower=1e-12)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    u0 = [1.0]
    tspan = (0.0, 1.0)

    fθ_fd(θ) = begin
        p = build_de_params(de, θ;
                            random_effects=ComponentArray(NamedTuple()),
                            constant_covariates=NamedTuple(),
                            varying_covariates=NamedTuple(),
                            helpers=NamedTuple(),
                            model_funs=NamedTuple(),
                            prede_builder=(fe, re, consts, model_funs, helpers) -> NamedTuple(),
                            inverse_transform=inverse_transform)
        prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
        return sol.u[end][1]
    end

    val_fwd, grad_fwd = value_and_gradient(fθ_fd, AutoForwardDiff(), θ0)

    hess = ForwardDiff.hessian(fθ_fd, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

end

@testset "ODE solve AD (edge cases)" begin
    de = @DifferentialEquation begin
        s(t) = sin(t)
        D(x1) ~ -a * x1 + b * tanh(x1) + w1(t) + s(t) + pre
        D(x2) ~ -b * x2 + a * x1 + pre
    end
    f! = get_de_f!(de)
    fe = @fixedEffects begin
        a = RealNumber(1.0, scale=:log, lower=1e-12)
        b = RealNumber(0.5, scale=:log, lower=1e-12)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    prede = @preDifferentialEquation begin
        pre = a + b + η1
    end
    η = ComponentArray(η1=0.1)
    u0 = [0.2, -0.1]
    tspan = (0.0, 0.5)

    fθ_fd(θ) = begin
        fe_un = inverse_transform(θ)
        pre = get_prede_builder(prede)(fe_un, η, NamedTuple(), NamedTuple(), NamedTuple())
        p = (; fixed_effects = fe_un,
              random_effects = η,
              constant_covariates = NamedTuple(),
              varying_covariates = (w1 = t -> 0.1 * t,),
              helpers = NamedTuple(),
              model_funs = NamedTuple(),
              preDE = pre)
        pc = get_de_compiler(de)(p)
        prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
        return sum(sol.u[end])
    end

    val_fwd, grad_fwd = value_and_gradient(fθ_fd, AutoForwardDiff(), θ0)
    hess = ForwardDiff.hessian(fθ_fd, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

end

@testset "ODE solve AD (random effects)" begin
    de = @DifferentialEquation begin
        D(x1) ~ -(a + η1) * x1 + w1(t)
    end
    f! = get_de_f!(de)
    fe = @fixedEffects begin
        a = RealNumber(1.0, scale=:log, lower=1e-12)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    u0 = [0.3]
    tspan = (0.0, 0.4)

    fη_fd(ηv) = begin
        fe_un = inverse_transform(θ0)
        η = ComponentArray(η1=ηv[1])
        p = (; fixed_effects = fe_un,
              random_effects = η,
              constant_covariates = NamedTuple(),
              varying_covariates = (w1 = t -> 0.2 * t,),
              helpers = NamedTuple(),
              model_funs = NamedTuple(),
              preDE = NamedTuple())
        pc = get_de_compiler(de)(p)
        prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
        return sol.u[end][1]
    end

    val_fwd, grad_fwd = value_and_gradient(fη_fd, AutoForwardDiff(), [0.1])
    hess = ForwardDiff.hessian(fη_fd, [0.1])
    @test isfinite(val_fwd)
    @test all(isfinite, grad_fwd)
    @test size(hess) == (1, 1)

end
