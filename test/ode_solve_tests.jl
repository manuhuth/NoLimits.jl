using Test
using NoLimits
using ComponentArrays
using OrdinaryDiffEq

@testset "ODE solve integration" begin
    de = @DifferentialEquation begin
        D(x1) ~ -a * x1
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    p = (; fixed_effects = ComponentArray(a=1.0),
          random_effects = ComponentArray(),
          constant_covariates = NamedTuple(),
          varying_covariates = NamedTuple(),
          helpers = NamedTuple(),
          model_funs = NamedTuple(),
          preDE = NamedTuple())
    pc = compile(p)
    u0 = [1.0]
    tspan = (0.0, 1.0)

    prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, pc)
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
    @test isapprox(sol.u[end][1], exp(-1.0); rtol=1e-5, atol=1e-6)
end

@testset "ODE solve with covariates and preDE" begin
    de = @DifferentialEquation begin
        s(t) = sin(t)
        D(x1) ~ -a * x1 + w1(t) + s(t) + pre
        D(x2) ~ -b * x2 + pre
    end
    f! = get_de_f!(de)
    fe = @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.5)
    end
    prede = @preDifferentialEquation begin
        pre = a + b + η1
    end
    fe0 = get_θ0_untransformed(fe)
    η = ComponentArray(η1=0.2)
    pre = get_prede_builder(prede)(fe0, η, NamedTuple(), NamedTuple(), NamedTuple())
    p = (; fixed_effects = fe0,
          random_effects = η,
          constant_covariates = NamedTuple(),
          varying_covariates = (w1 = t -> 0.1 * t,),
          helpers = NamedTuple(),
          model_funs = NamedTuple(),
          preDE = pre)
    pc = get_de_compiler(de)(p)
    u0 = [0.5, 1.0]
    tspan = (0.0, 0.5)
    prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, pc)
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)
    @test length(sol.u[end]) == 2
    @test all(isfinite, sol.u[end])
end

@testset "ODE solve with helpers, random effects, preDE, and accessors" begin
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    de = @DifferentialEquation begin
        s(t) = sat(x1) + pre
        D(x1) ~ -a * x1 + w1(t) + s(t)
        D(x2) ~ -b * x2 + pre
    end
    fe = @fixedEffects begin
        a = RealNumber(0.7)
        b = RealNumber(0.4)
    end
    prede = @preDifferentialEquation begin
        pre = a + η1
    end

    fe0 = get_θ0_untransformed(fe)
    η = ComponentArray(η1=0.2)
    helper_functions = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    pre = get_prede_builder(prede)(fe0, η, NamedTuple(), NamedTuple(), helper_functions)
    p = (; fixed_effects = fe0,
          random_effects = η,
          constant_covariates = NamedTuple(),
          varying_covariates = (w1 = t -> 0.1 * t,),
          helpers = helper_functions,
          model_funs = NamedTuple(),
          preDE = pre)
    pc = get_de_compiler(de)(p)

    u0 = [0.5, 1.0]
    tspan = (0.0, 0.5)
    prob = OrdinaryDiffEq.ODEProblem(get_de_f!(de), u0, tspan, pc)
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5(); abstol=1e-9, reltol=1e-9)

    accessors = get_de_accessors_builder(de)(sol, pc);
    t = 0.3
    @test isapprox(accessors.x1(t), sol(t; idxs=1); rtol=1e-6, atol=1e-8)
    @test isapprox(accessors.x2(t), sol(t; idxs=2); rtol=1e-6, atol=1e-8)
    @test isapprox(accessors.s(t), helper_functions.sat(sol(t; idxs=1)) + pre.pre; rtol=1e-6, atol=1e-8)
end
