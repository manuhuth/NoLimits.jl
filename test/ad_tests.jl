using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ComponentArrays
using Distributions
using Lux
using OrdinaryDiffEq
using SciMLSensitivity
using FiniteDifferences
using DataInterpolations

@testset "FixedEffects builders AD" begin
    # AD through model function builders on transformed scale.
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length = 5))
    fe = @fixedEffects begin
        σ = RealNumber(0.4, scale = :log, lower = 1.0e-12)
        ζ = NNParameters(chain; function_name = :NNB, calculate_se = false)
        Γ = SoftTreeParameters(2, 2; function_name = :STB, calculate_se = false)
        sp = SplineParameters(knots; function_name = :SPB, calculate_se = false)
        ψ = NPFParameter(2, 2, seed = 1, calculate_se = false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)

    x = [0.2, -0.1]
    function f(feθ)
        θ = inverse_transform(feθ)
        nn = model_funs.NNB(x, θ.ζ)[1]
        st = model_funs.STB(x, θ.Γ)[1]
        sp = model_funs.SPB(0.3, θ.sp)
        flow = model_funs.NPF_ψ(θ.ψ)
        return nn + st + sp + logpdf(flow, x) + θ.σ
    end

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), fixed_effects0)

    hess = ForwardDiff.hessian(f, fixed_effects0)
    @test size(hess, 1) == length(fixed_effects0)
    @test size(hess, 2) == length(fixed_effects0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "PreDE AD (model_funs)" begin
    # AD through preDE with NN/SoftTree/Spline on transformed scale.
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length = 6))
    fe = @fixedEffects begin
        ζ = NNParameters(chain; function_name = :NNB, calculate_se = false)
        Γ = SoftTreeParameters(2, 2; function_name = :STB, calculate_se = false)
        sp = SplineParameters(knots; function_name = :SPB, calculate_se = false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)
    random_effects = ComponentArray()
    constant_features_i = (x = (Age = 0.3, BMI = 1.2),)

    prede = @preDifferentialEquation begin
        nn = NNB([x.Age, x.BMI], ζ)[1]
        st = STB([x.Age, x.BMI], Γ)[1]
        spv = SPB(0.4, sp)
        total = nn + st + spv
    end

    build = get_prede_builder(prede)
    f(feθ) = build(
        inverse_transform(feθ), random_effects,
        constant_features_i, model_funs, NamedTuple()
    ).total

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), fixed_effects0)
    @test isfinite(val_fwd)
    @test all(isfinite, grad_fwd)
end

@testset "DifferentialEquation AD" begin
    # AD through out-of-place RHS with compiled context.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + s(t)
        D(x2) ~ -b * x2 + c
        s(t) = sin(t)
    end
    compile = get_de_compiler(de)
    f = get_de_f(de)
    p = (;
        fixed_effects = ComponentArray(a = 2.0, b = 3.0, c = 1.0),
        random_effects = ComponentArray(),
        constant_covariates = NamedTuple(),
        varying_covariates = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple(),
        preDE = NamedTuple(),
    )
    pc = compile(p)

    f_u(u) = sum(f(u, pc, 0.5))
    u0 = [1.0, 2.0]
    val_fwd, grad_fwd = value_and_gradient(f_u, AutoForwardDiff(), u0)

    hess = ForwardDiff.hessian(f_u, u0)
    @test size(hess, 1) == length(u0)
    @test size(hess, 2) == length(u0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
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
        a = RealNumber(2.0, scale = :log, lower = 1.0e-12)
        b = RealNumber(3.0, scale = :log, lower = 1.0e-12)
        c = RealNumber(1.0, scale = :identity)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    p0 = (;
        fixed_effects = inverse_transform(θ0),
        random_effects = ComponentArray(),
        constant_covariates = NamedTuple(),
        varying_covariates = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple(),
        preDE = NamedTuple(),
    )
    pc = compile(p0)
    u0 = [1.0, 2.0]

    fθ(θ) = begin
        fe_un = inverse_transform(θ)
        p = (;
            fixed_effects = fe_un,
            random_effects = ComponentArray(),
            constant_covariates = NamedTuple(),
            varying_covariates = NamedTuple(),
            helpers = NamedTuple(),
            model_funs = NamedTuple(),
            preDE = NamedTuple(),
        )
        pcθ = compile(p)
        sum(f(u0, pcθ, 0.5))
    end

    val_fwd, grad_fwd = value_and_gradient(fθ, AutoForwardDiff(), θ0)

    hess = ForwardDiff.hessian(fθ, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "DifferentialEquation AD (in-place)" begin
    # ForwardDiff through f! (in-place) with fixed context.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1
        D(x2) ~ -b * x2 + c
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    p = (;
        fixed_effects = ComponentArray(a = 2.0, b = 3.0, c = 1.0),
        random_effects = ComponentArray(),
        constant_covariates = NamedTuple(),
        varying_covariates = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple(),
        preDE = NamedTuple(),
    )
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
        a = RealNumber(2.0, scale = :log, lower = 1.0e-12)
        b = RealNumber(3.0, scale = :log, lower = 1.0e-12)
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
        pre = get_prede_builder(prede)(
            fe_un, ComponentArray(), NamedTuple(), NamedTuple(), helpers
        )
        p = (;
            fixed_effects = fe_un,
            random_effects = ComponentArray(),
            constant_covariates = NamedTuple(),
            varying_covariates = NamedTuple(),
            helpers = helpers,
            model_funs = NamedTuple(),
            preDE = pre,
        )
        pc = get_de_compiler(de)(p)
        sum(get_de_f(de)(u0, pc, 0.0))
    end

    val_fwd, grad_fwd = value_and_gradient(fθ, AutoForwardDiff(), θ0)

    hess = ForwardDiff.hessian(fθ, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "ODE solve AD (transformed params, richer)" begin
    de = @DifferentialEquation begin
        s(t) = sat(x1) + pre
        D(x1) ~ -a * x1 + b * s(t)
        D(x2) ~ -b * x2 + a * x1
    end
    fe = @fixedEffects begin
        a = RealNumber(0.7, scale = :log, lower = 1.0e-12)
        b = RealNumber(0.4, scale = :log, lower = 1.0e-12)
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
        pre = get_prede_builder(prede)(
            fe_un, ComponentArray(), NamedTuple(), NamedTuple(), helper_functions
        )
        p = (;
            fixed_effects = fe_un,
            random_effects = ComponentArray(),
            constant_covariates = NamedTuple(),
            varying_covariates = NamedTuple(),
            helpers = helper_functions,
            model_funs = NamedTuple(),
            preDE = pre,
        )
        pc = compile(p)
        prob = OrdinaryDiffEq.ODEProblem(de_f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(
            prob, OrdinaryDiffEq.Tsit5(); abstol = 1.0e-9, reltol = 1.0e-9
        )
        acc = de_accessors(sol, pc)
        return acc.x1(0.4) + acc.x2(0.4) + acc.s(0.4)
    end

    val_fwd, grad_fwd = value_and_gradient(fθ_fd, AutoForwardDiff(), θ0)
    hess = ForwardDiff.hessian(fθ_fd, θ0)
    @test size(hess) == (length(θ0), length(θ0))
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "ODE solve AD (edge cases)" begin
    de = @DifferentialEquation begin
        s(t) = sin(t)
        D(x1) ~ -a * x1 + b * tanh(x1) + w1(t) + s(t) + pre
        D(x2) ~ -b * x2 + a * x1 + pre
    end
    f! = get_de_f!(de)
    fe = @fixedEffects begin
        a = RealNumber(1.0, scale = :log, lower = 1.0e-12)
        b = RealNumber(0.5, scale = :log, lower = 1.0e-12)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    prede = @preDifferentialEquation begin
        pre = a + b + η1
    end
    η = ComponentArray(η1 = 0.1)
    u0 = [0.2, -0.1]
    tspan = (0.0, 0.5)

    fθ_fd(θ) = begin
        fe_un = inverse_transform(θ)
        pre = get_prede_builder(prede)(fe_un, η, NamedTuple(), NamedTuple(), NamedTuple())
        p = (;
            fixed_effects = fe_un,
            random_effects = η,
            constant_covariates = NamedTuple(),
            varying_covariates = (w1 = t -> 0.1 * t,),
            helpers = NamedTuple(),
            model_funs = NamedTuple(),
            preDE = pre,
        )
        pc = get_de_compiler(de)(p)
        prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(
            prob, OrdinaryDiffEq.Tsit5(); abstol = 1.0e-9, reltol = 1.0e-9
        )
        return sum(sol.u[end])
    end

    val_fwd, grad_fwd = value_and_gradient(fθ_fd, AutoForwardDiff(), θ0)
    hess = ForwardDiff.hessian(fθ_fd, θ0)
    @test size(hess, 1) == length(θ0)
    @test size(hess, 2) == length(θ0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "ODE solve AD (random effects)" begin
    de = @DifferentialEquation begin
        D(x1) ~ -(a + η1) * x1 + w1(t)
    end
    f! = get_de_f!(de)
    fe = @fixedEffects begin
        a = RealNumber(1.0, scale = :log, lower = 1.0e-12)
    end
    θ0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    u0 = [0.3]
    tspan = (0.0, 0.4)

    fη_fd(ηv) = begin
        fe_un = inverse_transform(θ0)
        η = ComponentArray(η1 = ηv[1])
        p = (;
            fixed_effects = fe_un,
            random_effects = η,
            constant_covariates = NamedTuple(),
            varying_covariates = (w1 = t -> 0.2 * t,),
            helpers = NamedTuple(),
            model_funs = NamedTuple(),
            preDE = NamedTuple(),
        )
        pc = get_de_compiler(de)(p)
        prob = OrdinaryDiffEq.ODEProblem(f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(
            prob, OrdinaryDiffEq.Tsit5(); abstol = 1.0e-9, reltol = 1.0e-9
        )
        return sol.u[end][1]
    end

    val_fwd, grad_fwd = value_and_gradient(fη_fd, AutoForwardDiff(), [0.1])
    hess = ForwardDiff.hessian(fη_fd, [0.1])
    @test all(isfinite, grad_fwd)
    @test size(hess) == (1, 1)
end

@testset "ODE solve AD (random effects, richer)" begin
    de = @DifferentialEquation begin
        D(x1) ~ -(a + η1) * x1 + w1(t)
        D(x2) ~ -(a + η2) * x2 + x1
    end
    fe = @fixedEffects begin
        a = RealNumber(0.9, scale = :log, lower = 1.0e-12)
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
        η = ComponentArray(η1 = ηv[1], η2 = ηv[2])
        p = (;
            fixed_effects = fe_un,
            random_effects = η,
            constant_covariates = NamedTuple(),
            varying_covariates = (w1 = t -> 0.2 * t,),
            helpers = NamedTuple(),
            model_funs = NamedTuple(),
            preDE = NamedTuple(),
        )
        pc = compile(p)
        prob = OrdinaryDiffEq.ODEProblem(de_f!, u0, tspan, pc)
        sol = OrdinaryDiffEq.solve(
            prob, OrdinaryDiffEq.Tsit5(); abstol = 1.0e-9, reltol = 1.0e-9
        )
        acc = de_accessors(sol, pc)
        return acc.x1(0.3) + acc.x2(0.3)
    end

    η0 = [0.1, -0.05]
    val_fwd, grad_fwd = value_and_gradient(fη_fd, AutoForwardDiff(), η0)
    hess = ForwardDiff.hessian(fη_fd, η0)
    @test size(hess) == (length(η0), length(η0))
end

@testset "AD through full model (callbacks + formulas)" begin
    model = @Model begin
        @helpers begin
            add1(x) = x + 1.0
        end

        @fixedEffects begin
            a = RealNumber(1.0)
            b = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
            z = Covariate()
            w1 = DynamicCovariate(; interpolation = LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :id)
        end

        @preDifferentialEquation begin
            pre = add1(a) + x.Age + η
        end

        @DifferentialEquation begin
            D(x1) ~ -b * x1 + w1(t) + pre
        end

        @initialDE begin
            x1 = pre
        end

        @formulas begin
            lin = x1(t) + z
            obs ~ Normal(lin, σ)
        end
    end

    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0, z = 1.0, w1 = (t -> 0.3 * t))
    η = ComponentArray((η = 0.1,))
    helpers = get_helper_funs(model)
    model_funs = get_model_funs(model)
    tspan = (0.0, 1.0)

    condition(u, t, integrator) = t - 0.5
    affect!(integrator) = (integrator.u[1] = integrator.u[1])
    cb = ContinuousCallback(condition, affect!)

    function objective_fd(θt)
        pre = calculate_prede(model, θt, η, const_covariates_i)
        pc = (;
            fixed_effects = θt,
            random_effects = η,
            constant_covariates = const_covariates_i,
            varying_covariates = varying_covariates,
            helpers = helpers,
            model_funs = model_funs,
            preDE = pre,
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θt, η, const_covariates_i)
        prob = ODEProblem(get_de_f!(model.de.de), u0, tspan, compiled)
        sol = solve(prob, Tsit5(); callback = cb, abstol = 1.0e-9, reltol = 1.0e-9)
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
        obs = calculate_formulas_obs(
            model, θt, η, const_covariates_i, varying_covariates, sol_accessors
        )
        return logpdf(obs.obs, 1.0)
    end

    θ0 = get_θ0_transformed(model.fixed.fixed)

    grad_fwd = ForwardDiff.gradient(objective_fd, θ0)
    grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), objective_fd, θ0)
    @test isapprox(grad_fwd, grad_fd[1]; rtol = 1.0e-5, atol = 1.0e-8)
end
