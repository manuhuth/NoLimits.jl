using Test
using NoLimits
using ComponentArrays
using Distributions
using OrdinaryDiffEq
using Lux
using DataInterpolations

@testset "Model full example (all macros)" begin
    model = @Model begin
        @helpers begin
            add1(x) = x + 1.0
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
            b = RealNumber(0.2)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
            z = Covariate()
            w1 = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:id)
        end

        @preDifferentialEquation begin
            pre = add1(a) + x.Age
        end

        @DifferentialEquation begin
            D(x1) ~ -b * x1 + w1(t) + pre
        end

        @initialDE begin
            x1 = pre
        end

        @formulas begin
            lin = add1(a) + x.Age + η + z + x1(t) + sat(w1(t))
            obs ~ Normal(lin, σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray((η = 0.1,))
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0, z = 1.0, w1 = (t -> 0.3 * t))

    # Build DE accessors with a dummy solution to provide x1(t)
    pre = calculate_prede(model, θ, η, const_covariates_i)
    sol = (t; idxs) -> pre.pre
    pc = (vars = (pre = pre.pre,), funs = (w1 = varying_covariates.w1,))
    sol_accessors = get_de_accessors_builder(model.de.de)(sol, pc)

    obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates, sol_accessors)
    @test obs.obs isa Normal
end

@testset "Model full example (ODE solve)" begin
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
            w1 = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @preDifferentialEquation begin
            pre = add1(a) + x.Age
        end

        @DifferentialEquation begin
            D(x1) ~ -b * x1 + w1(t) + pre
        end

        @initialDE begin
            x1 = pre
        end

        @formulas begin
            lin = x1(t)
            obs ~ Normal(lin, σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0, w1 = (t -> 0.3 * t))

    pre = calculate_prede(model, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = varying_covariates,
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)

    u0 = calculate_initial_state(model, θ, η, const_covariates_i)
    tspan = (0.0, 1.0)
    prob = ODEProblem(get_de_f!(model.de.de), u0, tspan, compiled)
    sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-9)
    sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)

    obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates, sol_accessors)
    @test obs.obs isa Normal
end

@testset "Model full example (ODE solve with callback, complex structure)" begin
    model = @Model begin
        @helpers begin
            add1(x) = x + 1.0
            hill(u) = abs(u)^2 / (1 + abs(u)^2)
        end

        @fixedEffects begin
            a = RealNumber(0.6)
            b = RealNumber(0.15)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
            z = Covariate()
            w1 = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:id)
        end

        @preDifferentialEquation begin
            pre = add1(a) + x.Age + η
        end

        @DifferentialEquation begin
            s(t) = hill(w1(t)) + pre
            D(x1) ~ -b * x1 + s(t)
            D(x2) ~ a * x1 - b * x2
        end

        @initialDE begin
            x1 = pre
            x2 = 0.0
        end

        @formulas begin
            lin = x1(t) + x2(t) + z + hill(w1(t))
            obs ~ Normal(lin, σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray((η = 0.1,))
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0, z = 1.0, w1 = (t -> 0.3 * t))

    pre = calculate_prede(model, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = varying_covariates,
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)

    u0 = calculate_initial_state(model, θ, η, const_covariates_i)
    tspan = (0.0, 1.0)

    condition(u, t, integrator) = u[1] - 2.0
    affect!(integrator) = (integrator.u[1] = 2.0)
    cb = ContinuousCallback(condition, affect!)

    prob = ODEProblem(get_de_f!(model.de.de), u0, tspan, compiled)
    sol = solve(prob, Tsit5(); callback=cb, abstol=1e-9, reltol=1e-9)
    sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled);

    obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates, sol_accessors)
    @test obs.obs isa Normal
end

@testset "Model example (NN + SoftTree + Spline + NPF)" begin
    chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
    knots = collect(range(0.0, 1.0; length=6))

    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            σ = RealNumber(0.5)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
            ψ = NPFParameter(1, 3, seed=1, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI])
        end

        @randomEffects begin
            η = RandomEffect(NormalizingPlanarFlow(ψ); column=:id)
        end

        @formulas begin
            lin = NN1([x.Age, x.BMI], ζ)[1] + ST([x.Age, x.BMI], Γ)[1] + SP1(0.25, sp) + η
            obs ~ Normal(lin, σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray((η = 0.1,))
    const_covariates_i = (x = (Age = 2.0, BMI = 1.0),)
    varying_covariates = (t = 0.0,)

    obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates)
    @test obs.obs isa Normal
    @test isfinite(logpdf(obs.obs, 1.0))
end

@testset "Model ODE logpdf at observation times" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = NamedTuple()
    varying_covariates = (t = 0.0,)

    pre = calculate_prede(model, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = varying_covariates,
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)

    u0 = calculate_initial_state(model, θ, η, const_covariates_i)
    tspan = (0.0, 1.0)
    t_obs = [0.0, 0.5, 1.0]
    y_obs = [1.0, 0.9, 0.8]

    prob = ODEProblem(get_de_f!(model.de.de), u0, tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=t_obs, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)

    loglik = 0.0
    for (t, y) in zip(t_obs, y_obs)
        varying_covariates = (t = t,)
        obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end

@testset "Model rejects varying covariate in DE" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + z
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
end

@testset "Model rejects dynamic covariate without (t) in DE" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + w
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
end

@testset "Model rejects constant covariate called as function in DE" begin
    @test_throws ErrorException @eval @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            c = ConstantCovariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + c(t)
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
end


@testset "Model ODE logpdf with covariate interpolation" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
            w1 = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + w1(t)
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + w1(t), σ)
        end
    end

    θ = get_θ0_untransformed(model.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = NamedTuple()

    t_obs = [0.0, 0.5, 1.0]
    w_vals = [1.0, 1.2, 1.4]
    w1_itp = LinearInterpolation(w_vals, t_obs)
    varying_covariates = (t = 0.0, w1 = w1_itp)

    pre = calculate_prede(model, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = varying_covariates,
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)

    u0 = calculate_initial_state(model, θ, η, const_covariates_i)
    tspan = (0.0, 1.0)
    y_obs = [1.0, 1.1, 1.2]

    prob = ODEProblem(get_de_f!(model.de.de), u0, tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=t_obs, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)

    loglik = 0.0
    for (t, y) in zip(t_obs, y_obs)
        varying_covariates = (t = t, w1 = w1_itp)
        obs = calculate_formulas_obs(model, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end
