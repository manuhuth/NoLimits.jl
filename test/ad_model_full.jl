using Test
using NoLimits
using ComponentArrays
using Distributions
using OrdinaryDiffEq
using SciMLSensitivity
using Zygote
using ForwardDiff
using FiniteDifferences
using DataInterpolations

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
            w1 = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:id)
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

    inverse_transform = get_inverse_transform(model.fixed.fixed)

    function objective_fd(θt)
        pre = calculate_prede(model, θt, η, const_covariates_i)
        pc = (;
            fixed_effects = θt,
            random_effects = η,
            constant_covariates = const_covariates_i,
            varying_covariates = varying_covariates,
            helpers = helpers,
            model_funs = model_funs,
            preDE = pre
        )
        compiled = get_de_compiler(model.de.de)(pc)
        u0 = calculate_initial_state(model, θt, η, const_covariates_i)
        prob = ODEProblem(get_de_f!(model.de.de), u0, tspan, compiled)
        sol = solve(prob, Tsit5(); callback=cb, abstol=1e-9, reltol=1e-9)
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
        obs = calculate_formulas_obs(model, θt, η, const_covariates_i, varying_covariates, sol_accessors)
        return logpdf(obs.obs, 1.0)
    end

    function objective_zyg(θt)
        f!_local = (du, u, θp, t) -> begin
            fe = inverse_transform(ComponentArray((a = θp[1], b = θp[2], σ = θp[3])))
            pre = calculate_prede(model, fe, η, const_covariates_i)
            pc = (;
                fixed_effects = fe,
                random_effects = η,
                constant_covariates = const_covariates_i,
                varying_covariates = varying_covariates,
                helpers = helpers,
                model_funs = model_funs,
                preDE = pre
            )
            compiled = get_de_compiler(model.de.de)(pc)
            get_de_f!(model.de.de)(du, u, compiled, t)
            return nothing
        end
        fe0 = inverse_transform(ComponentArray((a = θt[1], b = θt[2], σ = θt[3])))
        u0 = calculate_initial_state(model, fe0, η, const_covariates_i)
        prob = ODEProblem(f!_local, u0, tspan, θt)
        sol = solve(prob, Tsit5();
                    callback=cb, abstol=1e-9, reltol=1e-9,
                    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        sol_accessors = get_de_accessors_builder(model.de.de)(sol, get_de_compiler(model.de.de)((;
            fixed_effects = fe0,
            random_effects = η,
            constant_covariates = const_covariates_i,
            varying_covariates = varying_covariates,
            helpers = helpers,
            model_funs = model_funs,
            preDE = calculate_prede(model, fe0, η, const_covariates_i)
        )))
        obs = calculate_formulas_obs(model, fe0, η, const_covariates_i, varying_covariates, sol_accessors)
        return logpdf(obs.obs, 1.0)
    end

    θ0 = get_θ0_transformed(model.fixed.fixed)
    
    grad_fwd = ForwardDiff.gradient(objective_fd, θ0)
    grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), objective_fd, θ0)
    
end
