using Test
using NoLimits
using ComponentArrays
using Distributions
using OrdinaryDiffEq
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
