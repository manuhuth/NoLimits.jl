using Test
using NoLimits
using DataFrames
using Distributions
using OrdinaryDiffEq
using ComponentArrays
using DataInterpolations
using Lux

function _varying_at(ind, idx)
    pairs = Pair{Symbol, Any}[]
    vary = ind.series.vary
    for name in keys(vary)
        v = getfield(vary, name)
        if name == :t
            push!(pairs, :t => v[idx])
        elseif v isa AbstractVector
            push!(pairs, name => v[idx])
        elseif v isa NamedTuple
            sub = NamedTuple{keys(v)}(Tuple(getfield(v, k)[idx] for k in keys(v)))
            push!(pairs, name => sub)
        end
    end
    return merge(NamedTuple(pairs), ind.series.dyn)
end

@testset "DataModel ODE logpdf (basic)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
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

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        y = [1.0, 0.9, 0.8]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

    ind = get_individual(dm, 1)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = NamedTuple()
    varying_covariates = (t = 0.0,)

    pre = calculate_prede(model_saveat, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = NamedTuple(),
        helpers = get_helper_funs(model_saveat),
        model_funs = get_model_funs(model_saveat),
        preDE = pre
    )
    compiled = get_de_compiler(model_saveat.de.de)(pc)

    u0 = calculate_initial_state(model_saveat, θ, η, const_covariates_i)
    prob = ODEProblem(get_de_f!(model_saveat.de.de), u0, ind.tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=ind.saveat, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model_saveat.de.de)(sol, compiled)

    loglik = 0.0
    for (i, y) in enumerate(ind.series.obs.y)
        varying_covariates = _varying_at(ind, i)
        obs = calculate_formulas_obs(model_saveat, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end

@testset "DataModel ODE logpdf with time offsets" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4)
        end

        @covariates begin
            t = Covariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + x1(t + 0.25), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 0.9]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

    ind = get_individual(dm, 1)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = NamedTuple()

    pre = calculate_prede(model_saveat, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = NamedTuple(),
        helpers = get_helper_funs(model_saveat),
        model_funs = get_model_funs(model_saveat),
        preDE = pre
    )
    compiled = get_de_compiler(model_saveat.de.de)(pc)

    u0 = calculate_initial_state(model_saveat, θ, η, const_covariates_i)
    prob = ODEProblem(get_de_f!(model_saveat.de.de), u0, ind.tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=ind.saveat, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model_saveat.de.de)(sol, compiled)

    loglik = 0.0
    for (i, y) in enumerate(ind.series.obs.y)
        varying_covariates = _varying_at(ind, i)
        obs = calculate_formulas_obs(model_saveat, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end

@testset "DataModel ODE logpdf with covariate interpolation" begin
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

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 0.5, 1.0],
        w1 = [1.0, 1.2, 1.4],
        y = [1.0, 1.1, 1.2]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

    ind = get_individual(dm, 1)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η = ComponentArray()
    const_covariates_i = NamedTuple()

    pre = calculate_prede(model_saveat, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = get_helper_funs(model_saveat),
        model_funs = get_model_funs(model_saveat),
        preDE = pre
    )
    compiled = get_de_compiler(model_saveat.de.de)(pc)

    u0 = calculate_initial_state(model_saveat, θ, η, const_covariates_i)
    prob = ODEProblem(get_de_f!(model_saveat.de.de), u0, ind.tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=ind.saveat, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model_saveat.de.de)(sol, compiled)

    loglik = 0.0
    for (i, y) in enumerate(ind.series.obs.y)
        varying_covariates = _varying_at(ind, i)
        obs = calculate_formulas_obs(model_saveat, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end

@testset "DataModel ODE logpdf with NN/SoftTree/Spline + multiple RE groups (1)" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=6))

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.15)
            σ = RealNumber(0.3)
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @preDifferentialEquation begin
            pre = NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1] + SP1(x.Age / 100, sp) + η_id
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + pre + w(t) + η_site
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0],
        BMI = [20.0, 20.0, 25.0, 25.0],
        w = [0.2, 0.4, 0.1, 0.3],
        y = [1.0, 0.9, 1.1, 1.0]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

    ind = get_individual(dm, 1)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η = ComponentArray((η_id = 0.1, η_site = -0.05))
    const_covariates_i = ind.const_cov

    pre = calculate_prede(model_saveat, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = get_helper_funs(model_saveat),
        model_funs = get_model_funs(model_saveat),
        preDE = pre
    )
    compiled = get_de_compiler(model_saveat.de.de)(pc)

    u0 = calculate_initial_state(model_saveat, θ, η, const_covariates_i)
    prob = ODEProblem(get_de_f!(model_saveat.de.de), u0, ind.tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=ind.saveat, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model_saveat.de.de)(sol, compiled)

    loglik = 0.0
    for (i, y) in enumerate(ind.series.obs.y)
        varying_covariates = _varying_at(ind, i)
        obs = calculate_formulas_obs(model_saveat, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end

@testset "DataModel ODE logpdf with NN/SoftTree/Spline + multiple RE groups (2)" begin
    chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
    knots = collect(range(0.0, 1.0; length=5))

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.25)
            σ = RealNumber(0.35)
            ζ = NNParameters(chain; function_name=:NN2, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST2, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP2, degree=2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
            w = DynamicCovariate(; interpolation=CubicSpline)
            z = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
            η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
        end

        @preDifferentialEquation begin
            pre = NN2([x.Age, x.BMI], ζ)[1] + ST2([x.Age, x.BMI], Γ)[1] + SP2(x.BMI / 50, sp) + η_id
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + pre + w(t) + z(t) + η_site + η_year
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + z(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        SITE = [:A, :A, :A, :B, :B, :B],
        YEAR = [2020, 2020, 2020, 2021, 2021, 2021],
        t = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        Age = [35.0, 35.0, 35.0, 45.0, 45.0, 45.0],
        BMI = [22.0, 22.0, 22.0, 28.0, 28.0, 28.0],
        w = [0.2, 0.45, 0.6, 0.1, 0.3, 0.4],
        z = [0.1, 0.15, 0.2, 0.05, 0.1, 0.15],
        y = [1.0, 0.98, 0.95, 1.05, 1.02, 1.0]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

    ind = get_individual(dm, 1)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η = ComponentArray((η_id = 0.05, η_site = -0.02, η_year = 0.01))
    const_covariates_i = ind.const_cov

    pre = calculate_prede(model_saveat, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = get_helper_funs(model_saveat),
        model_funs = get_model_funs(model_saveat),
        preDE = pre
    )
    compiled = get_de_compiler(model_saveat.de.de)(pc)

    u0 = calculate_initial_state(model_saveat, θ, η, const_covariates_i)
    prob = ODEProblem(get_de_f!(model_saveat.de.de), u0, ind.tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=ind.saveat, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model_saveat.de.de)(sol, compiled)

    loglik = 0.0
    for (i, y) in enumerate(ind.series.obs.y)
        varying_covariates = _varying_at(ind, i)
        obs = calculate_formulas_obs(model_saveat, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end

@testset "DataModel ODE logpdf with NN/SoftTree/Spline + multiple RE groups (3)" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))

    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.25)
            ζ = NNParameters(chain; function_name=:NN3, calculate_se=false)
            Γ = SoftTreeParameters(2, 2; function_name=:ST3, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP3, degree=2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
            w = DynamicCovariate(; interpolation=LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_year = RandomEffect(Normal(0.0, 1.0); column=:YEAR)
        end

        @preDifferentialEquation begin
            pre = sat(NN3([x.Age, x.BMI], ζ)[1] + ST3([x.Age, x.BMI], Γ)[1]) + SP3(x.Age / 100, sp) + η_id
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + pre + w(t) + η_year
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        YEAR = [2020, 2020, 2021, 2021],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [32.0, 32.0, 42.0, 42.0],
        BMI = [21.0, 21.0, 27.0, 27.0],
        w = [0.15, 0.35, 0.1, 0.25],
        y = [1.0, 0.92, 1.08, 1.0]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

    ind = get_individual(dm, 1)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η = ComponentArray((η_id = 0.08, η_year = -0.03))
    const_covariates_i = ind.const_cov

    pre = calculate_prede(model_saveat, θ, η, const_covariates_i)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_covariates_i,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = get_helper_funs(model_saveat),
        model_funs = get_model_funs(model_saveat),
        preDE = pre
    )
    compiled = get_de_compiler(model_saveat.de.de)(pc)

    u0 = calculate_initial_state(model_saveat, θ, η, const_covariates_i)
    prob = ODEProblem(get_de_f!(model_saveat.de.de), u0, ind.tspan, compiled)
    sol = solve(prob, Tsit5(); saveat=ind.saveat, save_everystep=false, dense=false)
    sol_accessors = get_de_accessors_builder(model_saveat.de.de)(sol, compiled)

    loglik = 0.0
    for (i, y) in enumerate(ind.series.obs.y)
        varying_covariates = _varying_at(ind, i)
        obs = calculate_formulas_obs(model_saveat, θ, η, const_covariates_i, varying_covariates, sol_accessors)
        loglik += logpdf(obs.y, y)
    end
    @test isfinite(loglik)
end
