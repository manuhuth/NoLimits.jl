using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using MCMCChains
using Lux
using Random
using SciMLBase

@testset "MCMC sampler-dependent defaults" begin
    nuts_defaults = NoLimits._mcmc_sampler_defaults(NUTS())
    @test nuts_defaults.n_samples == 1000
    @test nuts_defaults.n_adapt == 500

    mh_defaults = NoLimits._mcmc_sampler_defaults(MH())
    @test mh_defaults.n_samples == 2500
    @test mh_defaults.n_adapt == 0

    hmc_defaults = NoLimits._mcmc_sampler_defaults(HMC(0.01, 5))
    @test hmc_defaults.n_samples == 1500
    @test hmc_defaults.n_adapt == 750
end



@testset "MCMC basic (no RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(softplus(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1.0, 1.05, 0.98, 1.02]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    meth = NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=30, n_adapt=10, progress=true))
    res = fit_model(dm, meth)

    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
    @test NoLimits.get_observed(res).y == df.y
end

@testset "MCMC serial vs threaded is reproducible (MH)" begin
    Threads.nthreads() < 2 && return

    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(softplus(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1.0, 1.05, 0.98, 1.02]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false, verbose=false))
    res_serial = fit_model(dm, method; serialization=SciMLBase.EnsembleSerial(), rng=MersenneTwister(123))
    res_threads = fit_model(dm, method; serialization=SciMLBase.EnsembleThreads(), rng=MersenneTwister(123))
    @test Array(NoLimits.get_chain(res_serial)) == Array(NoLimits.get_chain(res_threads))
end

@testset "MCMC requires priors" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MCMC())
end

#TODO think about we actually wanna reject it or allow for the same naming. estimation kwargs and args should be the same anyway.
@testset "MCMC supports random effects (basic)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(exp(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=15, n_adapt=5, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC supports constants for fixed effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false));
                    constants=(a=0.2,))

    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC fixed-effects-only rejects all constants" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    err = try
        fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false));
                  constants=(a=0.2, σ=0.5))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("at least one sampled parameter", sprint(showerror, err))
end

@testset "MCMC rejects penalty terms" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.MCMC(); penalty=(a=1.0,))
end

@testset "MCMC fixed vector parameters" begin
    model = @Model begin
        @fixedEffects begin
            β = RealVector([0.2, -0.1], prior=MvNormal(zeros(2), [2.0 0.0; 0.0 1.0]))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            μ = exp(β[1] * z + β[2] * z^2)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        z = [0.1, 0.2, 0.15, 0.3],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))

    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC with NN/SoftTree/Spline (fixed blocks)" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))
    ps, st = Lux.setup(Random.default_rng(), chain)

    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false, prior=filldist(Normal(0.0, 1.0), Lux.parameterlength(ps)))
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
            z = Covariate()
        end

        @formulas begin
            μ = softplus(exp(a) + NN1([x.Age, x.BMI], ζ)[1] +
                         ST1([x.Age, x.BMI], Γ)[1] + SP1(x.BMI / 50, sp) + z)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 40.0, 40.0],
        BMI = [20.0, 20.0, 22.0, 22.0],
        z = [0.2, 0.1, 0.15, 0.05],
        y = [1.0, 1.05, 1.1, 1.08]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false));
                    constants=( Γ=θ0.Γ, sp=θ0.sp))

    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains

    res_mle = fit_model(dm, NoLimits.MLE(); constants=(Γ=θ0.Γ, sp=θ0.sp))
    @test res_mle isa FitResult
    @test isfinite(NoLimits.get_objective(res_mle))

    res_map = fit_model(dm, NoLimits.MAP(); constants=(Γ=θ0.Γ, sp=θ0.sp))
    @test res_map isa FitResult
    @test isfinite(NoLimits.get_objective(res_map))
end

@testset "MCMC ODE with NN/SoftTree/Spline (fixed blocks)" begin
    chain = Chain(Dense(1, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))
    ps, st = Lux.setup(Random.default_rng(), chain)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false, prior=filldist(Normal(0.0, 1.0), Lux.parameterlength(ps)))
            Γ = SoftTreeParameters(1, 2; function_name=:ST1, calculate_se=false)
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on=:ID)
            z = Covariate()
        end

        @preDifferentialEquation begin
            pre = NN1([x.Age], ζ)[1] + ST1([x.Age], Γ)[1] + SP1(x.Age / 100, sp) + exp(a)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + pre 
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t) + z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        z = [0.1, 0.15, 0.05, 0.1],
        y = [1.0, 1.05, 1.1, 1.08]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false));
                    constants=(ζ=θ0.ζ, Γ=θ0.Γ, sp=θ0.sp))

    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains

    res_mle = fit_model(dm, NoLimits.MLE(); constants=(ζ=θ0.ζ, Γ=θ0.Γ, sp=θ0.sp));
    @test res_mle isa FitResult
    @test isfinite(NoLimits.get_objective(res_mle))

    res_map = fit_model(dm, NoLimits.MAP(); constants=(ζ=θ0.ζ, Γ=θ0.Γ, sp=θ0.sp))
    @test res_map isa FitResult
    @test isfinite(NoLimits.get_objective(res_map))
end

@testset "MCMC fixed-only non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        end

        @formulas begin
            λ = exp(a + b * z)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.2, 0.5, 0.7, 1.0, 1.2],
        y = [1, 1, 2, 2, 2, 3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))

    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
    @test NoLimits.get_observed(res).y == df.y
end
