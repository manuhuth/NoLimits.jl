using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using MCMCChains
using LinearAlgebra
using Random
using Lux
using SciMLBase

@testset "MCMC RE multivariate (NUTS)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η_id = RandomEffect(MvNormal([0.0, 0.0], I(2)); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_id[1], σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE multiple groups (NUTS)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE constants_re (NUTS)" begin
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
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=20, n_adapt=10, progress=false));
                    constants_re=(; η=(; A=0.0,)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE-only sampling with fixed constants" begin
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
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false));
                    constants=(a=0.2, σ=0.5))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC constants_re validates scalar shape early" begin
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
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    err = try
        fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=10, n_adapt=5, progress=false));
                  constants_re=(; η=(; A=[0.0],)))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("scalar number", sprint(showerror, err))
end

@testset "MCMC constants_re validates multivariate length early" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], I(2)); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    err = try
        fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=10, n_adapt=5, progress=false));
                  constants_re=(; η=(; A=[0.0],)))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("must have length 2", sprint(showerror, err))
end

@testset "MCMC constants_re accepts valid multivariate constants" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], I(2)); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=20, n_adapt=10, progress=false));
                    constants_re=(; η=(; A=[0.0, 0.0],)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE MH sampler" begin
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
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE non-normal distribution (Laplace)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Distributions.Laplace(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE NormalizingPlanarFlow" begin
    npf0 = NPFParameter(1, 2, seed=1, calculate_se=false)
    n_npf = length(npf0.value)
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
            ψ = NPFParameter(1, 2, seed=1, calculate_se=false,
                             prior=filldist(Normal(0.0, 1.0), n_npf))
        end

        @randomEffects begin
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η_flow[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res_nuts = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=20, n_adapt=10, progress=false)))
    res_mh = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))
    @test res_nuts isa FitResult
    @test NoLimits.get_chain(res_nuts) isa MCMCChains.Chains
    @test res_mh isa FitResult
    @test NoLimits.get_chain(res_mh) isa MCMCChains.Chains
end

@testset "MCMC RE with NN/SoftTree/Spline" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=4))
    ps, _ = Lux.setup(Random.default_rng(), chain)
    st0 = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
    sp0 = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)

    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
            ζ = NNParameters(chain; function_name=:NN1, calculate_se=false,
                             prior=filldist(Normal(0.0, 1.0), Lux.parameterlength(ps)))
            Γ = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false,
                                   prior=filldist(Normal(0.0, 1.0), length(st0.value)))
            sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false,
                                  prior=filldist(Normal(0.0, 1.0), length(sp0.value)))
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
            z = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            μ = softplus(exp(a) + NN1([x.Age, x.BMI], ζ)[1] +
                         ST1([x.Age, x.BMI], Γ)[1] + SP1(x.BMI / 50, sp) + z + η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        Age = [30.0, 30.0, 35.0, 35.0],
        BMI = [22.0, 22.0, 25.0, 25.0],
        z = [0.1, 0.2, 0.15, 0.3],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=15, n_adapt=5, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE with ODE model" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            k = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @initialDE begin
            x1 = exp(a + η)
        end

        @DifferentialEquation begin
            D(x1) ~ -exp(k) * x1
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 0.8, 1.05, 0.85]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=10, n_adapt=5, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE with threaded likelihood" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=10, n_adapt=5, progress=false));
                    serialization=EnsembleThreads())
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE with HMM" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p1_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p2_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p1 = 0.8 / (1 + exp(-(p1_r + η))) + 0.1
            p2 = 0.8 / (1 + exp(-p2_r)) + 0.1
            P = [0.9 0.1;
                 0.2 0.8]
            y ~ DiscreteTimeDiscreteStatesHMM(P,
                                              (Bernoulli(p1), Bernoulli(p2)),
                                              Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE with continuous-time HMM" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log, prior=LogNormal(0.01, 0.03))
            λ21_r = RealNumber(0.1, scale=:log, prior=LogNormal(0.01, 0.03))
            p1_r  = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p2_r  = RealNumber(0.0, prior=Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 0.8 / (1 + exp(-(p1_r + η))) + 0.1
            p2 = 0.8 / (1 + exp(-p2_r)) + 0.1
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=2, n_adapt=0, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end

@testset "MCMC RE non-HMM Bernoulli outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.8); column=:ID)
        end

        @formulas begin
            p = logistic(a + b * z + η)
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.4, -0.2, 0.1, 0.3, 0.8],
        y = [0, 1, 0, 0, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))
    @test res isa FitResult
    @test NoLimits.get_chain(res) isa MCMCChains.Chains
end
