using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using Turing

@testset "HMM DataModel + loglikelihood" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log)
            λ21_r = RealNumber(0.1, scale=:log)
            p1_r  = RealNumber(0.0)
            p2_r  = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    @test isfinite(ll)
end

@testset "HMM ForwardDiff" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log)
            λ21_r = RealNumber(0.1, scale=:log)
            p1_r  = RealNumber(0.0)
            p2_r  = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = [0, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
end

@testset "HMM MLE optimization" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log, prior = LogNormal(0.01, 1.0))
            λ21_r = RealNumber(0.1, scale=:log, prior = LogNormal(0.01, 1.0))
            p1_r  = RealNumber(0.0, prior = Normal(0.0, 1.0))
            p2_r  = RealNumber(0.0, prior = Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res_mle = fit_model(dm, NoLimits.MLE())
    res_map = fit_model(dm, NoLimits.MAP())
    res_mcmc = fit_model(dm, NoLimits.MCMC(sampler=MH(), turing_kwargs=(; n_samples=15, n_adapt=5)))


    @test res_mle isa FitResult
    @test isfinite(NoLimits.get_objective(res_mle))
    @test res_map isa FitResult
    @test isfinite(NoLimits.get_objective(res_map))
    @test res_mcmc isa FitResult

end
