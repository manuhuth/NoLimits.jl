using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using Turing

@testset "Discrete-time HMM transition matrix is used" begin
    emissions = (Bernoulli(0.95), Bernoulli(0.05))
    init = Categorical([1.0, 0.0])

    hmm_stay = DiscreteTimeDiscreteStatesHMM([1.0 0.0; 0.0 1.0], emissions, init)
    hmm_flip = DiscreteTimeDiscreteStatesHMM([0.0 1.0; 1.0 0.0], emissions, init)

    p_stay = probabilities_hidden_states(hmm_stay)
    p_flip = probabilities_hidden_states(hmm_flip)

    @test isapprox(p_stay, [1.0, 0.0]; rtol=0.0, atol=1e-12)
    @test isapprox(p_flip, [0.0, 1.0]; rtol=0.0, atol=1e-12)
    @test pdf(hmm_stay, 1) > pdf(hmm_flip, 1)

    post_flip = posterior_hidden_states(hmm_flip, 1)
    @test post_flip[2] > post_flip[1]
end

@testset "Discrete-time HMM loglikelihood + update" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @formulas begin
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            P = [0.9 0.1;
                 0.2 0.8]
            y ~ DiscreteTimeDiscreteStatesHMM(P,
                                              (Bernoulli(p1), Bernoulli(p2)),
                                              Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    @test isfinite(ll)
end

@testset "Discrete-time HMM ForwardDiff" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
        end

        @formulas begin
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            P = [0.9 0.1;
                 0.2 0.8]
            y ~ DiscreteTimeDiscreteStatesHMM(P,
                                              (Bernoulli(p1), Bernoulli(p2)),
                                              Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
end

@testset "Discrete-time HMM MLE/MAP/MCMC" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p1_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
            p2_r = RealNumber(0.0, prior=Normal(0.0, 1.0))
        end

        @formulas begin
            p1 = 0.8 / (1 + exp(-p1_r)) + 0.1
            p2 = 0.8 / (1 + exp(-p2_r)) + 0.1
            P = [0.9 0.1;
                 0.2 0.8]
            y ~ DiscreteTimeDiscreteStatesHMM(P,
                                              (Bernoulli(p1), Bernoulli(p2)),
                                              Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs=(; iterations=5)))
    @test res_mle isa FitResult
    @test isfinite(NoLimits.get_objective(res_mle))

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs=(; iterations=5)))
    @test res_map isa FitResult
    @test isfinite(NoLimits.get_objective(res_map))

    res_mcmc = fit_model(dm, NoLimits.MCMC(; sampler=MH(), turing_kwargs=(n_samples=20, n_adapt=0, progress=false)))
    res_mcmc = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains
end
