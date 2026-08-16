using Test
using NoLimits
using MCMCChains
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

    @test isapprox(p_stay, [1.0, 0.0]; rtol = 0.0, atol = 1.0e-12)
    @test isapprox(p_flip, [0.0, 1.0]; rtol = 0.0, atol = 1.0e-12)
    @test pdf(hmm_stay, 1) > pdf(hmm_flip, 1)

    post_flip = posterior_hidden_states(hmm_flip, 1)
    @test post_flip[2] > post_flip[1]
end

@testset "Discrete-time HMM loglikelihood uses recursive filtering" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            P = [
                0.6 0.4 0.0;
                0.0 0.7 0.3;
                0.0 0.0 1.0
            ]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (
                    Categorical([1.0, 0.0, 0.0]),
                    Categorical([0.0, 1.0, 0.0]),
                    Categorical([0.0, 0.0, 1.0]),
                ),
                Categorical([1.0, 0.0, 0.0])
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [2, 2, 3]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    dist = DiscreteTimeDiscreteStatesHMM(
        [0.6 0.4 0.0; 0.0 0.7 0.3; 0.0 0.0 1.0],
        (
            Categorical([1.0, 0.0, 0.0]),
            Categorical([0.0, 1.0, 0.0]),
            Categorical([0.0, 0.0, 1.0]),
        ),
        Categorical([1.0, 0.0, 0.0])
    )
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1.0e-12)
end

@testset "Discrete-time HMM missing observations still propagate hidden state" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            P = [
                0.6 0.4 0.0;
                0.0 0.7 0.3;
                0.0 0.0 1.0
            ]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (
                    Categorical([1.0, 0.0, 0.0]),
                    Categorical([0.0, 1.0, 0.0]),
                    Categorical([0.0, 0.0, 1.0]),
                ),
                Categorical([1.0, 0.0, 0.0])
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = Union{Missing, Int}[2, missing, 3]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    dist = DiscreteTimeDiscreteStatesHMM(
        [0.6 0.4 0.0; 0.0 0.7 0.3; 0.0 0.0 1.0],
        (
            Categorical([1.0, 0.0, 0.0]),
            Categorical([0.0, 1.0, 0.0]),
            Categorical([0.0, 0.0, 1.0]),
        ),
        Categorical([1.0, 0.0, 0.0])
    )
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1.0e-12)
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
            P = [
                0.9 0.1;
                0.2 0.8
            ]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4])
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0, 1, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
end

@testset "Discrete-time HMM MLE/MAP/MCMC/VI" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p1_r = RealNumber(0.0, prior = Normal(0.0, 1.0))
            p2_r = RealNumber(0.0, prior = Normal(0.0, 1.0))
        end

        @formulas begin
            p1 = 0.8 / (1 + exp(-p1_r)) + 0.1
            p2 = 0.8 / (1 + exp(-p2_r)) + 0.1
            P = [
                0.9 0.1;
                0.2 0.8
            ]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4])
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 5)))
    @test res_mle isa FitResult

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs = (; iterations = 5)))
    @test res_map isa FitResult

    res_mcmc = fit_model(
        dm,
        NoLimits.MCMC(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
        )
    )
    res_mcmc = fit_model(
        dm, NoLimits.MCMC(; turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false))
    )
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains

    res_vi = fit_model(dm, NoLimits.VI(; turing_kwargs = (max_iter = 10, progress = false)))
    @test res_vi isa FitResult
end

@testset "HMM constructor and quantile validation" begin
    E = (Normal(0.0, 1.0), Normal(3.0, 1.0))
    C = Categorical([0.5, 0.5])
    @test_throws ErrorException DiscreteTimeDiscreteStatesHMM([1.2 -0.2; 0.5 0.5], E, C)
    @test_throws ErrorException DiscreteTimeDiscreteStatesHMM([1.0 1.0; 1.0 1.0], E, C)
    @test_throws ErrorException DiscreteTimeDiscreteStatesHMM([NaN 0.0; 0.0 1.0], E, C)
    @test_throws ErrorException DiscreteTimeDiscreteStatesHMM([0.0 0.0; 0.0 0.0], E, C)
    @test_throws ErrorException ContinuousTimeDiscreteStatesHMM(
        [-1.0 -1.0; 2.0 -2.0], E, C, 1.0
    )
    @test_throws ErrorException ContinuousTimeDiscreteStatesHMM(
        [-1.0 2.0; 2.0 -2.0], E, C, 1.0
    )
    @test_throws ErrorException ContinuousTimeDiscreteStatesHMM(
        [-1.0 1.0; 2.0 -2.0], E, C, -1.0
    )

    # An observation impossible under every state must not poison the filter with NaNs.
    far = DiscreteTimeDiscreteStatesHMM(
        [1.0 0.0; 0.0 1.0], (Normal(0.0, 0.1), Normal(10.0, 0.1)), C
    )
    @test all(isfinite, posterior_hidden_states(far, 1000.0))

    # Quantiles of a discrete mixture must land on the support, not between its atoms.
    disc = DiscreteTimeDiscreteStatesHMM(
        [1.0 0.0; 0.0 1.0], (Categorical([1.0, 0.0]), Categorical([0.0, 1.0])), C
    )
    @test quantile(disc, 0.5) in (1, 2)
    @test_throws DomainError quantile(disc, -0.1)
    @test_throws DomainError quantile(disc, 1.1)

    # Continuous emissions still invert the mixture CDF.
    cont = DiscreteTimeDiscreteStatesHMM([0.8 0.2; 0.1 0.9], E, C)
    @test cdf(cont, quantile(cont, 0.5)) ≈ 0.5 atol = 1.0e-6

    mv = MVDiscreteTimeDiscreteStatesHMM(
        [0.8 0.2; 0.1 0.9],
        ((Normal(0.0, 1.0), Normal(0.0, 1.0)), (Normal(3.0, 1.0), Normal(3.0, 1.0))), C
    )
    @test_throws ErrorException logpdf(mv, [1.0])
end

@testset "HMM value support, state dims, and missing observations" begin
    C = Categorical([0.5, 0.5])
    P = [0.8 0.2; 0.1 0.9]
    Q = [-1.0 1.0; 2.0 -2.0]
    Econt = (Normal(0.0, 1.0), Normal(3.0, 1.0))
    Edisc = (Poisson(1.0), Poisson(5.0))
    Emix = (Poisson(1.0), Normal(3.0, 1.0))

    # Value support follows the emissions: all-discrete is Discrete, mixed is Continuous.
    @test DiscreteTimeDiscreteStatesHMM(P, Edisc, C) isa DiscreteUnivariateDistribution
    @test DiscreteTimeDiscreteStatesHMM(P, Econt, C) isa ContinuousUnivariateDistribution
    @test DiscreteTimeDiscreteStatesHMM(P, Emix, C) isa ContinuousUnivariateDistribution
    @test ContinuousTimeDiscreteStatesHMM(Q, Edisc, C, 1.0) isa
        DiscreteUnivariateDistribution
    @test ContinuousTimeDiscreteStatesHMM(Q, Econt, C, 1.0) isa
        ContinuousUnivariateDistribution
    mv_disc = MVDiscreteTimeDiscreteStatesHMM(
        P, ((Poisson(1.0), Poisson(2.0)), (Poisson(3.0), Poisson(4.0))), C
    )
    mv_cont = MVContinuousTimeDiscreteStatesHMM(
        Q, ((Normal(), Normal()), (Normal(), Normal())), C, 1.0
    )
    @test mv_disc isa DiscreteMultivariateDistribution
    @test mv_cont isa ContinuousMultivariateDistribution

    # Likelihood values are untouched by the support parameterization.
    @test logpdf(DiscreteTimeDiscreteStatesHMM(P, Econt, C), 0.4) ≈
        log(0.45 * pdf(Normal(0.0, 1.0), 0.4) + 0.55 * pdf(Normal(3.0, 1.0), 0.4))

    # State counts are validated at construction, on both DT and CT.
    @test_throws ArgumentError DiscreteTimeDiscreteStatesHMM(P, (Normal(),), C)
    @test_throws ArgumentError ContinuousTimeDiscreteStatesHMM(
        Q, (Normal(), Normal(), Normal()), C, 1.0
    )
    @test_throws ArgumentError ContinuousTimeDiscreteStatesHMM(
        Q, Econt, Categorical([0.3, 0.3, 0.4]), 1.0
    )
    @test_throws ArgumentError ContinuousTimeDiscreteStatesHMM(
        [-1.0 1.0 0.0; 2.0 -2.0 0.0], Econt, C, 1.0
    )

    # A missing observation contributes nothing and leaves the propagated prior intact.
    omm = DiscreteTimeObservedStatesMarkovModel(P, C)
    ct_omm = ContinuousTimeObservedStatesMarkovModel(Q, C, 1.0)
    for d in (
            DiscreteTimeDiscreteStatesHMM(P, Econt, C),
            ContinuousTimeDiscreteStatesHMM(Q, Edisc, C, 1.0),
            mv_disc, mv_cont, omm, ct_omm, coarsed(omm), coarsed(ct_omm),
        )
        @test logpdf(d, missing) == 0.0
        @test pdf(d, missing) == 1.0
        @test posterior_hidden_states(d, missing) ≈ probabilities_hidden_states(d)
    end
    @test posterior_hidden_states(coarsed(omm), missing) ≈
        posterior_hidden_states(omm, missing)

    # coarsed delegates scalar summaries instead of iterating the wrapper.
    @test quantile(coarsed(omm), 0.5) == quantile(omm, 0.5)
    @test median(coarsed(omm)) == median(omm)
end
