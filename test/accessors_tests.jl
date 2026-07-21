using Test
using NoLimits
using Distributions
using MCMCChains
using DataFrames
using ComponentArrays
using Turing

const NL = NoLimits

# Accessors are exercised against the shared fixture fits (built once in
# fixtures.jl) instead of re-fitting each method here.

@testset "Accessors: fixed-effects methods (MLE / MAP)" begin
    res_mle = fx_mle()
    @test NL.get_params(res_mle; scale = :untransformed) isa ComponentArray
    @test NL.get_converged(res_mle) isa Bool
    @test_throws ErrorException NL.get_chain(res_mle)

    # Without a stored DataModel, get_loglikelihood must error.
    res_nostore = fit_model(
        fx_nore_dm(), NL.MLE(; optim_kwargs = (maxiters = 2,)); store_data_model = false)
    @test_throws ErrorException NL.get_loglikelihood(res_nostore)

    @test_throws ErrorException NL.get_chain(fx_map())
end

@testset "Accessors: MCMC" begin
    res = fx_mcmc()
    @test NL.get_chain(res) isa MCMCChains.Chains
    @test NL.get_observed(res).y == fx_nore_df().y
    @test NL.get_sampler(res) isa Any
    @test NL.get_n_samples(res) == 20
    @test_throws ErrorException NL.get_loglikelihood(res)
end

@testset "Accessors: random-effects methods" begin
    # get_random_effects works for every RE estimator (all share FrequentistREResult-style EB modes).
    for res in (fx_laplace(), fx_focei(), fx_ghq(), fx_mcem(), fx_saem())
        @test !isempty(NL.get_random_effects(res))
    end
    @test fx_mcem().result.eb_modes !== nothing
    @test fx_saem().result.eb_modes !== nothing

    # sample_random_effects returns n_samples × per-level rows, tagged by :sample.
    for (res, n, kw) in ((fx_laplace(), 5, ()),
        (fx_mcem(), 4, (; n_adapt = 2)),
        (fx_saem(), 3, (; n_adapt = 2)))
        base = nrow(NL.get_random_effects(res).η)
        s = NL.sample_random_effects(res; n_samples = n, kw...)
        @test !isempty(s)
        @test :sample in propertynames(s.η)
        @test nrow(s.η) == n * base
    end
    @test sort(unique(NL.sample_random_effects(fx_laplace(); n_samples = 5).η.sample)) ==
          collect(1:5)

    # get_random_effect_distribution rebuilds the fitted population distribution p(b | θ̂).
    d_re = NL.get_random_effect_distribution(fx_laplace(), :η)
    @test d_re isa Normal
    @test_throws ErrorException NL.get_random_effect_distribution(fx_laplace(), :nope)
    @test_throws ErrorException NL.get_random_effect_distribution(fx_laplace(), :η;
        individual = 0)
end

@testset "Accessors: random-effects covariate usage" begin
    # Bespoke multi-RE model: usage must separate covariate-free and per-covariate REs.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end

        @covariates begin
            t = Covariate()
            age = ConstantCovariate()
            age2 = ConstantCovariate()
        end

        @randomEffects begin
            η0 = RandomEffect(Normal(0.0, 0.5); column = :ID)
            ηA = RandomEffect(Normal(b * age, 0.5); column = :ID)
            ηt = RandomEffect(Normal(age2, 0.5); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η0 + ηA + ηt, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        age = [30.0, 30.0, 40.0, 40.0],
        age2 = [30.0, 30.0, 40.0, 40.0],
        y = [0.1, 0.2, 0.0, 0.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    usage_dm = get_re_covariate_usage(dm)
    @test usage_dm.η0 == Symbol[]
    @test usage_dm.ηA == [:age]
    @test usage_dm.ηt == [:age2]

    # FitResult accessor agrees with the DataModel one (shared covariate-RE fixture).
    usage_fx = get_re_covariate_usage(fx_recov_dm())
    @test usage_fx.η == [:Age]
    @test get_re_covariate_usage(fx_recov_laplace()) == usage_fx
end
