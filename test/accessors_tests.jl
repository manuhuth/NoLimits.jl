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
    # get_random_effects works for every RE estimator (all share LaplaceResult-style EB modes).
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
end
