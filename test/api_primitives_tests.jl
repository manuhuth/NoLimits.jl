using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using FiniteDifferences
using LinearAlgebra
using Random
using Turing
using Optimization
using OptimizationOptimJL

# Method-developer API primitives (src/estimation/dev_api.jl) plus the Phase 0/1
# public renames. Reuses the shared fixtures; asserts each public primitive equals
# the internal it wraps (bit-identical where a pure cover) and the density identities.

# Extension-seam demo: a one-method custom curvature (diagonal of the exact Hessian).
struct APITestDiagCurvature <: NoLimits.AbstractCurvature end
function NoLimits.inner_curvature(::APITestDiagCurvature, dm, batch, θ, b, cc, cache, ws;
        ctx = "", tctx = nothing)
    H = NoLimits.inner_curvature(NoLimits.ExactHessianCurvature(), dm, batch, θ, b, cc,
        cache, ws; ctx = ctx, tctx = tctx)
    return Matrix(Diagonal(diag(H)))
end

# Protocol demo (plan Skeleton B): a new fixed-effects method defined only by delegating
# to `fit_fixed_effects` with a ridge `objective_term`.
struct RidgeMLE{O, K, A, L, U} <: NoLimits.FittingMethod
    λ::Float64
    optimizer::O
    optim_kwargs::K
    adtype::A
    lb::L
    ub::U
    ignore_model_bounds::Bool
end
function RidgeMLE(; λ = 1.0, optimizer = OptimizationOptimJL.LBFGS(), optim_kwargs = (;),
        adtype = Optimization.AutoForwardDiff(), lb = nothing, ub = nothing,
        ignore_model_bounds = false)
    return RidgeMLE(λ, optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds)
end
function NoLimits.fit_method(dm, m::RidgeMLE, args...; constants = NamedTuple(),
        penalty = NamedTuple(), extra_objective = nothing, ode_args = (),
        ode_kwargs = NamedTuple(), serialization = NoLimits.EnsembleSerial(),
        rng = Random.default_rng(), theta_0_untransformed = nothing,
        store_data_model = true)
    ridge = θu -> m.λ * sum(abs2, θu)
    return NoLimits.fit_fixed_effects(dm, m;
        objective_term = NoLimits._combine_add_terms(ridge, extra_objective),
        constants = constants, penalty = penalty, ode_args = ode_args,
        ode_kwargs = ode_kwargs, serialization = serialization,
        theta_0_untransformed = theta_0_untransformed, store_data_model = store_data_model,
        fit_args = args)
end

# Protocol demo (plan Skeleton A): a new marginal-RE method = a custom curvature plugged
# into `fit_laplace_family`. Reuses the diagonal curvature above; the result is a
# `FrequentistREResult`, so it is first-class across get_random_effects / get_loglikelihood.
struct APITestDiagonalLaplace <: NoLimits.FittingMethod
    base::NoLimits.Laplace
end
APITestDiagonalLaplace(; kwargs...) = APITestDiagonalLaplace(NoLimits.Laplace(; kwargs...))
function NoLimits.fit_method(
        dm, m::APITestDiagonalLaplace, args...; constants = NamedTuple(),
        constants_re = NamedTuple(), penalty = NamedTuple(), extra_objective = nothing,
        ode_args = (), ode_kwargs = NamedTuple(), serialization = NoLimits.EnsembleThreads(),
        rng = Random.default_rng(), theta_0_untransformed = nothing, store_data_model = true)
    fit_kwargs = (; constants, constants_re, penalty, ode_args, ode_kwargs, serialization,
        rng, theta_0_untransformed, store_data_model)
    return NoLimits.fit_laplace_family(
        dm, m.base, APITestDiagCurvature(), args, fit_kwargs,
        _ -> nothing; nan_recovery = m.base.nan_recovery, allow_bbo = true,
        constants = constants, constants_re = constants_re, penalty = penalty,
        ode_args = ode_args, ode_kwargs = ode_kwargs, serialization = serialization,
        rng = rng, theta_0_untransformed = theta_0_untransformed,
        store_data_model = store_data_model, extra_objective = extra_objective)
end

# Protocol demo (Skeleton C): a bespoke iterative estimator that keeps its OWN method type,
# packages the result with `build_fit_result`, and opts into Wald UQ via `uq_family`.
struct APITestClosedFormEM <: NoLimits.FittingMethod
    n_iter::Int
end
APITestClosedFormEM(; n_iter = 5) = APITestClosedFormEM(n_iter)
NoLimits.uq_family(::APITestClosedFormEM) = :wald_re
function NoLimits.fit_method(dm, m::APITestClosedFormEM, args...;
        constants_re = NamedTuple(), store_data_model = true,
        theta_0_untransformed = nothing, kwargs...)
    fe = NoLimits.get_fixed(NoLimits.get_model(dm))
    inv_transform = NoLimits.get_inverse_transform(fe)
    # honour Multistart / pooled_init starts (the theta_0_untransformed contract)
    θ = theta_0_untransformed === nothing ? NoLimits.get_θ0_untransformed(fe) :
        copy(theta_0_untransformed)
    θt0 = NoLimits.get_transform(fe)(θ)
    _, batches, cc = NoLimits.build_re_batch_infos(dm, constants_re)
    cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
    obj = 0.0
    for _ in 1:(m.n_iter)
        pm = NoLimits.posterior_moments(dm, θ)
        function negQ(θt_vec, _)
            θn = NoLimits.symmetrize_psd_parameters(dm,
                inv_transform(ComponentArray(θt_vec, getaxes(θt0))))
            acc = zero(eltype(θt_vec))
            for bi in eachindex(batches)
                mb, Σ = pm[bi]
                Σ === nothing && continue
                jl = NoLimits.joint_loglikelihood(dm, batches[bi], θn, mb;
                    const_cache = cc, cache = cache)
                H = NoLimits.joint_loglikelihood_hessian(dm, batches[bi], θn, mb;
                    const_cache = cc, cache = cache)
                acc += jl + 0.5 * tr(Σ * H)
            end
            return -acc
        end
        prob = OptimizationProblem(
            OptimizationFunction(negQ, Optimization.AutoForwardDiff()),
            collect(NoLimits.get_transform(fe)(θ)))
        sol = Optimization.solve(prob, OptimizationOptimJL.LBFGS(); maxiters = 30)
        θ = inv_transform(ComponentArray(sol.u, getaxes(θt0)))
        obj = -negQ(collect(NoLimits.get_transform(fe)(θ)), nothing)
    end
    return build_fit_result(dm, m, θ; kind = :frequentist_re, objective = obj,
        iterations = m.n_iter,
        eb_modes = NoLimits.empirical_bayes(dm, θ; constants_re = constants_re),
        store_data_model = store_data_model, fit_args = args)
end

# Protocol demo (Skeleton D): a custom Bayesian estimator that keeps its own method type and
# packages a posterior chain via the chain method of build_fit_result.
struct APITestBayes <: NoLimits.FittingMethod end

@testset "dev-API primitives" begin
    @testset "public names alias the internals (===)" begin
        @test NoLimits.symmetrize_psd_parameters === NoLimits._symmetrize_psd_params
        @test NoLimits.apply_constants! === NoLimits._apply_constants!
        @test NoLimits.penalty_value === NoLimits._penalty_value
        @test NoLimits.free_parameter_indices === NoLimits._free_idx
        @test NoLimits.merge_free_parameters === NoLimits._merge_free_into_full
        @test NoLimits.resolve_optimizer_bounds === NoLimits._resolve_optim_bounds
        @test NoLimits.build_re_batch_infos === NoLimits._build_re_batch_infos
        @test NoLimits.random_effect_value === NoLimits._re_value_from_b
        @test NoLimits.build_eta_individual === NoLimits._build_eta_ind
        @test NoLimits.eta_from_modes === NoLimits._eta_from_eb
        @test NoLimits.build_likelihood_cache === NoLimits.build_ll_cache
        @test NoLimits.build_batch_theta_context === NoLimits._build_theta_ctx
        @test NoLimits.LikelihoodCache === NoLimits._LLCache
        @test NoLimits.RELevelInfo === NoLimits._REInfo
        @test NoLimits.BatchThetaContext === NoLimits._LaplaceThetaCtx
    end

    @testset "batching currency + density kernels (scalar RE)" begin
        dm = fx_re_dm()
        θ = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
        _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
        @test !isempty(infos)
        batch = infos[1]
        @test NoLimits.get_batch_re_dim(batch) >= 1
        @test length(NoLimits.get_batch_individuals(batch)) >= 1
        li = NoLimits.get_batch_re_info(batch)[1]
        @test length(NoLimits.get_re_levels(li)) == length(NoLimits.get_re_ranges(li))

        b = fill(0.3, NoLimits.get_batch_re_dim(batch))
        # batch joint is a pure cover of _laplace_logf_batch -> bit-identical
        @test NoLimits.joint_loglikelihood(
            dm, batch, θ, b; const_cache = cc, cache = cache) ===
              NoLimits._laplace_logf_batch(dm, batch, θ, b, cc, cache)
        # joint == conditional + re_logprior
        j = NoLimits.joint_loglikelihood(dm, batch, θ, b; const_cache = cc, cache = cache)
        cll = NoLimits.conditional_loglikelihood(
            dm, batch, θ, b; const_cache = cc, cache = cache)
        rlp = NoLimits.re_logprior(dm, batch, θ, b; const_cache = cc, cache = cache)
        @test j≈cll + rlp atol=1e-10
        # ∇_b: ForwardDiff cover vs finite differences
        g = NoLimits.joint_loglikelihood_gradient(
            dm, batch, θ, b; const_cache = cc, cache = cache)
        gfd = FiniteDifferences.grad(central_fdm(5, 1),
            bb -> NoLimits.joint_loglikelihood(
                dm, batch, θ, bb; const_cache = cc, cache = cache),
            b)[1]
        @test g≈gfd atol=1e-6
        # ∇²_b: pure cover -> bit-identical, symmetric
        H = NoLimits.joint_loglikelihood_hessian(
            dm, batch, θ, b; const_cache = cc, cache = cache)
        @test H == NoLimits._laplace_hessian_b(dm, batch, θ, b, cc, cache, nothing, 1)
        @test H ≈ transpose(H)
        # population conditional is loglikelihood
        η_vec = NoLimits.eta_from_modes(
            dm, infos, [zeros(NoLimits.get_batch_re_dim(bi)) for bi in infos], cc, θ)
        @test NoLimits.conditional_loglikelihood(dm, θ, η_vec) ===
              NoLimits.loglikelihood(dm, θ, η_vec)
    end

    @testset "crossed design (ID x SITE) exercises n_b > 1" begin
        dm = fx_mg_dm()
        θ = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
        _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
        batch = argmax(NoLimits.get_batch_re_dim, infos)
        @test NoLimits.get_batch_re_dim(batch) >= 2
        b = fill(0.2, NoLimits.get_batch_re_dim(batch))
        j = NoLimits.joint_loglikelihood(dm, batch, θ, b; const_cache = cc, cache = cache)
        cll = NoLimits.conditional_loglikelihood(
            dm, batch, θ, b; const_cache = cc, cache = cache)
        rlp = NoLimits.re_logprior(dm, batch, θ, b; const_cache = cc, cache = cache)
        @test j≈cll + rlp atol=1e-10
        g = NoLimits.joint_loglikelihood_gradient(
            dm, batch, θ, b; const_cache = cc, cache = cache)
        gfd = FiniteDifferences.grad(central_fdm(5, 1),
            bb -> NoLimits.joint_loglikelihood(
                dm, batch, θ, bb; const_cache = cc, cache = cache),
            b)[1]
        @test g≈gfd atol=1e-6
    end

    @testset "solve_individual + obs_distributions (ODE) reproduce the likelihood" begin
        dm = fx_ode_dm()
        θ = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
        η = ComponentArray(η = 0.1)
        sol = NoLimits.solve_individual(dm, 1, θ, η)
        @test sol !== nothing
        od = NoLimits.obs_distributions(dm, 1, θ, η)
        yv = NoLimits.get_obs(NoLimits.get_series(NoLimits.get_individuals(dm)[1])).y
        manual = sum(logpdf(od[i].y, yv[i]) for i in eachindex(od) if yv[i] !== missing)
        @test NoLimits.conditional_loglikelihood(dm, 1, θ, η)≈manual atol=1e-9
    end

    @testset "hmm_filter_step! passes non-HMM through" begin
        d = Normal(0.0, 1.0)
        pr = Dict{Symbol, Any}()
        @test NoLimits.hmm_filter_step!(pr, :y, d, 0.5) === d
        @test isempty(pr)
    end

    @testset "empirical Bayes / posterior moments / Laplace marginal" begin
        dm = fx_re_dm()
        res = fx_laplace()
        θhat = NoLimits.get_params(res; scale = :untransformed)
        _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)

        @test NoLimits.EBEOptions() isa NoLimits.EBEOptions
        bstars = NoLimits.empirical_bayes(
            dm, θhat; serialization = NoLimits.EnsembleSerial())
        @test length(bstars) == length(infos)
        # EB modes maximize the joint: ∇_b ≈ 0
        maxg = maximum(eachindex(infos)) do bi
            g = NoLimits.joint_loglikelihood_gradient(
                dm, infos[bi], θhat, bstars[bi]; const_cache = cc, cache = cache)
            isempty(g) ? 0.0 : maximum(abs, g)
        end
        @test maxg < 1e-6

        # posterior_moments: Σ = (−H)⁻¹
        b1, Σ1 = NoLimits.posterior_moments(
            dm, θhat, infos[1], bstars[1]; const_cache = cc, cache = cache)
        H1 = NoLimits.joint_loglikelihood_hessian(
            dm, infos[1], θhat, bstars[1]; const_cache = cc, cache = cache)
        @test Σ1 ≈ transpose(Σ1)
        @test -H1 * Σ1≈Matrix(I, size(Σ1)...) atol=1e-6

        # laplace_marginal batch equals its assembly; pop reproduces −objective
        lm1 = NoLimits.laplace_marginal(
            dm, θhat, infos[1], bstars[1]; const_cache = cc, cache = cache)
        logf1 = NoLimits._laplace_logf_batch(dm, infos[1], θhat, bstars[1], cc, cache)
        ldn, _, _ = NoLimits._laplace_logdet_negH(
            dm, infos[1], θhat, bstars[1], cc, cache, nothing, 1)
        @test lm1 ==
              logf1 + (NoLimits.get_batch_re_dim(infos[1]) / 2) * log(2 * pi) - ldn / 2
        lmpop = NoLimits.laplace_marginal(
            dm, θhat; serialization = NoLimits.EnsembleSerial())
        @test lmpop≈-NoLimits.get_objective(res) atol=1e-3

        @test NoLimits.empirical_bayes(
            dm, θhat, 1; serialization = NoLimits.EnsembleSerial()) isa ComponentArray
    end

    @testset "quadrature marginal / Fisher info / posterior sampling" begin
        dm = fx_re_dm()
        θ = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
        _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)

        # ghq_marginal batch is a bit-identical cover of _ghq_batch_ll; pop is finite
        @test NoLimits.ghq_marginal(
            dm, θ, infos[1]; level = 5, const_cache = cc, cache = cache) ===
              NoLimits._ghq_batch_ll(dm, infos[1], θ, cc, cache, 5)
        @test isfinite(NoLimits.ghq_marginal(dm, θ; level = 5))

        # expected_information registry aliases
        d = Normal(1.0, 0.5)
        @test NoLimits.has_expected_information(d)
        @test NoLimits.expected_information(d) == NoLimits._focei_expected_information(d)
        @test NoLimits.outcome_parameters(d) == NoLimits._focei_params(d)
        @test NoLimits.dispersion_indices(d) == NoLimits._focei_dispersion_indices(d)

        # GHQ node grid + accessors
        g = NoLimits.build_sparse_grid(2, 3)
        @test g isa NoLimits.GHQuadratureNodes
        @test size(NoLimits.get_nodes(g), 1) == NoLimits.get_dimension(g) == 2
        @test length(NoLimits.get_logweights(g)) == length(NoLimits.get_signs(g)) ==
              size(NoLimits.get_nodes(g), 2)
        @test NoLimits.get_level(g) == 3

        # sample_random_effect_draws: Laplace-Gaussian IS (exact for the linear-Gaussian model)
        res = fx_laplace()
        θhat = NoLimits.get_params(res; scale = :untransformed)
        bstars = NoLimits.empirical_bayes(
            dm, θhat; serialization = NoLimits.EnsembleSerial())
        s = NoLimits.sample_random_effect_draws(
            dm, θhat, infos[1], bstars[1]; n_samples = 400,
            const_cache = cc, cache = cache, rng = MersenneTwister(1))
        @test s isa NoLimits.RandomEffectPosteriorSample
        D = NoLimits.get_draws(s)
        lw = NoLimits.get_log_weights(s)
        @test size(D) == (NoLimits.get_batch_re_dim(infos[1]), 400)
        w = exp.(lw .- maximum(lw))
        w ./= sum(w)
        @test D * w≈bstars[1] atol=0.15
        @test NoLimits.get_ess(s) > 100

        # sample_random_effect_draws :mcmc wraps Turing directly
        smc = NoLimits.sample_random_effect_draws(
            dm, θhat, infos[1], Float64[]; method = :mcmc,
            sampler = MH(), n_samples = 40, n_adapt = 10, const_cache = cc, cache = cache,
            rng = MersenneTwister(2))
        @test smc isa NoLimits.RandomEffectPosteriorSample
        @test size(NoLimits.get_draws(smc), 1) == NoLimits.get_batch_re_dim(infos[1])
        @test NoLimits.get_log_weights(smc) === nothing
        @test_throws ErrorException NoLimits.sample_random_effect_draws(
            dm, θhat, infos[1], Float64[]; method = :mcmc, const_cache = cc, cache = cache)
    end

    @testset "curvature seam" begin
        dm = fx_re_dm()
        θ = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
        _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
        b = fill(0.3, NoLimits.get_batch_re_dim(infos[1]))

        @test NoLimits._HessMode === NoLimits.AbstractCurvature
        @test NoLimits._ExactHess === NoLimits.ExactHessianCurvature
        @test NoLimits._FOCEIHess === NoLimits.FisherInformationCurvature

        # exact curvature via the seam is bit-identical to the internal Hessian
        @test NoLimits.joint_loglikelihood_hessian(dm, infos[1], θ, b; const_cache = cc,
            cache = cache, curvature = NoLimits.ExactHessianCurvature()) ==
              NoLimits._laplace_hessian_b(dm, infos[1], θ, b, cc, cache, nothing, 1)

        # FOCEI curvature routes through the seam; −H is PD by construction
        Hfoc = NoLimits.joint_loglikelihood_hessian(dm, infos[1], θ, b; const_cache = cc,
            cache = cache, curvature = NoLimits.FisherInformationCurvature(true))
        @test Hfoc == NoLimits.inner_curvature(NoLimits.FisherInformationCurvature(true),
            dm, infos[1], θ, b, cc, cache, NoLimits.CurvatureWorkspace())
        @test isposdef(-Hfoc)

        # curvature kwarg threads through posterior_moments / laplace_marginal
        _, Σf = NoLimits.posterior_moments(dm, θ, infos[1], b; const_cache = cc,
            cache = cache, curvature = NoLimits.FisherInformationCurvature(true))
        @test Σf ≈ transpose(Σf)
        @test isfinite(NoLimits.laplace_marginal(dm, θ, infos[1], b; const_cache = cc,
            cache = cache, curvature = NoLimits.FisherInformationCurvature(false)))

        # extension seam: a custom curvature plugs in via one `inner_curvature` method
        dmx = fx_mg_dm()
        θx = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dmx)))
        _, infosx, ccx = NoLimits.build_re_batch_infos(dmx, NamedTuple())
        cachex = NoLimits.build_likelihood_cache(dmx; force_saveat = true)
        bxb = argmax(NoLimits.get_batch_re_dim, infosx)
        bx = fill(0.1, NoLimits.get_batch_re_dim(bxb))
        Hfull = NoLimits.joint_loglikelihood_hessian(
            dmx, bxb, θx, bx; const_cache = ccx, cache = cachex)
        Hdiag = NoLimits.joint_loglikelihood_hessian(dmx, bxb, θx, bx; const_cache = ccx,
            cache = cachex, curvature = APITestDiagCurvature())
        @test size(Hfull, 1) >= 2
        @test Hdiag == Matrix(Diagonal(diag(Hfull)))
    end

    @testset "fixed-effects method protocol (RidgeMLE skeleton)" begin
        @test NoLimits.fit_method === NoLimits._fit_model
        @test NoLimits.fit_laplace_family === NoLimits._fit_laplace_family
        dm = fx_nore_dm()
        res = fit_model(dm, RidgeMLE(λ = 3.0); serialization = NoLimits.EnsembleSerial())
        @test NoLimits.get_converged(res) isa Bool
        @test NoLimits.get_params(res; scale = :untransformed) isa ComponentArray
        @test isfinite(NoLimits.get_objective(res))
        @test NoLimits.get_diagnostics(res) !== nothing
    end

    @testset "marginal-RE method protocol (DiagonalLaplace skeleton)" begin
        dm = fx_re_dm()
        res = fit_model(dm, APITestDiagonalLaplace(; optim_kwargs = (maxiters = 3,));
            serialization = NoLimits.EnsembleSerial())
        # a custom curvature method produces a FrequentistREResult -> first-class accessors
        @test NoLimits.get_result(res) isa NoLimits.FrequentistREResult
        @test NoLimits.get_random_effects(dm, res) isa NamedTuple
        @test isfinite(NoLimits.get_loglikelihood(dm, res))
        @test NoLimits.get_params(res; scale = :untransformed) isa ComponentArray
    end

    @testset "logabsdetjac (change of variables)" begin
        m = @Model begin
            @fixedEffects begin
                a = RealNumber(1.5, scale = :log)
                p = RealNumber(0.3, scale = :logit, lower = 0.0, upper = 1.0)
                c = RealNumber(0.7)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(c, a)
            end
        end
        fe = NoLimits.get_fixed(m)
        it = NoLimits.get_inverse_transform(fe)
        θt = NoLimits.get_transform(fe)(NoLimits.get_θ0_untransformed(fe))
        # log -> log(a); logit(0,1) -> log(σ(1-σ)); identity -> 0
        @test logabsdetjac(it, θt)≈log(1.5) + log(0.3 * 0.7) atol=1e-8
        g = ForwardDiff.gradient(
            z -> logabsdetjac(it, ComponentArray(z, getaxes(θt))), collect(θt))
        @test all(isfinite, g)
        # structured scales (cholesky/expm/stickbreak/stickbreakrows/lograterows/lie)
        # are covered in test/logabsdetjac_tests.jl
    end

    @testset "custom estimator: build_fit_result + uq_family" begin
        # built-in uq_family defaults are preserved
        @test NoLimits.uq_family(NoLimits.MLE()) == :wald_no_re
        @test NoLimits.uq_family(NoLimits.Laplace()) == :wald_re
        @test NoLimits.uq_family(NoLimits.MCMC()) == :chain
        # a custom method with no declared family has no built-in Wald UQ
        @test NoLimits.uq_family(RidgeMLE()) == :none
        @test NoLimits.uq_family(APITestClosedFormEM()) == :wald_re

        dm = fx_re_dm()
        res = fit_model(dm, APITestClosedFormEM(; n_iter = 5);
            serialization = NoLimits.EnsembleSerial())
        # first-class result that keeps its own method type
        @test res isa NoLimits.FitResult
        @test NoLimits.get_method(res) isa APITestClosedFormEM
        @test NoLimits.get_result(res) isa NoLimits.FrequentistREResult      # kind = :frequentist_re
        @test NoLimits.get_params(res; scale = :untransformed) isa ComponentArray
        @test NoLimits.get_params(res; scale = :transformed) isa ComponentArray
        @test NoLimits.get_random_effects(dm, res) isa NamedTuple
        @test isfinite(NoLimits.get_loglikelihood(dm, res))
        @test NoLimits.build_plot_cache(res) !== nothing
        # UQ works through the trait, without masquerading as a built-in method
        @test compute_uq(res; method = :wald, pseudo_inverse = true) isa NoLimits.UQResult

        # a custom method without uq_family raises an informative error (not a wrong answer)
        res_ridge = fit_model(fx_nore_dm(), RidgeMLE(; λ = 1.0);
            serialization = NoLimits.EnsembleSerial())
        @test_throws ErrorException compute_uq(res_ridge; method = :wald)
    end

    @testset "custom estimator: multistart/pooled_init/save + kind validation" begin
        dm = fx_re_dm()
        fe = NoLimits.get_fixed(NoLimits.get_model(dm))
        θ0 = NoLimits.get_θ0_untransformed(fe)

        # theta_0_untransformed is honoured: different starts -> different 1-step fits
        θa = copy(θ0)
        θa.a = θ0.a + 0.5
        r1 = fit_model(dm, APITestClosedFormEM(; n_iter = 1);
            theta_0_untransformed = θ0, serialization = NoLimits.EnsembleSerial())
        r2 = fit_model(dm, APITestClosedFormEM(; n_iter = 1);
            theta_0_untransformed = θa, serialization = NoLimits.EnsembleSerial())
        @test !(NoLimits.get_params(r1; scale = :untransformed) ≈
                NoLimits.get_params(r2; scale = :untransformed))

        # Multistart delivers its starts through the same kwarg
        ms = NoLimits.Multistart(dists = (; a = Normal(0.0, 1.0)),
            n_draws_requested = 2, n_draws_used = 2, rng = Random.Xoshiro(3))
        res_ms = fit_model(ms, dm, APITestClosedFormEM(; n_iter = 1);
            serialization = NoLimits.EnsembleSerial())
        objs = NoLimits.get_objective.(NoLimits.get_multistart_results(res_ms))
        @test length(objs) == 2 && length(unique(objs)) == 2

        # pooled_init warm start reaches the custom method
        res_pi = fit_model(dm, APITestClosedFormEM(; n_iter = 1); pooled_init = true,
            serialization = NoLimits.EnsembleSerial())
        @test isfinite(NoLimits.get_objective(res_pi))

        # save/load roundtrip (generic _strip_fitting_method covers custom methods)
        path = tempname() * ".jld2"
        NoLimits.save_fit(path, r1)
        r1b = NoLimits.load_fit(path; dm = dm)
        @test NoLimits.get_objective(r1b) ≈ NoLimits.get_objective(r1)
        @test NoLimits.get_params(r1b; scale = :untransformed) ≈
              NoLimits.get_params(r1; scale = :untransformed)

        # kind validation errors at build time, not deep in an accessor
        @test_throws ErrorException build_fit_result(dm, APITestClosedFormEM(), θ0;
            kind = :laplace, objective = 0.0)
        @test_throws ErrorException build_fit_result(dm, APITestClosedFormEM(), θ0;
            kind = :frequentist, objective = 0.0, eb_modes = [zeros(1)])
    end

    @testset "FitContext convenience tier" begin
        dm = fx_re_dm()
        ctx = build_fit_context(dm)
        θ = initial_parameters(ctx)
        _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
        cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
        b = zeros(NoLimits.get_batch_re_dim(infos[1]))

        # context calls are bit-identical to the explicit cache-threaded calls
        @test joint_loglikelihood(ctx, 1, θ, b) ===
              joint_loglikelihood(dm, infos[1], θ, b; const_cache = cc, cache = cache)
        @test conditional_loglikelihood(ctx, 1, θ, b) ===
              conditional_loglikelihood(
            dm, infos[1], θ, b; const_cache = cc, cache = cache)
        @test re_logprior(ctx, 1, θ, b) ===
              re_logprior(dm, infos[1], θ, b; const_cache = cc, cache = cache)
        @test joint_loglikelihood_hessian(ctx, 1, θ, b) ==
              joint_loglikelihood_hessian(
            dm, infos[1], θ, b; const_cache = cc, cache = cache)

        # population forms reuse the ctx caches and align with the batch structure
        pm = posterior_moments(ctx, θ)
        @test length(pm) == length(get_batch_infos(ctx))
        @test pm[1][2] isa Matrix
        @test isfinite(laplace_marginal(ctx, θ))
        @test isfinite(ghq_marginal(ctx, θ))
        s = sample_random_effect_draws(ctx, θ; n_samples = 30,
            rng = Random.MersenneTwister(1))
        @test length(s) == length(get_batch_infos(ctx))
        @test size(NoLimits.get_draws(s[1]), 2) == 30

        # optimize_parameters: natural-scale objective, transformed-scale solve
        modes = NoLimits.empirical_bayes(ctx, θ; rng = Random.MersenneTwister(2))
        θ̂, sol = optimize_parameters(ctx; θ_start = θ,
            optim_kwargs = (; iterations = 30)) do θn
            -sum(joint_loglikelihood(ctx, bi, θn, modes[bi])
            for bi in eachindex(get_batch_infos(ctx)))
        end
        @test θ̂ isa ComponentArray && isfinite(sol.objective)
        @test θ̂.σ > 0 && θ̂.ω > 0        # log-scale bounds respected via the transform

        # ctx build_fit_result fills eb_modes automatically for RE kinds
        res = build_fit_result(ctx, APITestClosedFormEM(), θ̂; kind = :frequentist_re,
            objective = sol.objective)
        @test NoLimits.get_eb_modes(NoLimits.get_result(res)) !== nothing
        @test NoLimits.get_random_effects(dm, res) isa NamedTuple
    end

    @testset "custom Bayesian estimator: build_fit_result(chain)" begin
        base = fx_mcmc()                       # built-in MCMC fit; reuse its chain + dm
        dm = NoLimits.get_data_model(base)
        chain = NoLimits.get_chain(base)

        res = build_fit_result(dm, APITestBayes(), chain;
            sampler = NoLimits.get_sampler(base), n_samples = NoLimits.get_n_samples(base))

        # first-class Bayesian result that keeps its own method type
        @test res isa NoLimits.FitResult
        @test NoLimits.get_method(res) isa APITestBayes
        @test NoLimits.get_result(res) isa NoLimits.MCMCResult
        @test NoLimits.get_chain(res) === chain
        @test NoLimits.get_observed(res) !== nothing
        # summarize reports Bayesian inference (routed on the result, not the method type)
        @test NoLimits.summarize(res).inference == :bayesian
        # chain UQ works without the method being a built-in MCMC/VI
        @test compute_uq(res; method = :chain) isa NoLimits.UQResult
    end
end
