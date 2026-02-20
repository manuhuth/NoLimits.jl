using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using LinearAlgebra
using Random
using Turing

# ---------------------------------------------------------------------------
# Block A: Standalone struct tests
# ---------------------------------------------------------------------------

@testset "MVDiscreteTimeHMM: constructor" begin
    T  = [0.9 0.1; 0.2 0.8]
    em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([0.6, 0.4])

    hmm = MVDiscreteTimeDiscreteStatesHMM(T, em, init)
    @test hmm.n_states   == 2
    @test hmm.n_outcomes == 2
    @test Base.length(hmm) == 2

    # Joint MvNormal emission mode
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm_mv = MVDiscreteTimeDiscreteStatesHMM(T, em_mv, init)
    @test hmm_mv.n_states   == 2
    @test hmm_mv.n_outcomes == 2
end

@testset "MVDiscreteTimeHMM: constructor validation" begin
    T    = [0.9 0.1; 0.2 0.8]
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([0.6, 0.4])

    # Non-square transition matrix
    @test_throws ErrorException MVDiscreteTimeDiscreteStatesHMM(
        [1.0 0.0 0.0; 0.0 1.0 0.0], em, init)

    # Wrong number of emission elements (3 for n_states=2)
    @test_throws ErrorException MVDiscreteTimeDiscreteStatesHMM(
        T, (em..., (Normal(), Normal())), init)

    # Wrong initial_dist size (3 categories for n_states=2)
    @test_throws ErrorException MVDiscreteTimeDiscreteStatesHMM(
        T, em, Categorical([1/3, 1/3, 1/3]))

    # Mismatched n_outcomes across states: state 1 has 2, state 2 has 1
    @test_throws ErrorException MVDiscreteTimeDiscreteStatesHMM(
        T,
        ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0),)),
        init)
end

@testset "MVDiscreteTimeHMM: probabilities_hidden_states" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([1.0, 0.0])

    # Identity transition: state distribution is preserved
    hmm_stay = MVDiscreteTimeDiscreteStatesHMM([1.0 0.0; 0.0 1.0], em, init)
    @test isapprox(probabilities_hidden_states(hmm_stay), [1.0, 0.0]; atol=1e-12)

    # Deterministic flip: state 1 becomes state 2
    hmm_flip = MVDiscreteTimeDiscreteStatesHMM([0.0 1.0; 1.0 0.0], em, init)
    @test isapprox(probabilities_hidden_states(hmm_flip), [0.0, 1.0]; atol=1e-12)
end

@testset "MVDiscreteTimeHMM: logpdf — independent emissions" begin
    T    = [0.9 0.1; 0.2 0.8]
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([0.6, 0.4])
    hmm  = MVDiscreteTimeDiscreteStatesHMM(T, em, init)
    y    = [0.1, 2.1]

    lp = logpdf(hmm, y)
    @test isfinite(lp)
    @test isapprox(exp(lp), pdf(hmm, y); atol=1e-12)

    # Different y gives different logpdf
    @test logpdf(hmm, [3.1, -0.9]) != lp

    # Manual check: p(y|state k) = p(y1|state k) * p(y2|state k) under independence
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(em[k][1], y[1]) * pdf(em[k][2], y[2]) for k in 1:2]
    @test isapprox(exp(lp), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVDiscreteTimeHMM: logpdf — joint MvNormal emissions" begin
    T     = [0.9 0.1; 0.2 0.8]
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    init  = Categorical([0.6, 0.4])
    hmm   = MVDiscreteTimeDiscreteStatesHMM(T, em_mv, init)
    y     = [0.1, 2.1]

    lp = logpdf(hmm, y)
    @test isfinite(lp)
    @test isapprox(exp(lp), pdf(hmm, y); atol=1e-12)

    # Manual check
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(em_mv[k], y) for k in 1:2]
    @test isapprox(exp(lp), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVDiscreteTimeHMM: missing — independent, partial" begin
    T    = [0.9 0.1; 0.2 0.8]
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm  = MVDiscreteTimeDiscreteStatesHMM(T, em, Categorical([0.6, 0.4]))

    lp_partial = logpdf(hmm, [0.1, missing])
    @test isfinite(lp_partial)

    # Partial missing ≠ full observation
    @test lp_partial != logpdf(hmm, [0.1, 2.1])

    # Equals the logpdf of a mixture over the first outcome only
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(em[k][1], 0.1) for k in 1:2]
    @test isapprox(exp(lp_partial), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVDiscreteTimeHMM: missing — independent, all missing" begin
    T   = [0.9 0.1; 0.2 0.8]
    em  = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm = MVDiscreteTimeDiscreteStatesHMM(T, em, Categorical([0.6, 0.4]))

    # All emissions contribute 0 to log-likelihood → logsumexp(log p_hidden) = log(1) = 0
    @test isapprox(logpdf(hmm, [missing, missing]), 0.0; atol=1e-10)
end

@testset "MVDiscreteTimeHMM: missing — MvNormal, partial" begin
    T     = [0.9 0.1; 0.2 0.8]
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm   = MVDiscreteTimeDiscreteStatesHMM(T, em_mv, Categorical([0.6, 0.4]))

    lp_partial = logpdf(hmm, [0.1, missing])
    @test isfinite(lp_partial)

    # Manual: marginal of MvNormal(μ, I) over index 1 is Normal(μ[1], 1.0)
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(Normal(em_mv[k].μ[1], 1.0), 0.1) for k in 1:2]
    @test isapprox(exp(lp_partial), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVDiscreteTimeHMM: missing — MvNormal, all missing" begin
    T     = [0.9 0.1; 0.2 0.8]
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm   = MVDiscreteTimeDiscreteStatesHMM(T, em_mv, Categorical([0.6, 0.4]))

    @test isapprox(logpdf(hmm, [missing, missing]), 0.0; atol=1e-10)
end

@testset "MVDiscreteTimeHMM: missing — non-MvNormal joint throws error" begin
    T    = [0.9 0.1; 0.2 0.8]
    init = Categorical([0.6, 0.4])

    # Dirichlet is Distribution{Multivariate} but not MvNormal
    em_bad = (Dirichlet([1.0, 1.0]), Dirichlet([2.0, 2.0]))
    hmm    = MVDiscreteTimeDiscreteStatesHMM(T, em_bad, init)

    @test_throws ErrorException logpdf(hmm, [0.3, missing])
end

@testset "MVDiscreteTimeHMM: posterior_hidden_states" begin
    T    = [0.9 0.1; 0.2 0.8]
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm  = MVDiscreteTimeDiscreteStatesHMM(T, em, Categorical([0.6, 0.4]))

    # All missing: posterior equals prior
    p_prior = probabilities_hidden_states(hmm)
    @test isapprox(posterior_hidden_states(hmm, [missing, missing]), p_prior; atol=1e-10)

    # Observation at state-1 means → posterior favours state 1
    post1 = posterior_hidden_states(hmm, [0.0, 2.0])
    @test post1[1] > post1[2]

    # Observation at state-2 means → posterior favours state 2
    post2 = posterior_hidden_states(hmm, [3.0, -1.0])
    @test post2[2] > post2[1]
end

@testset "MVDiscreteTimeHMM: rand" begin
    T    = [0.9 0.1; 0.2 0.8]
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm  = MVDiscreteTimeDiscreteStatesHMM(T, em, Categorical([0.6, 0.4]))

    s = rand(MersenneTwister(42), hmm)
    @test s isa Vector
    @test length(s) == 2
    @test all(isfinite, s)

    # Joint MvNormal mode
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm_mv = MVDiscreteTimeDiscreteStatesHMM(T, em_mv, Categorical([0.6, 0.4]))
    s_mv = rand(MersenneTwister(42), hmm_mv)
    @test s_mv isa Vector
    @test length(s_mv) == 2
    @test all(isfinite, s_mv)
end

@testset "MVDiscreteTimeHMM: mean, cov, var" begin
    T    = [0.9 0.1; 0.2 0.8]
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm  = MVDiscreteTimeDiscreteStatesHMM(T, em, Categorical([0.6, 0.4]))

    m = mean(hmm)
    @test m isa Vector
    @test length(m) == 2
    @test all(isfinite, m)

    C = cov(hmm)
    @test C isa Matrix
    @test size(C) == (2, 2)
    @test isapprox(C, C'; atol=1e-10)  # symmetric

    v = var(hmm)
    @test v isa Vector
    @test length(v) == 2
    @test isapprox(v, diag(C); atol=1e-10)

    # Joint MvNormal mode
    em_mv  = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
              MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm_mv = MVDiscreteTimeDiscreteStatesHMM(T, em_mv, Categorical([0.6, 0.4]))
    C_mv   = cov(hmm_mv)
    @test C_mv isa Matrix
    @test size(C_mv) == (2, 2)
    @test isapprox(C_mv, C_mv'; atol=1e-10)
    @test isapprox(var(hmm_mv), diag(C_mv); atol=1e-10)
end

@testset "MVDiscreteTimeHMM: ForwardDiff through logpdf" begin
    T    = [0.9 0.1; 0.2 0.8]
    init = Categorical([0.6, 0.4])
    y    = [0.5, 1.5]

    # Differentiate w.r.t. the 4 emission means
    f = x -> begin
        em = ((Normal(x[1], 1.0), Normal(x[2], 0.5)), (Normal(x[3], 1.0), Normal(x[4], 0.5)))
        logpdf(MVDiscreteTimeDiscreteStatesHMM(T, em, init), y)
    end

    g = ForwardDiff.gradient(f, [0.0, 2.0, 3.0, -1.0])
    @test length(g) == 4
    @test all(isfinite, g)
end

# ---------------------------------------------------------------------------
# Block B: Integration tests
# ---------------------------------------------------------------------------

@testset "MVDiscreteTimeHMM: loglikelihood + DataModel" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            μ1 = RealNumber(0.0)
            μ2 = RealNumber(3.0)
        end
        @formulas begin
            P  = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t  = [0.0, 1.0, 2.0],
        y  = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ  = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    @test isfinite(ll)
end

@testset "MVDiscreteTimeHMM: ForwardDiff through full model" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            μ1 = RealNumber(0.0)
            μ2 = RealNumber(3.0)
        end
        @formulas begin
            P  = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t  = [0.0, 1.0, 2.0],
        y  = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]],
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0  = get_θ0_untransformed(dm.model.fixed.fixed)
    g   = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "MVDiscreteTimeHMM: MLE and MAP" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            μ1 = RealNumber(0.0, prior=Normal(0.0, 2.0))
            μ2 = RealNumber(3.0, prior=Normal(3.0, 2.0))
        end
        @formulas begin
            P  = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y  = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9],
              [0.0, 2.2], [2.9, -0.8], [0.1, 2.0]],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs=(; iterations=5)))
    @test res_mle isa FitResult
    @test isfinite(NoLimits.get_objective(res_mle))

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs=(; iterations=5)))
    @test res_map isa FitResult
    @test isfinite(NoLimits.get_objective(res_map))

    res_mcmc = fit_model(dm, NoLimits.MCMC(; sampler=MH(),
        turing_kwargs=(n_samples=15, n_adapt=5, progress=false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains
end
