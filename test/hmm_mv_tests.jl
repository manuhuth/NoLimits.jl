using Test
using NoLimits
using MCMCChains
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using LinearAlgebra
using Random
using Turing

function _recursive_hmm_loglikelihood(dists, ys)
    prior = nothing
    ll = 0.0
    for (dist, y) in zip(dists, ys)
        dist_use = prior === nothing ? dist : NoLimits._hmm_with_initial_probs(dist, prior)
        if ismissing(y)
            prior = probabilities_hidden_states(dist_use)
        else
            ll += logpdf(dist_use, y)
            prior = posterior_hidden_states(dist_use, y)
        end
    end
    return ll
end

# Discrete-time transition matrix and continuous-time rate matrix used throughout.
# Q has stationary distribution [2/3, 1/3].
const _T_MV = [0.9 0.1; 0.2 0.8]
const _Q_MV = [-1.0 1.0; 2.0 -2.0]

# Each variant: (testset label, transition/rate matrix, constructor closure).
mv_variants = (
    ("MVDiscreteTimeHMM", _T_MV,
        (A, em, init) -> MVDiscreteTimeDiscreteStatesHMM(A, em, init)),
    ("MVContinuousTimeHMM", _Q_MV,
        (A, em, init) -> MVContinuousTimeDiscreteStatesHMM(A, em, init, 1.0)))

# ---------------------------------------------------------------------------
# Block A: Standalone struct tests shared by both variants
# ---------------------------------------------------------------------------

for (label, M, mk) in mv_variants
    @testset "$label: constructor" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        init = Categorical([0.6, 0.4])

        hmm = mk(M, em, init)
        @test hmm.n_states == 2
        @test hmm.n_outcomes == 2
        @test Base.length(hmm) == 2
        if label == "MVContinuousTimeHMM"
            @test hmm.Δt == 1.0
        end

        # Joint MvNormal emission mode
        em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
            MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
        hmm_mv = mk(M, em_mv, init)
        @test hmm_mv.n_states == 2
        @test hmm_mv.n_outcomes == 2
    end

    @testset "$label: constructor validation" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        init = Categorical([0.6, 0.4])

        # Non-square transition/rate matrix
        @test_throws ErrorException mk([1.0 0.0 0.0; 0.0 1.0 0.0], em, init)

        # Wrong number of emission elements (3 for n_states=2)
        @test_throws ErrorException mk(M, (em..., (Normal(), Normal())), init)

        # Wrong initial_dist size (3 categories for n_states=2)
        @test_throws ErrorException mk(M, em, Categorical([1 / 3, 1 / 3, 1 / 3]))

        # Mismatched n_outcomes across states: state 1 has 2, state 2 has 1
        @test_throws ErrorException mk(
            M, ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0),)), init)
    end

    @testset "$label: logpdf — independent emissions" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        init = Categorical([0.6, 0.4])
        hmm = mk(M, em, init)
        y = [0.1, 2.1]

        lp = logpdf(hmm, y)
        @test isapprox(exp(lp), pdf(hmm, y); atol = 1e-12)

        # Different y gives different logpdf
        @test logpdf(hmm, [3.1, -0.9]) != lp

        # Manual check: p(y|state k) = p(y1|state k) * p(y2|state k) under independence
        p_h = probabilities_hidden_states(hmm)
        p_obs = [pdf(em[k][1], y[1]) * pdf(em[k][2], y[2]) for k in 1:2]
        @test isapprox(exp(lp), sum(p_h .* p_obs); atol = 1e-10)
    end

    @testset "$label: logpdf — joint MvNormal emissions" begin
        em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
            MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
        init = Categorical([0.6, 0.4])
        hmm = mk(M, em_mv, init)
        y = [0.1, 2.1]

        lp = logpdf(hmm, y)
        @test isapprox(exp(lp), pdf(hmm, y); atol = 1e-12)

        # Manual check
        p_h = probabilities_hidden_states(hmm)
        p_obs = [pdf(em_mv[k], y) for k in 1:2]
        @test isapprox(exp(lp), sum(p_h .* p_obs); atol = 1e-10)
    end

    @testset "$label: missing — independent, partial" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        hmm = mk(M, em, Categorical([0.6, 0.4]))

        lp_partial = logpdf(hmm, [0.1, missing])

        # Partial missing ≠ full observation
        @test lp_partial != logpdf(hmm, [0.1, 2.1])

        # Equals the logpdf of a mixture over the first outcome only
        p_h = probabilities_hidden_states(hmm)
        p_obs = [pdf(em[k][1], 0.1) for k in 1:2]
        @test isapprox(exp(lp_partial), sum(p_h .* p_obs); atol = 1e-10)
    end

    @testset "$label: missing — independent, all missing" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        hmm = mk(M, em, Categorical([0.6, 0.4]))

        # All emissions contribute 0 → logsumexp(log p_hidden) = log(1) = 0
        @test isapprox(logpdf(hmm, [missing, missing]), 0.0; atol = 1e-10)
    end

    @testset "$label: missing — MvNormal, partial" begin
        em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
            MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
        hmm = mk(M, em_mv, Categorical([0.6, 0.4]))

        lp_partial = logpdf(hmm, [0.1, missing])

        # Manual: marginal of MvNormal(μ, I) over index 1 is Normal(μ[1], 1.0)
        p_h = probabilities_hidden_states(hmm)
        p_obs = [pdf(Normal(em_mv[k].μ[1], 1.0), 0.1) for k in 1:2]
        @test isapprox(exp(lp_partial), sum(p_h .* p_obs); atol = 1e-10)
    end

    @testset "$label: missing — MvNormal, all missing" begin
        em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
            MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
        hmm = mk(M, em_mv, Categorical([0.6, 0.4]))

        @test isapprox(logpdf(hmm, [missing, missing]), 0.0; atol = 1e-10)
    end

    @testset "$label: missing — non-MvNormal joint throws error" begin
        init = Categorical([0.6, 0.4])

        # Dirichlet is Distribution{Multivariate} but not MvNormal
        em_bad = (Dirichlet([1.0, 1.0]), Dirichlet([2.0, 2.0]))
        hmm = mk(M, em_bad, init)

        @test_throws ErrorException logpdf(hmm, [0.3, missing])
    end

    @testset "$label: posterior_hidden_states" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        hmm = mk(M, em, Categorical([0.6, 0.4]))

        # All missing: posterior equals prior
        p_prior = probabilities_hidden_states(hmm)
        @test isapprox(
            posterior_hidden_states(hmm, [missing, missing]), p_prior; atol = 1e-10)

        # Observation at state-1 means → posterior favors state 1
        post1 = posterior_hidden_states(hmm, [0.0, 2.0])
        @test post1[1] > post1[2]

        # Observation at state-2 means → posterior favors state 2
        post2 = posterior_hidden_states(hmm, [3.0, -1.0])
        @test post2[2] > post2[1]
    end

    @testset "$label: rand" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        hmm = mk(M, em, Categorical([0.6, 0.4]))

        s = rand(MersenneTwister(42), hmm)
        @test s isa Vector
        @test length(s) == 2
        @test all(isfinite, s)

        # Joint MvNormal mode
        em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
            MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
        hmm_mv = mk(M, em_mv, Categorical([0.6, 0.4]))
        s_mv = rand(MersenneTwister(42), hmm_mv)
        @test s_mv isa Vector
        @test length(s_mv) == 2
        @test all(isfinite, s_mv)
    end

    @testset "$label: mean, cov, var" begin
        em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
        hmm = mk(M, em, Categorical([0.6, 0.4]))

        m = mean(hmm)
        @test m isa Vector
        @test length(m) == 2
        @test all(isfinite, m)

        C = cov(hmm)
        @test C isa Matrix
        @test size(C) == (2, 2)
        @test isapprox(C, C'; atol = 1e-10)  # symmetric

        v = var(hmm)
        @test v isa Vector
        @test length(v) == 2
        @test isapprox(v, diag(C); atol = 1e-10)

        # Joint MvNormal mode
        em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
            MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
        hmm_mv = mk(M, em_mv, Categorical([0.6, 0.4]))
        C_mv = cov(hmm_mv)
        @test C_mv isa Matrix
        @test size(C_mv) == (2, 2)
        @test isapprox(C_mv, C_mv'; atol = 1e-10)
        @test isapprox(var(hmm_mv), diag(C_mv); atol = 1e-10)
    end

    @testset "$label: ForwardDiff through logpdf" begin
        init = Categorical([0.6, 0.4])
        y = [0.5, 1.5]

        # Differentiate w.r.t. the 4 emission means
        f = x -> begin
            em = ((Normal(x[1], 1.0), Normal(x[2], 0.5)),
                (Normal(x[3], 1.0), Normal(x[4], 0.5)))
            logpdf(mk(M, em, init), y)
        end

        g = ForwardDiff.gradient(f, [0.0, 2.0, 3.0, -1.0])
        @test length(g) == 4
        @test all(isfinite, g)
    end
end

# ---------------------------------------------------------------------------
# Block A (variant-specific): hidden-state propagation
# ---------------------------------------------------------------------------

@testset "MVDiscreteTimeHMM: probabilities_hidden_states" begin
    em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([1.0, 0.0])

    # Identity transition: state distribution is preserved
    hmm_stay = MVDiscreteTimeDiscreteStatesHMM([1.0 0.0; 0.0 1.0], em, init)
    @test isapprox(probabilities_hidden_states(hmm_stay), [1.0, 0.0]; atol = 1e-12)

    # Deterministic flip: state 1 becomes state 2
    hmm_flip = MVDiscreteTimeDiscreteStatesHMM([0.0 1.0; 1.0 0.0], em, init)
    @test isapprox(probabilities_hidden_states(hmm_flip), [0.0, 1.0]; atol = 1e-12)
end

@testset "MVContinuousTimeHMM: probabilities_hidden_states" begin
    em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([1.0, 0.0])

    # Zero Δt: state distribution is unchanged
    hmm_zero = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 0.0)
    p_zero = probabilities_hidden_states(hmm_zero)
    @test isapprox(p_zero, [1.0, 0.0]; atol = 1e-6)

    # Large Δt: approaches stationary distribution [2/3, 1/3]
    hmm_long = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 50.0)
    p_long = probabilities_hidden_states(hmm_long)
    @test isapprox(p_long[1], 2 / 3; atol = 1e-4)
    @test isapprox(p_long[2], 1 / 3; atol = 1e-4)
end

@testset "MVContinuousTimeHMM: Δt affects state probabilities" begin
    em = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([1.0, 0.0])

    hmm_small = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 0.01)
    hmm_large = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 5.0)

    p_small = probabilities_hidden_states(hmm_small)
    p_large = probabilities_hidden_states(hmm_large)

    # Small Δt: still close to initial [1, 0]
    @test p_small[1] > 0.9

    # Large Δt: more mixed — first state probability has decreased toward 2/3
    @test p_large[1] < p_small[1]
end

# ---------------------------------------------------------------------------
# Block B: Integration tests — discrete-time variant
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
            P = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
end

@testset "MVDiscreteTimeHMM: loglikelihood uses recursive filtering" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            dummy = RealNumber(0.0)
        end
        @formulas begin
            P = [0.6 0.4 0.0;
                 0.0 0.7 0.3;
                 0.0 0.0 1.0]
            e1 = (Categorical([1.0, 0.0, 0.0]),
                Categorical([1.0, 0.0, 0.0]))
            e2 = (Categorical([0.0, 1.0, 0.0]),
                Categorical([0.0, 1.0, 0.0]))
            e3 = (Categorical([0.0, 0.0, 1.0]),
                Categorical([0.0, 0.0, 1.0]))
            y ~ MVDiscreteTimeDiscreteStatesHMM(
                P, (e1, e2, e3), Categorical([1.0, 0.0, 0.0]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [[2, 2], [2, 2], [3, 3]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    dist = MVDiscreteTimeDiscreteStatesHMM(
        [0.6 0.4 0.0; 0.0 0.7 0.3; 0.0 0.0 1.0],
        (
            (Categorical([1.0, 0.0, 0.0]), Categorical([1.0, 0.0, 0.0])),
            (Categorical([0.0, 1.0, 0.0]), Categorical([0.0, 1.0, 0.0])),
            (Categorical([0.0, 0.0, 1.0]), Categorical([0.0, 0.0, 1.0]))
        ),
        Categorical([1.0, 0.0, 0.0])
    )
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "MVDiscreteTimeHMM: missing observations still propagate hidden state" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            dummy = RealNumber(0.0)
        end
        @formulas begin
            P = [0.6 0.4 0.0;
                 0.0 0.7 0.3;
                 0.0 0.0 1.0]
            e1 = (Categorical([1.0, 0.0, 0.0]),
                Categorical([1.0, 0.0, 0.0]))
            e2 = (Categorical([0.0, 1.0, 0.0]),
                Categorical([0.0, 1.0, 0.0]))
            e3 = (Categorical([0.0, 0.0, 1.0]),
                Categorical([0.0, 0.0, 1.0]))
            y ~ MVDiscreteTimeDiscreteStatesHMM(
                P, (e1, e2, e3), Categorical([1.0, 0.0, 0.0]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = Any[[2, 2], missing, [3, 3]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    dist = MVDiscreteTimeDiscreteStatesHMM(
        [0.6 0.4 0.0; 0.0 0.7 0.3; 0.0 0.0 1.0],
        (
            (Categorical([1.0, 0.0, 0.0]), Categorical([1.0, 0.0, 0.0])),
            (Categorical([0.0, 1.0, 0.0]), Categorical([0.0, 1.0, 0.0])),
            (Categorical([0.0, 0.0, 1.0]), Categorical([0.0, 0.0, 1.0]))
        ),
        Categorical([1.0, 0.0, 0.0])
    )
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
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
            P = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "MVDiscreteTimeHMM: MLE/MAP/MCMC/VI" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            μ1 = RealNumber(0.0, prior = Normal(0.0, 2.0))
            μ2 = RealNumber(3.0, prior = Normal(3.0, 2.0))
        end
        @formulas begin
            P = [0.9 0.1; 0.2 0.8]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9],
            [0.0, 2.2], [2.9, -0.8], [0.1, 2.0]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 5)))
    @test res_mle isa FitResult

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs = (; iterations = 5)))
    @test res_map isa FitResult

    res_mcmc = fit_model(dm,
        NoLimits.MCMC(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains

    res_vi = fit_model(dm, NoLimits.VI(; turing_kwargs = (max_iter = 10, progress = false)))
    @test res_vi isa FitResult
end

# ---------------------------------------------------------------------------
# Block B: Integration tests — continuous-time variant (dt covariate)
# ---------------------------------------------------------------------------

@testset "MVContinuousTimeHMM: loglikelihood + DataModel" begin
    model = @Model begin
        @fixedEffects begin
            μ1 = RealNumber(0.0)
            μ2 = RealNumber(3.0)
        end
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end
        @formulas begin
            Q = [-1.0 1.0; 2.0 -2.0]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
end

@testset "MVContinuousTimeHMM: loglikelihood uses recursive filtering" begin
    Q = [-1.2 1.2 0.0;
         0.0 -1.0 1.0;
         0.0 0.0 0.0]
    emissions = (
        (Categorical([1.0, 0.0, 0.0]), Categorical([1.0, 0.0, 0.0])),
        (Categorical([0.0, 1.0, 0.0]), Categorical([0.0, 1.0, 0.0])),
        (Categorical([0.0, 0.0, 1.0]), Categorical([0.0, 0.0, 1.0]))
    )
    init = Categorical([1.0, 0.0, 0.0])

    model = @Model begin
        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end
        @formulas begin
            y ~ MVContinuousTimeDiscreteStatesHMM(
                [-1.2 1.2 0.0;
                 0.0 -1.0 1.0;
                 0.0 0.0 0.0],
                (
                    (Categorical([1.0, 0.0, 0.0]), Categorical([1.0, 0.0, 0.0])),
                    (Categorical([0.0, 1.0, 0.0]), Categorical([0.0, 1.0, 0.0])),
                    (Categorical([0.0, 0.0, 1.0]), Categorical([0.0, 0.0, 1.0]))
                ),
                Categorical([1.0, 0.0, 0.0]),
                dt
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = [[2, 2], [2, 2], [3, 3]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    dist = MVContinuousTimeDiscreteStatesHMM(Q, emissions, init, 1.0)

    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "MVContinuousTimeHMM: missing observations still propagate hidden state" begin
    Q = [-1.2 1.2 0.0;
         0.0 -1.0 1.0;
         0.0 0.0 0.0]
    emissions = (
        (Categorical([1.0, 0.0, 0.0]), Categorical([1.0, 0.0, 0.0])),
        (Categorical([0.0, 1.0, 0.0]), Categorical([0.0, 1.0, 0.0])),
        (Categorical([0.0, 0.0, 1.0]), Categorical([0.0, 0.0, 1.0]))
    )
    init = Categorical([1.0, 0.0, 0.0])

    model = @Model begin
        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end
        @formulas begin
            y ~ MVContinuousTimeDiscreteStatesHMM(
                [-1.2 1.2 0.0;
                 0.0 -1.0 1.0;
                 0.0 0.0 0.0],
                (
                    (Categorical([1.0, 0.0, 0.0]), Categorical([1.0, 0.0, 0.0])),
                    (Categorical([0.0, 1.0, 0.0]), Categorical([0.0, 1.0, 0.0])),
                    (Categorical([0.0, 0.0, 1.0]), Categorical([0.0, 0.0, 1.0]))
                ),
                Categorical([1.0, 0.0, 0.0]),
                dt
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = Any[[2, 2], missing, [3, 3]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    dist = MVContinuousTimeDiscreteStatesHMM(Q, emissions, init, 1.0)

    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "MVContinuousTimeHMM: ForwardDiff through full model" begin
    model = @Model begin
        @fixedEffects begin
            μ1 = RealNumber(0.0)
            μ2 = RealNumber(3.0)
        end
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end
        @formulas begin
            Q = [-1.0 1.0; 2.0 -2.0]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "MVContinuousTimeHMM: MLE/MAP/MCMC/VI" begin
    model = @Model begin
        @fixedEffects begin
            μ1 = RealNumber(0.0, prior = Normal(0.0, 2.0))
            μ2 = RealNumber(3.0, prior = Normal(3.0, 2.0))
        end
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end
        @formulas begin
            Q = [-1.0 1.0; 2.0 -2.0]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9],
            [0.0, 2.2], [2.9, -0.8], [0.1, 2.0]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 5)))
    @test res_mle isa FitResult

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs = (; iterations = 5)))
    @test res_map isa FitResult

    res_mcmc = fit_model(dm,
        NoLimits.MCMC(; sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains

    res_vi = fit_model(dm, NoLimits.VI(; turing_kwargs = (max_iter = 10, progress = false)))
    @test res_vi isa FitResult
end
