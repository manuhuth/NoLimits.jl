using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using LinearAlgebra
using Random
using Turing

# Rate matrix used throughout: Q = [-1 1; 2 -2], stationary distribution [2/3, 1/3]
const _Q_MV = [-1.0 1.0; 2.0 -2.0]

# ---------------------------------------------------------------------------
# Block A: Standalone struct tests
# ---------------------------------------------------------------------------

@testset "MVContinuousTimeHMM: constructor" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([0.6, 0.4])

    hmm = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 1.0)
    @test hmm.n_states   == 2
    @test hmm.n_outcomes == 2
    @test Base.length(hmm) == 2
    @test hmm.Δt == 1.0

    # Joint MvNormal emission mode
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm_mv = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_mv, init, 1.0)
    @test hmm_mv.n_states   == 2
    @test hmm_mv.n_outcomes == 2
end

@testset "MVContinuousTimeHMM: constructor validation" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([0.6, 0.4])

    # Non-square rate matrix
    @test_throws ErrorException MVContinuousTimeDiscreteStatesHMM(
        [1.0 0.0 0.0; 0.0 1.0 0.0], em, init, 1.0)

    # Wrong number of emission elements
    @test_throws ErrorException MVContinuousTimeDiscreteStatesHMM(
        _Q_MV, (em..., (Normal(), Normal())), init, 1.0)

    # Wrong initial_dist size
    @test_throws ErrorException MVContinuousTimeDiscreteStatesHMM(
        _Q_MV, em, Categorical([1/3, 1/3, 1/3]), 1.0)

    # Mismatched n_outcomes across states
    @test_throws ErrorException MVContinuousTimeDiscreteStatesHMM(
        _Q_MV,
        ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0),)),
        init, 1.0)
end

@testset "MVContinuousTimeHMM: probabilities_hidden_states" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([1.0, 0.0])

    # Zero Δt: state distribution is unchanged
    hmm_zero = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 0.0)
    p_zero   = probabilities_hidden_states(hmm_zero)
    @test isapprox(p_zero, [1.0, 0.0]; atol=1e-6)

    # Large Δt: approaches stationary distribution [2/3, 1/3]
    hmm_long = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 50.0)
    p_long   = probabilities_hidden_states(hmm_long)
    @test isapprox(p_long[1], 2/3; atol=1e-4)
    @test isapprox(p_long[2], 1/3; atol=1e-4)
end

@testset "MVContinuousTimeHMM: Δt affects state probabilities" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
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

@testset "MVContinuousTimeHMM: logpdf — independent emissions" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    init = Categorical([0.6, 0.4])
    hmm  = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 1.0)
    y    = [0.1, 2.1]

    lp = logpdf(hmm, y)
    @test isfinite(lp)
    @test isapprox(exp(lp), pdf(hmm, y); atol=1e-12)

    # Different y gives different logpdf
    @test logpdf(hmm, [3.1, -0.9]) != lp

    # Manual check
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(em[k][1], y[1]) * pdf(em[k][2], y[2]) for k in 1:2]
    @test isapprox(exp(lp), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVContinuousTimeHMM: logpdf — joint MvNormal emissions" begin
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    init  = Categorical([0.6, 0.4])
    hmm   = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_mv, init, 1.0)
    y     = [0.1, 2.1]

    lp = logpdf(hmm, y)
    @test isfinite(lp)
    @test isapprox(exp(lp), pdf(hmm, y); atol=1e-12)

    # Manual check
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(em_mv[k], y) for k in 1:2]
    @test isapprox(exp(lp), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVContinuousTimeHMM: missing — independent, partial" begin
    em   = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm  = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, Categorical([0.6, 0.4]), 1.0)

    lp_partial = logpdf(hmm, [0.1, missing])
    @test isfinite(lp_partial)
    @test lp_partial != logpdf(hmm, [0.1, 2.1])

    # Equals mixture over first outcome only
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(em[k][1], 0.1) for k in 1:2]
    @test isapprox(exp(lp_partial), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVContinuousTimeHMM: missing — independent, all missing" begin
    em  = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, Categorical([0.6, 0.4]), 1.0)

    @test isapprox(logpdf(hmm, [missing, missing]), 0.0; atol=1e-10)
end

@testset "MVContinuousTimeHMM: missing — MvNormal, partial" begin
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm   = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_mv, Categorical([0.6, 0.4]), 1.0)

    lp_partial = logpdf(hmm, [0.1, missing])
    @test isfinite(lp_partial)

    # Manual: marginal of MvNormal(μ, I) over index 1 is Normal(μ[1], 1.0)
    p_h   = probabilities_hidden_states(hmm)
    p_obs = [pdf(Normal(em_mv[k].μ[1], 1.0), 0.1) for k in 1:2]
    @test isapprox(exp(lp_partial), sum(p_h .* p_obs); atol=1e-10)
end

@testset "MVContinuousTimeHMM: missing — MvNormal, all missing" begin
    em_mv = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
             MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm   = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_mv, Categorical([0.6, 0.4]), 1.0)

    @test isapprox(logpdf(hmm, [missing, missing]), 0.0; atol=1e-10)
end

@testset "MVContinuousTimeHMM: missing — non-MvNormal joint throws error" begin
    init   = Categorical([0.6, 0.4])
    em_bad = (Dirichlet([1.0, 1.0]), Dirichlet([2.0, 2.0]))
    hmm    = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_bad, init, 1.0)

    @test_throws ErrorException logpdf(hmm, [0.3, missing])
end

@testset "MVContinuousTimeHMM: posterior_hidden_states" begin
    em  = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, Categorical([0.6, 0.4]), 1.0)

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

@testset "MVContinuousTimeHMM: rand" begin
    em  = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, Categorical([0.6, 0.4]), 1.0)

    s = rand(MersenneTwister(42), hmm)
    @test s isa Vector
    @test length(s) == 2
    @test all(isfinite, s)

    # Joint MvNormal mode
    em_mv  = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
              MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm_mv = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_mv, Categorical([0.6, 0.4]), 1.0)
    s_mv   = rand(MersenneTwister(42), hmm_mv)
    @test s_mv isa Vector
    @test length(s_mv) == 2
    @test all(isfinite, s_mv)
end

@testset "MVContinuousTimeHMM: mean, cov, var" begin
    em  = ((Normal(0.0, 1.0), Normal(2.0, 0.5)), (Normal(3.0, 1.0), Normal(-1.0, 0.5)))
    hmm = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, Categorical([0.6, 0.4]), 1.0)

    m = mean(hmm)
    @test m isa Vector && length(m) == 2 && all(isfinite, m)

    C = cov(hmm)
    @test C isa Matrix && size(C) == (2, 2)
    @test isapprox(C, C'; atol=1e-10)

    v = var(hmm)
    @test v isa Vector && length(v) == 2
    @test isapprox(v, diag(C); atol=1e-10)

    # Joint MvNormal mode
    em_mv  = (MvNormal([0.0, 2.0], [1.0 0.0; 0.0 1.0]),
              MvNormal([3.0, -1.0], [1.0 0.0; 0.0 1.0]))
    hmm_mv = MVContinuousTimeDiscreteStatesHMM(_Q_MV, em_mv, Categorical([0.6, 0.4]), 1.0)
    C_mv   = cov(hmm_mv)
    @test C_mv isa Matrix && size(C_mv) == (2, 2)
    @test isapprox(C_mv, C_mv'; atol=1e-10)
    @test isapprox(var(hmm_mv), diag(C_mv); atol=1e-10)
end

@testset "MVContinuousTimeHMM: ForwardDiff through logpdf" begin
    init = Categorical([0.6, 0.4])
    y    = [0.5, 1.5]

    f = x -> begin
        em = ((Normal(x[1], 1.0), Normal(x[2], 0.5)), (Normal(x[3], 1.0), Normal(x[4], 0.5)))
        logpdf(MVContinuousTimeDiscreteStatesHMM(_Q_MV, em, init, 1.0), y)
    end

    g = ForwardDiff.gradient(f, [0.0, 2.0, 3.0, -1.0])
    @test length(g) == 4
    @test all(isfinite, g)
end

# ---------------------------------------------------------------------------
# Block B: Integration tests
# ---------------------------------------------------------------------------

@testset "MVContinuousTimeHMM: loglikelihood + DataModel" begin
    model = @Model begin
        @fixedEffects begin
            μ1 = RealNumber(0.0)
            μ2 = RealNumber(3.0)
        end
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end
        @formulas begin
            Q  = [-1.0 1.0; 2.0 -2.0]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t  = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y  = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]],
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ  = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    @test isfinite(ll)
end

@testset "MVContinuousTimeHMM: ForwardDiff through full model" begin
    model = @Model begin
        @fixedEffects begin
            μ1 = RealNumber(0.0)
            μ2 = RealNumber(3.0)
        end
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end
        @formulas begin
            Q  = [-1.0 1.0; 2.0 -2.0]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t  = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y  = [[0.1, 2.1], [0.2, 1.9], [3.1, -0.9]],
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0  = get_θ0_untransformed(dm.model.fixed.fixed)
    g   = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "MVContinuousTimeHMM: MLE and MAP" begin
    model = @Model begin
        @fixedEffects begin
            μ1 = RealNumber(0.0, prior=Normal(0.0, 2.0))
            μ2 = RealNumber(3.0, prior=Normal(3.0, 2.0))
        end
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end
        @formulas begin
            Q  = [-1.0 1.0; 2.0 -2.0]
            e1 = (Normal(μ1, 1.0), Normal(2.0, 0.5))
            e2 = (Normal(μ2, 1.0), Normal(-1.0, 0.5))
            y ~ MVContinuousTimeDiscreteStatesHMM(Q, (e1, e2), Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
