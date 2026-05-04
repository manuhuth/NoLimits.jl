using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using Turing
using Random
using MCMCChains

# ── Manual forward-filter reference ────────────────────────────────────────────
# Mirrors the estimation path: propagate initial_dist one step, apply one-hot
# posterior after observing a state, predicted distribution for missing.
function _recursive_markov_loglikelihood(dists, ys)
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

# ── 1. probabilities_hidden_states / posterior_hidden_states ───────────────────
@testset "DT observed MC: probabilities_hidden_states and posterior_hidden_states" begin
    T = [0.8 0.2; 0.3 0.7]
    init = Categorical([1.0, 0.0])  # certainty in state 1
    dist = DiscreteTimeObservedStatesMarkovModel(T, init)

    # After one step from state 1: [0.8, 0.2]
    p = probabilities_hidden_states(dist)
    @test isapprox(p, [0.8, 0.2]; atol=1e-12)

    # After one step from state 2: [0.3, 0.7]
    dist2 = DiscreteTimeObservedStatesMarkovModel(T, Categorical([0.0, 1.0]))
    p2 = probabilities_hidden_states(dist2)
    @test isapprox(p2, [0.3, 0.7]; atol=1e-12)

    # posterior_hidden_states is one-hot
    post1 = posterior_hidden_states(dist, 1)
    @test isapprox(post1, [1.0, 0.0]; atol=1e-12)
    post2 = posterior_hidden_states(dist, 2)
    @test isapprox(post2, [0.0, 1.0]; atol=1e-12)

    # logpdf is log of predicted probability for the observed state
    @test isapprox(logpdf(dist, 1), log(0.8); atol=1e-12)
    @test isapprox(logpdf(dist, 2), log(0.2); atol=1e-12)

    # State not in labels → -Inf
    @test logpdf(dist, 99) == -Inf
end

# ── 2. mean / var / cdf for integer labels ─────────────────────────────────────
@testset "DT observed MC: mean/var/cdf for integer labels" begin
    T = [0.8 0.2; 0.3 0.7]
    dist = DiscreteTimeObservedStatesMarkovModel(T, Categorical([1.0, 0.0]))
    p = probabilities_hidden_states(dist)  # [0.8, 0.2]

    @test isapprox(mean(dist), 0.8 * 1 + 0.2 * 2; atol=1e-12)
    @test isapprox(var(dist),  0.8 * (1 - mean(dist))^2 + 0.2 * (2 - mean(dist))^2; atol=1e-12)
    @test isapprox(cdf(dist, 1), 0.8; atol=1e-12)
    @test isapprox(cdf(dist, 2), 1.0; atol=1e-12)
    @test isapprox(cdf(dist, 0), 0.0; atol=1e-12)
end

# ── 3. Symbol labels ────────────────────────────────────────────────────────────
@testset "DT observed MC: symbol labels" begin
    T = [0.8 0.2; 0.3 0.7]
    labels = [:healthy, :sick]
    dist = DiscreteTimeObservedStatesMarkovModel(T, Categorical([1.0, 0.0]), labels)

    p = probabilities_hidden_states(dist)
    @test isapprox(logpdf(dist, :healthy), log(p[1]); atol=1e-12)
    @test isapprox(logpdf(dist, :sick),    log(p[2]); atol=1e-12)
    @test logpdf(dist, :unknown) == -Inf

    @test posterior_hidden_states(dist, :healthy) ≈ [1.0, 0.0]
    @test posterior_hidden_states(dist, :sick)    ≈ [0.0, 1.0]

    y = rand(Random.Xoshiro(1), dist)
    @test y ∈ labels

    @test_throws ArgumentError mean(dist)
    @test_throws ArgumentError var(dist)
    # cdf is only defined for Real-valued y; calling with Symbol gives MethodError
    @test_throws MethodError cdf(dist, :healthy)
end

# ── 4. Forward-filter correctness (integer states, no missing) ─────────────────
@testset "DT observed MC: forward-filter loglikelihood correctness" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            T_mat = [0.7 0.3; 0.2 0.8]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        y  = [1, 2, 1, 2]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ   = get_θ0_untransformed(dm.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4])
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol=1e-12)
end

# ── 5. Missing observations propagate state distribution ──────────────────────
@testset "DT observed MC: missing observations propagate state" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            T_mat = [0.7 0.3; 0.2 0.8]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.6, 0.4]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        y  = Union{Missing, Int}[1, missing, missing, 2]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ   = get_θ0_untransformed(dm.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4])
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol=1e-12)
end

# ── 6. Symbol-label DataModel ──────────────────────────────────────────────────
@testset "DT observed MC: symbol labels through DataModel" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            T_mat = [0.7 0.3; 0.2 0.8]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat,
                                                     Categorical([0.6, 0.4]),
                                                     [:healthy, :sick])
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t  = [0.0, 1.0, 2.0],
        y  = [:healthy, :sick, :healthy]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ  = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4]),
        [:healthy, :sick]
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, 3), df.y)

    @test isapprox(ll, expected; atol=1e-12)
end

# ── 7. ForwardDiff gradient ────────────────────────────────────────────────────
@testset "DT observed MC: ForwardDiff gradient through transition parameters" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p12_r = RealNumber(0.0)
            p21_r = RealNumber(0.0)
        end

        @formulas begin
            p12 = 1 / (1 + exp(-p12_r))
            p21 = 1 / (1 + exp(-p21_r))
            T_mat = [1 - p12  p12;
                       p21  1 - p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        y  = [1, 2, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g  = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

# ── 8. MLE / MAP / MCMC ────────────────────────────────────────────────────────
@testset "DT observed MC: MLE/MAP/MCMC" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p12_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
            p21_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
        end

        @formulas begin
            p12 = 0.8 / (1 + exp(-p12_r)) + 0.1
            p21 = 0.8 / (1 + exp(-p21_r)) + 0.1
            T_mat = [1 - p12  p12;
                       p21  1 - p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y  = [1, 2, 1, 2, 2, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs=(; iterations=5)))
    @test res_mle isa FitResult

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs=(; iterations=5)))
    @test res_map isa FitResult

    res_mcmc = fit_model(dm, NoLimits.MCMC(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains
end

# ── 9. Random effects: Laplace / SAEM ─────────────────────────────────────────
@testset "DT observed MC: random effects — Laplace and SAEM smoke test" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p12_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
            p21_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p12 = 0.8 / (1 + exp(-clamp(p12_r + η, -2.0, 2.0))) + 0.1
            p21 = 0.8 / (1 + exp(-clamp(p21_r, -2.0, 2.0))) + 0.1
            T_mat = [1 - p12  p12;
                       p21  1 - p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :A, :B, :B, :B],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y  = [1, 2, 1, 2, 2, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res_lap = fit_model(dm, NoLimits.Laplace(;
        optim_kwargs=(maxiters=2,),
        inner_kwargs=(maxiters=2,),
        multistart_n=2, multistart_k=2))
    @test res_lap isa FitResult
    re = NoLimits.get_random_effects(dm, res_lap)
    @test re isa NamedTuple

    res_saem = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        mcmc_steps=1, q_store_max=2, maxiters=2, progress=false, builtin_stats=:auto);
        rng=Random.Xoshiro(42))
    @test res_saem isa FitResult
end

# ── 10. simulate_data round-trip ───────────────────────────────────────────────
@testset "DT observed MC: simulate_data round-trip" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p12_r = RealNumber(0.0)
            p21_r = RealNumber(0.0)
        end

        @formulas begin
            p12 = 0.8 / (1 + exp(-p12_r)) + 0.1
            p21 = 0.8 / (1 + exp(-p21_r)) + 0.1
            T_mat = [1 - p12  p12;
                       p21  1 - p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y  = Union{Missing, Int}[missing, missing, missing, missing, missing, missing]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=Random.Xoshiro(7), replace_missings=true)

    @test sim isa DataFrame
    @test !any(ismissing, sim.y)
    @test all(v -> v ∈ [1, 2], sim.y)

    # Can refit simulated data
    dm_sim = DataModel(model, sim; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm_sim.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm_sim, θ0, ComponentArray())
end
