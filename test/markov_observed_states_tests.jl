using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using Turing
using Random
using MCMCChains
using LinearAlgebra

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

# ── CT-only: hidden-state propagation vs matrix exponential ───────────────────
@testset "CT observed MC: acyclic Q pathsum matches matrix exponential" begin
    λ12, λ13, λ23 = 0.4, 0.2, 0.7
    Q = [-(λ12 + λ13) λ12 λ13
         0.0 -λ23 λ23
         0.0 0.0 0.0]
    init = Categorical([1.0, 0.0, 0.0])
    dt = 1.7

    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)

    p_pathsum = probabilities_hidden_states(dist)
    p_expm = exp(transpose(Q) * dt) * init.p

    @test isapprox(p_pathsum, p_expm; rtol = 1e-9, atol = 1e-10)
    @test isapprox(sum(p_pathsum), 1.0; atol = 1e-12)
end

@testset "CT observed MC: cyclic Q auto mode is consistent with matrix exponential" begin
    λ12, λ23, λ31 = 0.4, 0.6, 0.5
    Q = [-λ12 λ12 0.0
         0.0 -λ23 λ23
         λ31 0.0 -λ31]
    init = Categorical([0.7, 0.2, 0.1])
    dt = 0.9

    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)
    p = probabilities_hidden_states(dist)
    p_expm = exp(transpose(Q) * dt) * init.p

    @test isapprox(p, p_expm; rtol = 1e-10, atol = 1e-12)
end

# ── Shared standalone tests ────────────────────────────────────────────────────
# Each variant: (testset label, constructor closure taking init [+ labels]).
markov_variants = (
    ("DT observed MC",
        (init, lbls...) -> DiscreteTimeObservedStatesMarkovModel(
            [0.8 0.2; 0.3 0.7], init, lbls...)),
    ("CT observed MC",
        (init, lbls...) -> ContinuousTimeObservedStatesMarkovModel(
            [-1.0 1.0; 0.5 -0.5], init, 0.5, lbls...)))

for (label, mk) in markov_variants
    @testset "$label: symbol labels" begin
        labels = [:healthy, :sick]
        dist = mk(Categorical([1.0, 0.0]), labels)

        p = probabilities_hidden_states(dist)
        @test isapprox(logpdf(dist, :healthy), log(p[1]); atol = 1e-12)
        @test isapprox(logpdf(dist, :sick), log(p[2]); atol = 1e-12)
        @test logpdf(dist, :unknown) == -Inf

        @test posterior_hidden_states(dist, :healthy) ≈ [1.0, 0.0]
        @test posterior_hidden_states(dist, :sick) ≈ [0.0, 1.0]

        y = rand(Random.Xoshiro(1), dist)
        @test y ∈ labels

        @test_throws ArgumentError mean(dist)
        @test_throws ArgumentError var(dist)
        # cdf is only defined for Real-valued y; calling with Symbol gives MethodError
        @test_throws MethodError cdf(dist, :healthy)
    end

    @testset "$label: set-valued observations" begin
        dist = coarsed(mk(Categorical([1.0, 0.0])))
        p = probabilities_hidden_states(dist)

        @test isapprox(logpdf(dist, [1, 2]), log(1.0); atol = 1e-12)
        @test isapprox(logpdf(dist, [2, 99]), log(p[2]); atol = 1e-12)
        @test logpdf(dist, [99, 100]) == -Inf

        @test isapprox(posterior_hidden_states(dist, [1, 2]), p; atol = 1e-12)
        @test isapprox(posterior_hidden_states(dist, [2, 99]), [0.0, 1.0]; atol = 1e-12)
        @test isapprox(posterior_hidden_states(dist, [99, 100]), [0.0, 0.0]; atol = 1e-12)
    end

    @testset "$label: set-valued observations require coarsed wrapper" begin
        dist = mk(Categorical([1.0, 0.0]))

        err = try
            logpdf(dist, [1, 2])
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("coarsed", sprint(showerror, err))
        @test_throws ErrorException posterior_hidden_states(dist, [1, 2])
    end
end

# ── Variant-specific standalone tests ──────────────────────────────────────────
@testset "DT observed MC: probabilities_hidden_states and posterior_hidden_states" begin
    T = [0.8 0.2; 0.3 0.7]
    init = Categorical([1.0, 0.0])  # certainty in state 1
    dist = DiscreteTimeObservedStatesMarkovModel(T, init)

    # After one step from state 1: [0.8, 0.2]
    p = probabilities_hidden_states(dist)
    @test isapprox(p, [0.8, 0.2]; atol = 1e-12)

    # After one step from state 2: [0.3, 0.7]
    dist2 = DiscreteTimeObservedStatesMarkovModel(T, Categorical([0.0, 1.0]))
    p2 = probabilities_hidden_states(dist2)
    @test isapprox(p2, [0.3, 0.7]; atol = 1e-12)

    # posterior_hidden_states is one-hot
    post1 = posterior_hidden_states(dist, 1)
    @test isapprox(post1, [1.0, 0.0]; atol = 1e-12)
    post2 = posterior_hidden_states(dist, 2)
    @test isapprox(post2, [0.0, 1.0]; atol = 1e-12)

    # logpdf is log of predicted probability for the observed state
    @test isapprox(logpdf(dist, 1), log(0.8); atol = 1e-12)
    @test isapprox(logpdf(dist, 2), log(0.2); atol = 1e-12)

    # State not in labels → -Inf
    @test logpdf(dist, 99) == -Inf
end

@testset "DT observed MC: mean/var/cdf for integer labels" begin
    T = [0.8 0.2; 0.3 0.7]
    dist = DiscreteTimeObservedStatesMarkovModel(T, Categorical([1.0, 0.0]))
    p = probabilities_hidden_states(dist)  # [0.8, 0.2]

    @test isapprox(mean(dist), 0.8 * 1 + 0.2 * 2; atol = 1e-12)
    @test isapprox(
        var(dist), 0.8 * (1 - mean(dist))^2 + 0.2 * (2 - mean(dist))^2; atol = 1e-12)
    @test isapprox(cdf(dist, 1), 0.8; atol = 1e-12)
    @test isapprox(cdf(dist, 2), 1.0; atol = 1e-12)
    @test isapprox(cdf(dist, 0), 0.0; atol = 1e-12)
end

@testset "CT observed MC: probabilities_hidden_states and posterior_hidden_states" begin
    Q = [-1.0 1.0; 0.5 -0.5]
    init = Categorical([1.0, 0.0])
    dt = 0.5
    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)

    p = probabilities_hidden_states(dist)
    @test isapprox(sum(p), 1.0; atol = 1e-12)
    @test all(>=(0), p)

    # posterior is one-hot
    @test isapprox(posterior_hidden_states(dist, 1), [1.0, 0.0]; atol = 1e-12)
    @test isapprox(posterior_hidden_states(dist, 2), [0.0, 1.0]; atol = 1e-12)

    # logpdf matches log(p[idx])
    @test isapprox(logpdf(dist, 1), log(p[1]); atol = 1e-12)
    @test isapprox(logpdf(dist, 2), log(p[2]); atol = 1e-12)
    @test logpdf(dist, 99) == -Inf
end

@testset "CT observed MC: mean/var/cdf for integer labels" begin
    Q = [-1.0 1.0; 0.5 -0.5]
    dist = ContinuousTimeObservedStatesMarkovModel(Q, Categorical([1.0, 0.0]), 0.5)
    p = probabilities_hidden_states(dist)

    @test isapprox(mean(dist), p[1] * 1 + p[2] * 2; atol = 1e-12)
    μ = mean(dist)
    @test isapprox(var(dist), p[1] * (1 - μ)^2 + p[2] * (2 - μ)^2; atol = 1e-12)
    @test isapprox(cdf(dist, 1), p[1]; atol = 1e-12)
    @test isapprox(cdf(dist, 2), 1.0; atol = 1e-12)
    @test isapprox(cdf(dist, 0), 0.0; atol = 1e-12)
end

# ── DT integration tests (DataModel plumbing, no dt covariate) ─────────────────
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
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1, 2, 1, 2]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4])
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

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
        t = [0.0, 1.0, 2.0, 3.0],
        y = Union{Missing, Int}[1, missing, missing, 2]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4])
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "DT observed MC: set-valued labels through DataModel" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            T_mat = [0.7 0.3; 0.2 0.8]
            y ~ coarsed(DiscreteTimeObservedStatesMarkovModel(
                T_mat, Categorical([0.6, 0.4])))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = Any[[1], [1, 2], missing, [2]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = coarsed(DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4])
    ))
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isfinite(ll)
    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "DT observed MC: DataModel set-valued labels require coarsed wrapper" begin
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
        t = [0.0, 1.0, 2.0, 3.0],
        y = Any[[1], [1, 2], missing, [2]]
    )

    err = try
        DataModel(model, df; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("non-coarsed", sprint(showerror, err))
end

@testset "DT observed MC: DataModel coarsed model requires AbstractVector observations" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            T_mat = [0.7 0.3; 0.2 0.8]
            y ~ coarsed(DiscreteTimeObservedStatesMarkovModel(
                T_mat, Categorical([0.6, 0.4])))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = Union{Missing, Int}[1, 2, missing, 2]
    )

    err = try
        DataModel(model, df; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin(
        "all non-missing observations must be AbstractVectors", sprint(showerror, err))
end

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
        t = [0.0, 1.0, 2.0],
        y = [:healthy, :sick, :healthy]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = DiscreteTimeObservedStatesMarkovModel(
        [0.7 0.3; 0.2 0.8],
        Categorical([0.6, 0.4]),
        [:healthy, :sick]
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, 3), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

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
            T_mat = [1-p12 p12;
                     p21 1-p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1, 2, 1, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "DT observed MC: MLE/MAP/MCMC" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p12_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
            p21_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
        end

        @formulas begin
            p12 = 0.8 / (1 + exp(-p12_r)) + 0.1
            p21 = 0.8 / (1 + exp(-p21_r)) + 0.1
            T_mat = [1-p12 p12;
                     p21 1-p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [1, 2, 1, 2, 2, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 5)))
    @test res_mle isa FitResult

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs = (; iterations = 5)))
    @test res_map isa FitResult

    res_mcmc = fit_model(dm,
        NoLimits.MCMC(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains
end

@testset "DT observed MC: random effects — Laplace and SAEM smoke test" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            p12_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
            p21_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            p12 = 0.8 / (1 + exp(-clamp(p12_r + η, -2.0, 2.0))) + 0.1
            p21 = 0.8 / (1 + exp(-clamp(p21_r, -2.0, 2.0))) + 0.1
            T_mat = [1-p12 p12;
                     p21 1-p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = [1, 2, 1, 2, 2, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_lap = fit_model(dm,
        NoLimits.Laplace(;
            optim_kwargs = (maxiters = 2,),
            inner_kwargs = (maxiters = 2,),
            multistart_n = 2, multistart_k = 2))
    @test res_lap isa FitResult
    re = NoLimits.get_random_effects(dm, res_lap)
    @test re isa NamedTuple

    res_saem = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            mcmc_steps = 1, q_store_max = 2, maxiters = 2, progress = false, builtin_stats = :auto);
        rng = Random.Xoshiro(42))
    @test res_saem isa FitResult
end

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
            T_mat = [1-p12 p12;
                     p21 1-p21]
            y ~ DiscreteTimeObservedStatesMarkovModel(T_mat, Categorical([0.5, 0.5]))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        y = Union{Missing, Int}[missing, missing, missing, missing, missing, missing]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    sim = simulate_data(dm; rng = Random.Xoshiro(7), replace_missings = true)

    @test sim isa DataFrame
    @test !any(ismissing, sim.y)
    @test all(v -> v ∈ [1, 2], sim.y)

    # Can refit simulated data
    dm_sim = DataModel(model, sim; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm_sim.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm_sim, θ0, ComponentArray())
end

# ── CT integration tests (DataModel plumbing with dt covariate) ────────────────
@testset "CT observed MC: forward-filter loglikelihood correctness" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0 1.0; 0.5 -0.5]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y = [1, 2, 1, 2]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = ContinuousTimeObservedStatesMarkovModel(
        [-1.0 1.0; 0.5 -0.5],
        Categorical([0.6, 0.4]),
        1.0
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "CT observed MC: missing observations propagate state" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0 1.0; 0.5 -0.5]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y = Union{Missing, Int}[1, missing, missing, 2]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = ContinuousTimeObservedStatesMarkovModel(
        [-1.0 1.0; 0.5 -0.5],
        Categorical([0.6, 0.4]),
        1.0
    )
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "CT observed MC: set-valued labels through DataModel" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0 1.0; 0.5 -0.5]
            y ~ coarsed(ContinuousTimeObservedStatesMarkovModel(
                Q, Categorical([0.6, 0.4]), dt))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y = Any[[1], [1, 2], missing, [2]]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = coarsed(ContinuousTimeObservedStatesMarkovModel(
        [-1.0 1.0; 0.5 -0.5],
        Categorical([0.6, 0.4]),
        1.0
    ))
    expected = _recursive_markov_loglikelihood(fill(dist_ref, nrow(df)), df.y)

    @test isfinite(ll)
    @test isapprox(ll, expected; atol = 1e-12)
end

@testset "CT observed MC: DataModel set-valued labels require coarsed wrapper" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0 1.0; 0.5 -0.5]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y = Any[[1], [1, 2], missing, [2]]
    )

    err = try
        DataModel(model, df; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("non-coarsed", sprint(showerror, err))
end

@testset "CT observed MC: DataModel coarsed model requires AbstractVector observations" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0 1.0; 0.5 -0.5]
            y ~ coarsed(ContinuousTimeObservedStatesMarkovModel(
                Q, Categorical([0.6, 0.4]), dt))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y = Union{Missing, Int}[1, 2, missing, 2]
    )

    err = try
        DataModel(model, df; primary_id = :ID, time_col = :t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin(
        "all non-missing observations must be AbstractVectors", sprint(showerror, err))
end

@testset "CT observed MC: ForwardDiff gradient through rate parameters" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0)
            λ21_r = RealNumber(0.0)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y = [1, 2, 1, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

@testset "CT observed MC: numerical stability — long sequence, small dt" begin
    Q = [-2.0 2.0; 1.0 -1.0]
    init = Categorical([1.0 - 1e-8, 1e-8])
    dt = 0.12

    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)
    ys = repeat([1, 2], 25)  # 50 observations

    ll = _recursive_markov_loglikelihood(fill(dist, length(ys)), ys)

    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0)
            λ21_r = RealNumber(0.0)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            Q_m = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q_m, Categorical([0.5, 0.5]), dt)
        end
    end

    n = 50
    df = DataFrame(
        ID = ones(Int, n),
        t = Float64.(0:(n - 1)) .* dt,
        dt = fill(dt, n),
        y = repeat([1, 2], n ÷ 2)
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test all(isfinite, g)
end

@testset "CT observed MC: MLE/MAP/MCMC" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
            λ21_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
        end

        @formulas begin
            λ12 = exp(clamp(λ12_r, -2.0, 2.0))
            λ21 = exp(clamp(λ21_r, -2.0, 2.0))
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = fill(1.0, 6),
        y = [1, 2, 1, 2, 2, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_mle = fit_model(dm, NoLimits.MLE(optim_kwargs = (; iterations = 5)))
    @test res_mle isa FitResult

    res_map = fit_model(dm, NoLimits.MAP(optim_kwargs = (; iterations = 5)))
    @test res_map isa FitResult

    res_mcmc = fit_model(dm,
        NoLimits.MCMC(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false)))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains
end

@testset "CT observed MC: random effects — Laplace and SAEM smoke test" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
            λ21_r = RealNumber(0.0, prior = Normal(0.0, 2.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            λ12 = exp(clamp(λ12_r + η, -2.0, 2.0))
            λ21 = exp(clamp(λ21_r, -2.0, 2.0))
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = fill(1.0, 6),
        y = [1, 2, 1, 2, 2, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_lap = fit_model(dm,
        NoLimits.Laplace(;
            optim_kwargs = (maxiters = 2,),
            inner_kwargs = (maxiters = 2,),
            multistart_n = 2, multistart_k = 2))
    @test res_lap isa FitResult
    re = NoLimits.get_random_effects(dm, res_lap)
    @test re isa NamedTuple

    res_saem = fit_model(dm,
        NoLimits.SAEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            mcmc_steps = 1, q_store_max = 2, maxiters = 2, progress = false, builtin_stats = :auto);
        rng = Random.Xoshiro(42))
    @test res_saem isa FitResult
end

@testset "CT observed MC: simulate_data round-trip" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0)
            λ21_r = RealNumber(0.0)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            Q = [-λ12 λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = fill(1.0, 6),
        y = Union{Missing, Int}[missing, missing, missing, missing, missing, missing]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    sim = simulate_data(dm; rng = Random.Xoshiro(9), replace_missings = true)

    @test sim isa DataFrame
    @test !any(ismissing, sim.y)
    @test all(v -> v ∈ [1, 2], sim.y)

    dm_sim = DataModel(model, sim; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm_sim.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm_sim, θ0, ComponentArray())
end
