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
function _recursive_markov_loglikelihood_ct(dists, ys)
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

# ── 1. Acyclic Q: pathsum matches matrix exponential ─────────────────────────
@testset "CT observed MC: acyclic Q pathsum matches matrix exponential" begin
    λ12, λ13, λ23 = 0.4, 0.2, 0.7
    Q = [
        -(λ12 + λ13)  λ12            λ13
        0.0           -λ23           λ23
        0.0           0.0            0.0
    ]
    init = Categorical([1.0, 0.0, 0.0])
    dt   = 1.7

    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)

    p_pathsum = probabilities_hidden_states(dist)
    p_expm    = exp(transpose(Q) * dt) * init.p

    @test isapprox(p_pathsum, p_expm; rtol=1e-9, atol=1e-10)
    @test isapprox(sum(p_pathsum), 1.0; atol=1e-12)
end

# ── 2. Cyclic Q: auto mode falls back to expv ─────────────────────────────────
@testset "CT observed MC: cyclic Q auto mode is consistent with matrix exponential" begin
    λ12, λ23, λ31 = 0.4, 0.6, 0.5
    Q = [
        -λ12   λ12   0.0
        0.0   -λ23   λ23
        λ31   0.0   -λ31
    ]
    init = Categorical([0.7, 0.2, 0.1])
    dt   = 0.9

    dist   = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)
    p      = probabilities_hidden_states(dist)
    p_expm = exp(transpose(Q) * dt) * init.p

    @test isapprox(p, p_expm; rtol=1e-10, atol=1e-12)
end

# ── 3. probabilities_hidden_states / posterior_hidden_states ──────────────────
@testset "CT observed MC: probabilities_hidden_states and posterior_hidden_states" begin
    Q    = [-1.0 1.0; 0.5 -0.5]
    init = Categorical([1.0, 0.0])
    dt   = 0.5
    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)

    p = probabilities_hidden_states(dist)
    @test isapprox(sum(p), 1.0; atol=1e-12)
    @test all(>=(0), p)

    # posterior is one-hot
    @test isapprox(posterior_hidden_states(dist, 1), [1.0, 0.0]; atol=1e-12)
    @test isapprox(posterior_hidden_states(dist, 2), [0.0, 1.0]; atol=1e-12)

    # logpdf matches log(p[idx])
    @test isapprox(logpdf(dist, 1), log(p[1]); atol=1e-12)
    @test isapprox(logpdf(dist, 2), log(p[2]); atol=1e-12)
    @test logpdf(dist, 99) == -Inf
end

# ── 4. mean / var / cdf for integer labels ────────────────────────────────────
@testset "CT observed MC: mean/var/cdf for integer labels" begin
    Q    = [-1.0 1.0; 0.5 -0.5]
    dist = ContinuousTimeObservedStatesMarkovModel(Q, Categorical([1.0, 0.0]), 0.5)
    p    = probabilities_hidden_states(dist)

    @test isapprox(mean(dist), p[1] * 1 + p[2] * 2; atol=1e-12)
    μ = mean(dist)
    @test isapprox(var(dist), p[1] * (1 - μ)^2 + p[2] * (2 - μ)^2; atol=1e-12)
    @test isapprox(cdf(dist, 1), p[1]; atol=1e-12)
    @test isapprox(cdf(dist, 2), 1.0; atol=1e-12)
    @test isapprox(cdf(dist, 0), 0.0; atol=1e-12)
end

# ── 5. Symbol labels ───────────────────────────────────────────────────────────
@testset "CT observed MC: symbol labels" begin
    Q      = [-1.0 1.0; 0.5 -0.5]
    labels = [:healthy, :sick]
    dist   = ContinuousTimeObservedStatesMarkovModel(Q, Categorical([1.0, 0.0]), 0.5, labels)

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
    @test_throws MethodError   cdf(dist, :healthy)
end

# ── 6. Set-valued (censored) observations ──────────────────────────────────────
@testset "CT observed MC: set-valued observations" begin
    Q = [-1.0 1.0; 0.5 -0.5]
    dist = coarsed(ContinuousTimeObservedStatesMarkovModel(
        Q, Categorical([1.0, 0.0]), 0.5))
    p = probabilities_hidden_states(dist)

    @test isapprox(logpdf(dist, [1, 2]), log(1.0); atol=1e-12)
    @test isapprox(logpdf(dist, [2, 99]), log(p[2]); atol=1e-12)
    @test logpdf(dist, [99, 100]) == -Inf

    @test isapprox(posterior_hidden_states(dist, [1, 2]), p; atol=1e-12)
    @test isapprox(posterior_hidden_states(dist, [2, 99]), [0.0, 1.0]; atol=1e-12)
    @test isapprox(posterior_hidden_states(dist, [99, 100]), [0.0, 0.0]; atol=1e-12)
end

@testset "CT observed MC: set-valued observations require coarsed wrapper" begin
    Q = [-1.0 1.0; 0.5 -0.5]
    dist = ContinuousTimeObservedStatesMarkovModel(Q, Categorical([1.0, 0.0]), 0.5)

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

# ── 7. Forward-filter correctness (integer states, no missing) ─────────────────
@testset "CT observed MC: forward-filter loglikelihood correctness" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0  1.0; 0.5 -0.5]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y  = [1, 2, 1, 2]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ   = get_θ0_untransformed(dm.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = ContinuousTimeObservedStatesMarkovModel(
        [-1.0 1.0; 0.5 -0.5],
        Categorical([0.6, 0.4]),
        1.0
    )
    expected = _recursive_markov_loglikelihood_ct(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol=1e-12)
end

# ── 8. Missing observations propagate state distribution ─────────────────────
@testset "CT observed MC: missing observations propagate state" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0  1.0; 0.5 -0.5]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y  = Union{Missing, Int}[1, missing, missing, 2]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ   = get_θ0_untransformed(dm.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = ContinuousTimeObservedStatesMarkovModel(
        [-1.0 1.0; 0.5 -0.5],
        Categorical([0.6, 0.4]),
        1.0
    )
    expected = _recursive_markov_loglikelihood_ct(fill(dist_ref, nrow(df)), df.y)

    @test isapprox(ll, expected; atol=1e-12)
end

# ── 9. Set-valued labels through DataModel ─────────────────────────────────────
@testset "CT observed MC: set-valued labels through DataModel" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0  1.0; 0.5 -0.5]
            y ~ coarsed(ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y  = Any[[1], [1, 2], missing, [2]]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ   = get_θ0_untransformed(dm.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm, θ, ComponentArray())

    dist_ref = coarsed(ContinuousTimeObservedStatesMarkovModel(
        [-1.0 1.0; 0.5 -0.5],
        Categorical([0.6, 0.4]),
        1.0
    ))
    expected = _recursive_markov_loglikelihood_ct(fill(dist_ref, nrow(df)), df.y)

    @test isfinite(ll)
    @test isapprox(ll, expected; atol=1e-12)
end

@testset "CT observed MC: DataModel set-valued labels require coarsed wrapper" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0  1.0; 0.5 -0.5]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y  = Any[[1], [1, 2], missing, [2]]
    )

    err = try
        DataModel(model, df; primary_id=:ID, time_col=:t)
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
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            dummy = RealNumber(0.0)
        end

        @formulas begin
            Q = [-1.0  1.0; 0.5 -0.5]
            y ~ coarsed(ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.6, 0.4]), dt))
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y  = Union{Missing, Int}[1, 2, missing, 2]
    )

    err = try
        DataModel(model, df; primary_id=:ID, time_col=:t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("all non-missing observations must be AbstractVectors", sprint(showerror, err))
end

# ── 10. ForwardDiff gradient through Q-matrix entries ─────────────────────────
@testset "CT observed MC: ForwardDiff gradient through rate parameters" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0)
            λ21_r = RealNumber(0.0)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            Q   = [-λ12  λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t  = [0.0, 1.0, 2.0, 3.0],
        dt = fill(1.0, 4),
        y  = [1, 2, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g  = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
    @test all(isfinite, g)
end

# ── 9. Numerical stability: long sequence, small Δt ──────────────────────────
@testset "CT observed MC: numerical stability — long sequence, small dt" begin
    Q    = [-2.0  2.0;  1.0 -1.0]
    init = Categorical([1.0 - 1e-8, 1e-8])
    dt   = 0.12

    dist = ContinuousTimeObservedStatesMarkovModel(Q, init, dt)
    ys   = repeat([1, 2], 25)  # 50 observations

    ll = _recursive_markov_loglikelihood_ct(fill(dist, length(ys)), ys)

    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0)
            λ21_r = RealNumber(0.0)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            Q_m = [-λ12  λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q_m, Categorical([0.5, 0.5]), dt)
        end
    end

    n = 50
    df = DataFrame(
        ID = ones(Int, n),
        t  = Float64.(0:n-1) .* dt,
        dt = fill(dt, n),
        y  = repeat([1, 2], n ÷ 2)
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g  = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test all(isfinite, g)
end

# ── 10. MLE / MAP / MCMC ───────────────────────────────────────────────────────
@testset "CT observed MC: MLE/MAP/MCMC" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
            λ21_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
        end

        @formulas begin
            λ12 = exp(clamp(λ12_r, -2.0, 2.0))
            λ21 = exp(clamp(λ21_r, -2.0, 2.0))
            Q   = [-λ12  λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = fill(1.0, 6),
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

# ── 11. Random effects: Laplace / SAEM smoke test ─────────────────────────────
@testset "CT observed MC: random effects — Laplace and SAEM smoke test" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
            λ21_r = RealNumber(0.0, prior=Normal(0.0, 2.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            λ12 = exp(clamp(λ12_r + η, -2.0, 2.0))
            λ21 = exp(clamp(λ21_r, -2.0, 2.0))
            Q   = [-λ12  λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :A, :B, :B, :B],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = fill(1.0, 6),
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

# ── 12. simulate_data round-trip ──────────────────────────────────────────────
@testset "CT observed MC: simulate_data round-trip" begin
    model = @Model begin
        @covariates begin
            t  = Covariate()
            dt = Covariate()
        end

        @fixedEffects begin
            λ12_r = RealNumber(0.0)
            λ21_r = RealNumber(0.0)
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            Q   = [-λ12  λ12; λ21 -λ21]
            y ~ ContinuousTimeObservedStatesMarkovModel(Q, Categorical([0.5, 0.5]), dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t  = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = fill(1.0, 6),
        y  = Union{Missing, Int}[missing, missing, missing, missing, missing, missing]
    )

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    sim = simulate_data(dm; rng=Random.Xoshiro(9), replace_missings=true)

    @test sim isa DataFrame
    @test !any(ismissing, sim.y)
    @test all(v -> v ∈ [1, 2], sim.y)

    dm_sim = DataModel(model, sim; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm_sim.model.fixed.fixed)
    ll  = NoLimits.loglikelihood(dm_sim, θ0, ComponentArray())
end
