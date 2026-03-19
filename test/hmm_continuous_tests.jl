using Test
using NoLimits
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using Turing
using LinearAlgebra

@testset "HMM acyclic path-sum propagation matches matrix exponential" begin
    λ12, λ13, λ23 = 0.4, 0.2, 0.7
    Q = [
        -(λ12 + λ13)  λ12           λ13
        0.0           -λ23          λ23
        0.0           0.0           0.0
    ]
    init = Categorical([1.0, 0.0, 0.0])
    dt = 1.7

    hmm = ContinuousTimeDiscreteStatesHMM(
        Q,
        (Bernoulli(0.2), Bernoulli(0.5), Bernoulli(0.8)),
        init,
        dt
    )

    p_pathsum = probabilities_hidden_states(hmm)
    p_expm = exp(transpose(Q) * dt) * init.p

    @test isapprox(p_pathsum, p_expm; rtol=1e-9, atol=1e-10)
    @test isapprox(sum(p_pathsum), 1.0; atol=1e-12)
end

@testset "HMM cyclic propagation falls back consistently" begin
    λ12, λ23, λ31 = 0.4, 0.6, 0.5
    Q = [
        -λ12  λ12  0.0
        0.0  -λ23  λ23
        λ31  0.0  -λ31
    ]
    init = Categorical([0.7, 0.2, 0.1])
    dt = 0.9

    hmm = ContinuousTimeDiscreteStatesHMM(
        Q,
        (Bernoulli(0.2), Bernoulli(0.5), Bernoulli(0.8)),
        init,
        dt
    )

    p = probabilities_hidden_states(hmm)
    p_expm = exp(transpose(Q) * dt) * init.p

    @test isapprox(p, p_expm; rtol=1e-10, atol=1e-12)
end

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

@testset "HMM DataModel + loglikelihood" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log)
            λ21_r = RealNumber(0.1, scale=:log)
            p1_r  = RealNumber(0.0)
            p2_r  = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())

    @test isfinite(ll)
end

@testset "HMM loglikelihood uses recursive filtering" begin
    Q = [-1.2 1.2 0.0;
          0.0 -1.0 1.0;
          0.0 0.0 0.0]
    emissions = (
        Categorical([1.0, 0.0, 0.0]),
        Categorical([0.0, 1.0, 0.0]),
        Categorical([0.0, 0.0, 1.0]),
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
            y ~ ContinuousTimeDiscreteStatesHMM(
                [-1.2 1.2 0.0;
                  0.0 -1.0 1.0;
                  0.0 0.0 0.0],
                (
                    Categorical([1.0, 0.0, 0.0]),
                    Categorical([0.0, 1.0, 0.0]),
                    Categorical([0.0, 0.0, 1.0]),
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
        y = [2, 2, 3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    dist = ContinuousTimeDiscreteStatesHMM(Q, emissions, init, 1.0)

    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol=1e-12)
end

@testset "HMM missing observations still propagate hidden state in loglikelihood" begin
    Q = [-1.2 1.2 0.0;
          0.0 -1.0 1.0;
          0.0 0.0 0.0]
    emissions = (
        Categorical([1.0, 0.0, 0.0]),
        Categorical([0.0, 1.0, 0.0]),
        Categorical([0.0, 0.0, 1.0]),
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
            y ~ ContinuousTimeDiscreteStatesHMM(
                [-1.2 1.2 0.0;
                  0.0 -1.0 1.0;
                  0.0 0.0 0.0],
                (
                    Categorical([1.0, 0.0, 0.0]),
                    Categorical([0.0, 1.0, 0.0]),
                    Categorical([0.0, 0.0, 1.0]),
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
        y = Union{Missing, Int}[2, missing, 3]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    dist = ContinuousTimeDiscreteStatesHMM(Q, emissions, init, 1.0)

    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isfinite(ll)
    @test isapprox(ll, expected; atol=1e-12)
end

@testset "HMM ForwardDiff" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log)
            λ21_r = RealNumber(0.1, scale=:log)
            p1_r  = RealNumber(0.0)
            p2_r  = RealNumber(0.0)
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = [0, 1, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
end

@testset "HMM ForwardDiff remains finite for tiny hidden-state masses" begin
    emissions = (
        Categorical([0.99999998, 1e-8, 1e-8]),
        Categorical([1e-8, 0.99999998, 1e-8]),
        Categorical([1e-8, 1e-8, 0.99999998]),
    )
    ys = vcat(fill(2, 5), fill(3, 80))
    dts = vcat([0.0], fill(0.12, length(ys) - 1))

    function seq_ll(x)
        q12 = exp(-1.4 + x)
        q13 = exp(-2.0 + x)
        q23 = exp(-0.5 + x)
        Q = [-(q12 + q13) q12 q13;
             0.0          -q23 q23;
             0.0           0.0 0.0]

        prior = nothing
        ll = zero(x)
        for i in eachindex(ys)
            dist = ContinuousTimeDiscreteStatesHMM(Q, emissions, Categorical([0.6, 0.3, 0.1]), dts[i])
            dist_use = prior === nothing ? dist : NoLimits._hmm_with_initial_probs(dist, prior)
            ll += logpdf(dist_use, ys[i])
            prior = posterior_hidden_states(dist_use, ys[i])
        end
        return ll
    end

    ll0 = seq_ll(0.0)
    g = ForwardDiff.derivative(seq_ll, 0.0)
    @test isfinite(ll0)
    @test isfinite(g)
end

@testset "HMM MLE/MAP/MCMC/VI optimization" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale=:log, prior = LogNormal(0.01, 1.0))
            λ21_r = RealNumber(0.1, scale=:log, prior = LogNormal(0.01, 1.0))
            p1_r  = RealNumber(0.0, prior = Normal(0.0, 1.0))
            p2_r  = RealNumber(0.0, prior = Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
            dt = Covariate()
        end

        @formulas begin
            λ12 = exp(λ12_r)
            λ21 = exp(λ21_r)
            p1 = 1 / (1 + exp(-p1_r))
            p2 = 1 / (1 + exp(-p2_r))
            Q = [-λ12  λ12;
                  λ21 -λ21]
            y ~ ContinuousTimeDiscreteStatesHMM(Q,
                                                (Bernoulli(p1), Bernoulli(p2)),
                                                Categorical([0.6, 0.4]),
                                                dt)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res_mle = fit_model(dm, NoLimits.MLE())
    res_map = fit_model(dm, NoLimits.MAP())
    res_mcmc = fit_model(dm, NoLimits.MCMC(sampler=MH(), turing_kwargs=(; n_samples=15, n_adapt=5)))


    @test res_mle isa FitResult
    @test isfinite(NoLimits.get_objective(res_mle))
    @test res_map isa FitResult
    @test isfinite(NoLimits.get_objective(res_map))
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains

    res_vi = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=10, progress=false)))
    @test res_vi isa FitResult
    @test isfinite(NoLimits.get_objective(res_vi))

end
