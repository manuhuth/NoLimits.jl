using Test
using NoLimits
using MCMCChains
using DataFrames
using Distributions
using ForwardDiff
using ComponentArrays
using Turing
using LinearAlgebra


@testset "HMM DataModel + loglikelihood" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale = :log)
            λ21_r = RealNumber(0.1, scale = :log)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
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
            Q = [
                -λ12 λ12;
                λ21 -λ21
            ]
            y ~ ContinuousTimeDiscreteStatesHMM(
                Q,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4]),
                dt
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
end

@testset "HMM loglikelihood uses recursive filtering" begin
    Q = [
        -1.2 1.2 0.0;
        0.0 -1.0 1.0;
        0.0 0.0 0.0
    ]
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
                [
                    -1.2 1.2 0.0;
                    0.0 -1.0 1.0;
                    0.0 0.0 0.0
                ],
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    dist = ContinuousTimeDiscreteStatesHMM(Q, emissions, init, 1.0)

    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1.0e-12)
end

@testset "HMM missing observations still propagate hidden state in loglikelihood" begin
    Q = [
        -1.2 1.2 0.0;
        0.0 -1.0 1.0;
        0.0 0.0 0.0
    ]
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
                [
                    -1.2 1.2 0.0;
                    0.0 -1.0 1.0;
                    0.0 0.0 0.0
                ],
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

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    dist = ContinuousTimeDiscreteStatesHMM(Q, emissions, init, 1.0)

    ll = NoLimits.loglikelihood(dm, θ, ComponentArray())
    expected = _recursive_hmm_loglikelihood(fill(dist, nrow(df)), df.y)

    @test isapprox(ll, expected; atol = 1.0e-12)
end

@testset "HMM ForwardDiff" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale = :log)
            λ21_r = RealNumber(0.1, scale = :log)
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.0)
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
            Q = [
                -λ12 λ12;
                λ21 -λ21
            ]
            y ~ ContinuousTimeDiscreteStatesHMM(
                Q,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4]),
                dt
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0],
        y = [0, 1, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    g = ForwardDiff.gradient(x -> NoLimits.loglikelihood(dm, x, ComponentArray()), θ0)

    @test length(g) == length(θ0)
end

@testset "CT HMM equilibrium shortcut under nested ForwardDiff and integer Δt" begin
    # max exit rate × Δt = 300 > _CT_HMM_EQUIL_THRESHOLD on a cyclic Q, so the spectral-gap
    # shortcut is taken. A Hessian reaches it with Dual{Dual} entries (as the random-effect
    # Hessian in GHQuadrature / Laplace / get_marginal_likelihood does); stripping only one
    # Dual layer used to hand `eigvals` a Dual matrix (MethodError: eigvals!).
    f(x) = begin
        λ = exp(x[1]); μ = exp(x[2])
        Q = [-λ λ; μ -μ]
        logpdf(
            ContinuousTimeDiscreteStatesHMM(
                Q, (Bernoulli(0.2), Bernoulli(0.7)), Categorical([0.6, 0.4]), 1.0
            ), 1
        )
    end
    # at equilibrium the state law is the stationary π = (μ, λ) / (λ + μ)
    fstat(x) = (λ = exp(x[1]); μ = exp(x[2]); log((0.2μ + 0.7λ) / (λ + μ)))
    x0 = [log(300.0), log(250.0)]
    @test f(x0) ≈ fstat(x0)
    @test ForwardDiff.gradient(f, x0) ≈ ForwardDiff.gradient(fstat, x0)
    H = ForwardDiff.hessian(f, x0)
    @test all(isfinite, H)
    @test H ≈ ForwardDiff.hessian(fstat, x0)

    # an integer Δt must not hit a missing `_ct_hmm_primal` method
    Q = [-300.0 300.0; 250.0 -250.0]
    hmm_int = ContinuousTimeDiscreteStatesHMM(
        Q, (Bernoulli(0.2), Bernoulli(0.7)), Categorical([0.6, 0.4]), 1
    )
    hmm_flt = ContinuousTimeDiscreteStatesHMM(
        Q, (Bernoulli(0.2), Bernoulli(0.7)), Categorical([0.6, 0.4]), 1.0
    )
    @test logpdf(hmm_int, 1) == logpdf(hmm_flt, 1)
end

@testset "CT HMM propagation stays bounded-time at extreme rates (#371)" begin
    # A fast 1<->2 exchange and a slow return from state 3: the spectral gap is ~ε, so the
    # stationary shortcut does not apply and `expv` would need ~K·Δt/10 Taylor substeps.
    Qf(K, ε) = [-K K 0.0; K -(K + ε) ε; ε 0.0 -ε]
    p0 = [0.2, 0.3, 0.5]
    emis = (Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(3.0, 1.0))
    hmm(Q; Δt = 1.5) = ContinuousTimeDiscreteStatesHMM(Q, emis, Categorical(p0), Δt)
    # Reference: exp(QᵀΔt) p0 for the same Float64 Q in 256-bit arithmetic, scaled to norm
    # 2^-12 first so that the Padé truncation error is negligible.
    function pref(Q; Δt = 1.5)
        A = setprecision(BigFloat, 256) do
            big.(Matrix(transpose(Q))) .* big(Δt)
        end
        return setprecision(BigFloat, 256) do
            k = max(0, ceil(Int, log2(opnorm(Float64.(A), 1)))) + 12
            E = NoLimits._matexp_pade(A ./ big(2)^k)
            for _ in 1:k
                E = E * E
            end
            Float64.(E * big.(p0))
        end
    end
    # forward error allowed by the conditioning: eps × ‖QΔt‖₁ (measured: <= 0.13 of it)
    tol(Q; Δt = 1.5) = max(eps() * opnorm(Q .* Δt, 1), 1.0e-14)

    # normal rates: max exit rate × Δt <= 200 keeps the `expv` result bit for bit
    for (K, Δt) in ((2.0, 1.5), (60.0, 3.0))
        Q = Qf(K, 0.3)
        @test probabilities_hidden_states(hmm(Q; Δt)) ==
            NoLimits.expv(Δt, transpose(Q), p0)
    end

    # extreme rates: fast, a probability vector, and equal to the dense exponential
    probabilities_hidden_states(hmm(Qf(1.0e3, 1.0e-3)))        # compile
    for K in (1.0e3, 1.0e6, 1.0e10, 1.0e14)
        Q = Qf(K, 1.0e-3)
        t = @elapsed p = probabilities_hidden_states(hmm(Q))
        @test t < 1.0
        @test all(>=(0), p) && sum(p) ≈ 1
        @test maximum(abs.(p .- pref(Q))) <= tol(Q)
        lp_ref = log(sum(pref(Q) .* pdf.(collect(emis), 0.7)))
        @test abs(logpdf(hmm(Q), 0.7) - lp_ref) <= 10 * tol(Q)
    end

    # absorbing state (non-ergodic, no stationary shortcut) at extreme rates
    Qa = [-2.0e9 1.0e9 1.0e9; 5.0e8 -5.0e8 0.0; 0.0 0.0 0.0]
    @test maximum(abs.(probabilities_hidden_states(hmm(Qa)) .- pref(Qa))) <= tol(Qa)
    hmm_expv = ContinuousTimeDiscreteStatesHMM(
        Qa, emis, Categorical(p0), 1.5; propagation_mode = :expv
    )
    @test maximum(abs.(probabilities_hidden_states(hmm_expv) .- pref(Qa))) <= tol(Qa)

    # ForwardDiff through the scaling-and-squaring path matches ForwardDiff through `expv`
    # (max exit rate × Δt = 450 > 200, cheap enough for `expv` as the reference)
    f(x) = logpdf(hmm(Qf(exp(x[1]), exp(x[2]))), 0.7)
    f_expv(x) = log(
        sum(NoLimits.expv(1.5, transpose(Qf(exp(x[1]), exp(x[2]))), p0) .* pdf.(collect(emis), 0.7))
    )
    x0 = [log(300.0), log(0.2)]
    @test f(x0) ≈ f_expv(x0) rtol = 1.0e-12
    @test ForwardDiff.gradient(f, x0) ≈ ForwardDiff.gradient(f_expv, x0) rtol = 1.0e-8 atol = 1.0e-12
    @test ForwardDiff.hessian(f, x0) ≈ ForwardDiff.hessian(f_expv, x0) rtol = 1.0e-6 atol = 1.0e-10
    x1 = [log(1.0e6), log(1.0e-3)]
    @test all(isfinite, ForwardDiff.gradient(f, x1))
    @test all(isfinite, ForwardDiff.hessian(f, x1))

    # non-finite rates are a numeric error (the estimators score the point as -Inf)
    err = try
        probabilities_hidden_states(hmm(Qf(Inf, 1.0)))
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test NoLimits._is_numeric_error(err)
    @test_throws DomainError probabilities_hidden_states(hmm([-1.0 NaN 1.0; 1.0 -1.0 0.0; 0.0 1.0 -1.0]))

    # a DataModel likelihood at extreme rates evaluates quickly and stays finite
    model = @Model begin
        @fixedEffects begin
            logK = RealNumber(0.0)
            logε = RealNumber(-3.0)
        end
        @covariates begin
            t = Covariate()
            dt = Covariate()
        end
        @formulas begin
            K = exp(logK)
            ε = exp(logε)
            Q = [-K K 0.0; K -(K + ε) ε; ε 0.0 -ε]
            y ~ ContinuousTimeDiscreteStatesHMM(
                Q, (Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(3.0, 1.0)),
                Categorical([0.2, 0.3, 0.5]), dt
            )
        end
    end
    df = DataFrame(
        ID = repeat([1, 2], inner = 4), t = repeat([0.0, 1.0, 2.0, 4.0], 2),
        dt = repeat([1.0, 1.0, 1.0, 2.0], 2), y = [0.1, 0.4, 2.9, 3.2, 1.1, 0.8, 0.2, 2.5]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    NoLimits.loglikelihood(dm, θ, ComponentArray())
    θx = copy(θ)
    θx.logK = log(1.0e12)
    NoLimits.loglikelihood(dm, θx, ComponentArray())
    t = @elapsed ll = NoLimits.loglikelihood(dm, θx, ComponentArray())
    @test isfinite(ll)
    @test t < 5.0
end

@testset "HMM ForwardDiff remains finite for tiny hidden-state masses" begin
    emissions = (
        Categorical([0.99999998, 1.0e-8, 1.0e-8]),
        Categorical([1.0e-8, 0.99999998, 1.0e-8]),
        Categorical([1.0e-8, 1.0e-8, 0.99999998]),
    )
    ys = vcat(fill(2, 5), fill(3, 80))
    dts = vcat([0.0], fill(0.12, length(ys) - 1))

    function seq_ll(x)
        q12 = exp(-1.4 + x)
        q13 = exp(-2.0 + x)
        q23 = exp(-0.5 + x)
        Q = [
            -(q12 + q13) q12 q13;
            0.0 -q23 q23;
            0.0 0.0 0.0
        ]

        prior = nothing
        ll = zero(x)
        for i in eachindex(ys)
            dist = ContinuousTimeDiscreteStatesHMM(
                Q, emissions, Categorical([0.6, 0.3, 0.1]), dts[i]
            )
            dist_use = prior === nothing ? dist :
                NoLimits._hmm_with_initial_probs(dist, prior)
            ll += logpdf(dist_use, ys[i])
            prior = posterior_hidden_states(dist_use, ys[i])
        end
        return ll
    end

    ll0 = seq_ll(0.0)
    g = ForwardDiff.derivative(seq_ll, 0.0)
end

@testset "HMM MLE/MAP/MCMC/VI optimization" begin
    model = @Model begin
        @fixedEffects begin
            λ12_r = RealNumber(0.1, scale = :log, prior = LogNormal(0.01, 1.0))
            λ21_r = RealNumber(0.1, scale = :log, prior = LogNormal(0.01, 1.0))
            p1_r = RealNumber(0.0, prior = Normal(0.0, 1.0))
            p2_r = RealNumber(0.0, prior = Normal(0.0, 1.0))
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
            Q = [
                -λ12 λ12;
                λ21 -λ21
            ]
            y ~ ContinuousTimeDiscreteStatesHMM(
                Q,
                (Bernoulli(p1), Bernoulli(p2)),
                Categorical([0.6, 0.4]),
                dt
            )
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        t = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        dt = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        y = [0, 1, 1, 1, 0, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res_mle = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    res_map = fit_model(dm, NoLimits.MAP(; optim_kwargs = (maxiters = 2,)))
    res_mcmc = fit_model(
        dm, NoLimits.MCMC(sampler = MH(), turing_kwargs = (; n_samples = 2, n_adapt = 1))
    )

    @test res_mle isa FitResult
    @test res_map isa FitResult
    @test res_mcmc isa FitResult
    @test NoLimits.get_chain(res_mcmc) isa MCMCChains.Chains

    res_vi = fit_model(dm, NoLimits.VI(; turing_kwargs = (max_iter = 10, progress = false)))
    @test res_vi isa FitResult
end
