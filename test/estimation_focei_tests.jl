using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using FiniteDifferences
using LinearAlgebra
using Random
using OrdinaryDiffEq
using SpecialFunctions: trigamma

const NL = NoLimits

@testset "FOCEI Fisher-information registry" begin
    # Closed-form values (Distributions.jl native parameterisation).
    @test NL._focei_expected_information(Normal(0.3, 2.0)) ≈ [0.25 0.0; 0.0 0.5]
    @test NL._focei_expected_information(LogNormal(0.1, 2.0)) ≈ [0.25 0.0; 0.0 0.5]
    @test NL._focei_expected_information(Poisson(3.0)) ≈ fill(1 / 3, 1, 1)
    @test NL._focei_expected_information(Bernoulli(0.3)) ≈ fill(1 / (0.3 * 0.7), 1, 1)
    @test NL._focei_expected_information(Exponential(1.4)) ≈ fill(1 / 1.4^2, 1, 1)
    @test NL._focei_expected_information(Binomial(10, 0.4)) ≈ fill(10 / (0.4 * 0.6), 1, 1)
    @test NL._focei_expected_information(Geometric(0.3)) ≈ fill(1 / (0.3^2 * 0.7), 1, 1)
    @test NL._focei_expected_information(Distributions.Laplace(0.0, 0.7)) ≈
          [1/0.49 0.0; 0.0 1/0.49]
    @test NL._focei_expected_information(Cauchy(0.0, 0.5)) ≈ [2.0 0.0; 0.0 2.0]

    tab = trigamma(5.0)
    @test NL._focei_expected_information(Gamma(2.0, 1.5)) ≈
          [trigamma(2.0) 1/1.5; 1/1.5 2.0/1.5^2]
    @test NL._focei_expected_information(Beta(2.0, 3.0)) ≈
          [trigamma(2.0)-tab -tab; -tab trigamma(3.0)-tab]

    # Symmetry + positive-definiteness.
    for d in (Normal(1.0, 0.5), Gamma(2.0, 1.5), Beta(2.0, 3.0),
        Distributions.Laplace(0.0, 0.7), Cauchy(0.0, 0.5))
        Im = NL._focei_expected_information(d)
        @test Im ≈ transpose(Im)
        @test isposdef(Symmetric(Im))
    end

    # Dispersion indices: only location-scale residual-error families freeze a parameter.
    @test NL._focei_dispersion_indices(Normal(0.0, 1.0)) == [2]
    @test NL._focei_dispersion_indices(LogNormal(0.0, 1.0)) == [2]
    @test NL._focei_dispersion_indices(Poisson(1.0)) == Int[]
    @test NL._focei_dispersion_indices(Gamma(2.0, 1.0)) == Int[]

    # Support flags.
    @test NL._focei_is_supported(Normal(0.0, 1.0))
    @test NL._focei_is_supported(Gamma(2.0, 1.0))
    @test NL._focei_is_supported(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
    @test !NL._focei_is_supported("not a distribution")

    # Float32 cleanliness of the kernel.
    I32 = NL._focei_expected_information(Normal(0.3f0, 0.5f0))
    @test eltype(I32) == Float32
    @test I32 ≈ Float32[1/0.25 0.0; 0.0 2/0.25]
end

@testset "FOCEI negH equals exact Laplace Hessian (linear-Gaussian)" begin
    dm = fx_re_dm()                       # shared scalar-RE linear-Gaussian model
    _, batch_infos, const_cache = NL._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NL.build_ll_cache(dm; force_saveat = true)
    θu = NL.get_θ0_untransformed(fx_re_model().fixed.fixed)

    for info in batch_infos
        info.n_b == 0 && continue
        b = fill(0.15, info.n_b)
        Hf = NL._focei_negH_batch(
            dm, info, θu, b, const_cache, ll_cache; interaction = true)
        He = -ForwardDiff.hessian(
            bb -> NL._laplace_logf_batch(dm, info, θu, bb, const_cache, ll_cache), b)
        # FOCEI is exact for a linear-Gaussian model.
        @test Hf≈He atol=1e-8
        @test Hf ≈ transpose(Hf)
        @test isposdef(Symmetric(Hf))
    end
end

@testset "FOCEI fit matches Laplace (linear-Gaussian)" begin
    # FOCEI ≡ Laplace is an exact identity for linear-Gaussian models, so the
    # shared fx_focei()/fx_laplace() fits (same fx_re_dm) must agree.
    rf = fx_focei()
    rl = fx_laplace()
    @test NL.get_objective(rf)≈NL.get_objective(rl) atol=1e-3
    @test collect(NL.get_params(
        rf; scale = :untransformed))≈collect(NL.get_params(rl; scale = :untransformed)) atol=1e-2
    @test rf.result.eb_modes !== nothing
    re = NL.get_random_effects(rf)
    @test re isa NamedTuple && re.η isa DataFrame
    @test NL.get_converged(rf) isa Bool
end

@testset "FOCEI Wald UQ matches Laplace (linear-Gaussian)" begin
    rf = fx_focei()
    rl = fx_laplace()
    uq_f = compute_uq(rf; n_draws = 30, serialization = NL.EnsembleSerial())
    uq_l = compute_uq(rl; n_draws = 30, serialization = NL.EnsembleSerial())

    @test NL.get_uq_backend(uq_f) == :wald
    @test NL.get_uq_source_method(uq_f) == :focei
    @test NL.get_uq_diagnostics(uq_f).approximation_method == :focei

    # Gaussian outcome: the FOCEI Fisher-information Hessian equals the exact inner
    # Hessian, so Wald SEs must match the Laplace ones.
    se_f = sqrt.(diag(NL.get_uq_vcov(uq_f; scale = :transformed)))
    se_l = sqrt.(diag(NL.get_uq_vcov(uq_l; scale = :transformed)))
    @test se_f≈se_l rtol=5e-2

    # intervals on natural scale exist for all three parameters
    @test NL.get_uq_intervals(uq_f; scale = :natural) !== nothing
    @test length(NL.get_uq_parameter_names(uq_f; scale = :transformed)) == 3
end

@testset "FOCE freezes dispersion at the random-effects prior mean" begin
    a0, c0, γ0, μη0, ω0, b1 = 1.0, -0.3, 0.8, 0.3, 0.6, 1.0
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            c = RealNumber(-0.3)
            γ = RealNumber(0.8)
            μη = RealNumber(0.3)
            ω = RealNumber(0.6, scale = :log, lower = 1e-8, upper = Inf)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(μη, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, exp(c + γ * η))
        end
    end

    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.3, 1.1])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _, batch_infos, const_cache = NL._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NL.build_ll_cache(dm; force_saveat = true)
    θu = NL.get_θ0_untransformed(dm.model.fixed.fixed)
    info = batch_infos[1]
    b = [b1]

    Hfoce = NL._focei_negH_batch(
        dm, info, θu, b, const_cache, ll_cache; interaction = false)
    Hfoci = NL._focei_negH_batch(dm, info, θu, b, const_cache, ll_cache; interaction = true)

    Λ = 1 / ω0^2
    # 2 observations, ∂μ/∂η = 1, ∂σ/∂η = γσ.
    foce_data = 2 * exp(-2 * (c0 + γ0 * μη0))          # σ frozen at η = mean = μη
    foci_data = 2 * (exp(-2 * (c0 + γ0 * b1)) + 2 * γ0^2)
    @test Hfoce[1, 1]≈foce_data + Λ atol=1e-6
    @test Hfoci[1, 1]≈foci_data + Λ atol=1e-6
    @test !isapprox(Hfoce[1, 1], Hfoci[1, 1]; atol = 1e-3)
    # Crucially, FOCE froze the dispersion at η = μη, NOT at η = 0.
    @test !isapprox(Hfoce[1, 1], 2 * exp(-2 * c0) + Λ; atol = 1e-3)

    # FOCE end-to-end fit runs.
    rf = fit_model(dm,
        NL.FOCEI(interaction = false, multistart_n = 1,
            multistart_k = 1, optim_kwargs = (maxiters = 3,));
        serialization = NL.EnsembleSerial())
    @test isfinite(NL.get_objective(rf))
end

@testset "FOCEI outer gradient matches finite differences" begin
    dm = fx_re_dm()                       # shared scalar-RE model
    fe = fx_re_model().fixed.fixed
    method = NL.FOCEI(multistart_n = 1, multistart_k = 1)
    inner_opts = NL._resolve_inner_options(method.inner, dm)
    ms_opts = NL._resolve_multistart_options(method.multistart, inner_opts)
    _, batch_infos, const_cache = NL._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NL.build_ll_cache(dm; force_saveat = true)
    nb_ = length(batch_infos)
    mkcache() = NL._LaplaceCache(nothing,
        NL._LaplaceBStarCache([Float64[] for _ in 1:nb_], falses(nb_)),
        NL._LaplaceGradCache([Float64[] for _ in 1:nb_], fill(NaN, nb_),
            [Float64[] for _ in 1:nb_], falses(nb_)),
        NL._init_laplace_ad_cache(nb_),
        NL._init_laplace_hess_cache(Float64, nb_))
    hmode = NL._FOCEIHess(true)
    θu = NL.get_θ0_untransformed(fe)

    # Analytic FOCEI gradient (trace estimator, with the implicit db*/dθ term) vs finite
    # differences of the marginal objective — both w.r.t. the untransformed parameters.
    _, g, _ = NL._laplace_objective_and_grad(
        dm, batch_infos, θu, const_cache, ll_cache, mkcache();
        inner = inner_opts, hessian = method.hessian, cache_opts = method.cache, multistart = ms_opts,
        rng = Xoshiro(0), serialization = NL.EnsembleSerial(), hmode = hmode)
    obju(x) = NL._laplace_objective_only(dm, batch_infos, ComponentArray(x, getaxes(θu)),
        const_cache, ll_cache, mkcache();
        inner = inner_opts, hessian = method.hessian, cache_opts = method.cache, multistart = ms_opts,
        rng = Xoshiro(0), serialization = NL.EnsembleSerial(), hmode = hmode)
    g_fd = FiniteDifferences.grad(central_fdm(5, 1), obju, collect(θu))[1]
    @test collect(g)≈g_fd rtol=1e-4
    @test all(isfinite, collect(g))
end

@testset "FOCEI rejects unsupported (HMM) outcome distributions" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            p1_r = RealNumber(0.0)
            p2_r = RealNumber(0.1)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @formulas begin
            p1 = 0.8 / (1 + exp(-clamp(p1_r + η, -2.0, 2.0))) + 0.1
            p2 = 0.8 / (1 + exp(-clamp(p2_r, -2.0, 2.0))) + 0.1
            P = [0.85 0.15; 0.25 0.75]
            y ~ DiscreteTimeDiscreteStatesHMM(
                P, (Bernoulli(p1), Bernoulli(p2)), Categorical([0.6, 0.4]))
        end
    end
    df = DataFrame(ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0, 1, 1, 0])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm, NL.FOCEI())
end

@testset "FOCEI fits an ODE model" begin
    # Smoke test: exercise the ODE FOCEI path + accessors on the shared ODE archetype.
    res = fit_model(fx_ode_dm(),
        NL.FOCEI(multistart_n = 1, multistart_k = 1, optim_kwargs = (maxiters = 3,));
        serialization = NL.EnsembleSerial())
    @test isfinite(NL.get_objective(res))
    @test res.result.eb_modes !== nothing
    @test NL.get_random_effects(res) isa NamedTuple
end

@testset "FOCEI Hessian catches numeric failures (backtracks, no crash)" begin
    # σ = sqrt(c + η) throws a DomainError when η < -c; the FOCEI Hessian builder must
    # convert that to a NaN Hessian (→ -Inf marginal → optimiser backtracks), mirroring
    # the robustness of _laplace_logf_batch, rather than crashing the fit.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            c = RealNumber(0.4)
            ω = RealNumber(0.5, scale = :log, lower = 1e-8, upper = Inf)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, sqrt(c + η))
        end
    end
    Random.seed!(11)
    df = DataFrame(
        ID = repeat(1:6, inner = 3), t = repeat(0:2, 6), y = randn(18) .* 0.3 .+ 0.5)
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _, batch_infos, const_cache = NL._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NL.build_ll_cache(dm; force_saveat = true)
    θu = NL.get_θ0_untransformed(dm.model.fixed.fixed)
    info = batch_infos[1]
    @test all(isfinite,
        NL._focei_negH_batch(
            dm, info, θu, [0.0], const_cache, ll_cache; interaction = true))
    # Must NOT throw at the invalid point — returns NaN so the caller backtracks.
    @test all(isnan,
        NL._focei_negH_batch(
            dm, info, θu, [-1.0], const_cache, ll_cache; interaction = true))
    # A full fit must complete (the optimiser may probe the invalid region).
    res = fit_model(
        dm, NL.FOCEI(multistart_n = 1, multistart_k = 1, optim_kwargs = (maxiters = 4,));
        serialization = NL.EnsembleSerial())
    @test isfinite(NL.get_objective(res))
end

@testset "MvNormal expected information: closed form ≡ vech-basis reference" begin
    # `_focei_expected_information(::MvNormal)` computes the covariance block in
    # closed form; `_focei_vech_basis` is retained as the reference definition of
    # the vech convention. Pin their equivalence (and the zero cross blocks).
    for k in (1, 2, 3, 4), seed in 1:2
        A = randn(Xoshiro(10k + seed), k, k)
        d = MvNormal(randn(Xoshiro(seed), k), Symmetric(A * A' + k * I))
        F = NL._focei_expected_information(d)
        Σ = Matrix(cov(d))
        P = inv(Σ)
        nv = k * (k + 1) ÷ 2
        bases = NL._focei_vech_basis(k)
        Fcov_ref = [0.5 * tr(P * bases[a] * P * bases[b]) for a in 1:nv, b in 1:nv]
        @test size(F) == (k + nv, k + nv)
        @test isapprox(F[1:k, 1:k], P; rtol = 1e-12)
        @test isapprox(F[(k + 1):end, (k + 1):end], Fcov_ref; rtol = 1e-12, atol = 1e-12)
        @test all(iszero, F[1:k, (k + 1):end])
        @test all(iszero, F[(k + 1):end, 1:k])
    end
end

@testset "_focei_obs_dists_batch returns a concretely-typed vector" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(ID = repeat(1:4, inner = 3), t = repeat(0:2, 4),
        y = randn(Xoshiro(3), 12) .* 0.3 .+ 0.2)
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _, batch_infos, const_cache = NL._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = NL.build_ll_cache(dm)
    θu = NL.get_θ0_untransformed(dm.model.fixed.fixed)
    info = batch_infos[1]
    dists = NL._focei_obs_dists_batch(dm, info, θu, zeros(info.n_b), const_cache, ll_cache)
    @test isconcretetype(eltype(dists))
    @test eltype(dists) <: Normal
end
