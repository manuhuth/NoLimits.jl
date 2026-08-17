using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Lux
using ForwardDiff
using SciMLBase
using Roots
using DataInterpolations
using LinearAlgebra

import NoLimits: loglikelihood

@testset "loglikelihood non-ODE" begin
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on = :ID)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            μ = softplus(a + x.Age + η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [ComponentArray(η = 0.1), ComponentArray(η = -0.1)]

    ll1 = loglikelihood(dm, θ, η_list)
    ll2 = loglikelihood(dm, θ, η_list)
    @test ll1 == ll2
end

@testset "loglikelihood ODE" begin
    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @preDifferentialEquation begin
            pre = sat(a + η)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2 + pre
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(log1p(x1(t)^2), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.1]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η_list = [ComponentArray((η = 0.2,))]

    ll1 = loglikelihood(dm, θ, η_list)
    ll2 = loglikelihood(dm, θ, η_list)
    @test ll1 == ll2
end

@testset "loglikelihood threading (non-ODE)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(exp(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [ComponentArray((η = 0.1,)), ComponentArray((η = -0.1,))]

    ll_serial = loglikelihood(dm, θ, η_list)
    ll_thread = loglikelihood(dm, θ, η_list; serialization = EnsembleThreads())
    ll_thread_cached = loglikelihood(
        dm, θ, η_list; serialization = EnsembleThreads(),
        cache = build_ll_cache(dm; nthreads = Threads.maxthreadid())
    )
    # Regression (single-thread MCMC crash): EnsembleThreads must also accept a
    # SCALAR `_LLCache`, not only a Vector of per-thread caches. `build_ll_cache`
    # with the serial default returns one `_LLCache`; passing it with
    # EnsembleThreads previously threw `MethodError: length(::_LLCache)`. That
    # branch is only reached when `maxthreadid() == 1` (e.g. a single-thread MCMC
    # fit), so the suite (run multi-threaded) never exercised it. This assertion
    # is thread-count-independent.
    scalar_cache = build_ll_cache(dm)
    @test scalar_cache isa NoLimits._LLCache
    ll_thread_scalar = loglikelihood(
        dm, θ, η_list;
        serialization = EnsembleThreads(), cache = scalar_cache
    )
    @test ll_serial == ll_thread
    @test ll_serial == ll_thread_cached
    @test ll_serial == ll_thread_scalar
end

@testset "loglikelihood complex (NN/SoftTree/MvNormal/NPF)" begin
    chain = Chain(Dense(3, 4, tanh), Dense(4, 2))

    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            σ = RealNumber(0.4)
            ζ = NNParameters(chain; function_name = :NN1, calculate_se = false)
            Γ = SoftTreeParameters(3, 2; function_name = :ST1, calculate_se = false)
            ψ = NPFParameter(1, 3, seed = 1, calculate_se = false)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI, :CRP]; constant_on = [:ID, :SITE])
        end

        @randomEffects begin
            η_mv = RandomEffect(MvNormal(zeros(2), LinearAlgebra.I); column = :ID)
            η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :SITE)
            η_nn = RandomEffect(
                LogNormal(NN1([x.Age, x.BMI, x.CRP], ζ)[1], 0.2); column = :ID
            )
            η_st = RandomEffect(
                Gumbel(ST1([x.Age, x.BMI, x.CRP], Γ)[1], 0.3); column = :SITE
            )
        end

        @formulas begin
            μ = sat(η_mv[1] + η_mv[2]^2 + η_flow[1] + η_nn + η_st)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [30.0, 30.0, 35.0, 35.0],
        BMI = [20.0, 20.0, 22.0, 22.0],
        CRP = [1.0, 1.0, 0.9, 0.9],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [
        ComponentArray((η_mv = zeros(2), η_flow = 0.1, η_nn = 0.2, η_st = 0.3)),
        ComponentArray((η_mv = zeros(2), η_flow = 0.1, η_nn = 0.2, η_st = 0.3)),
    ]

    ll = loglikelihood(dm, θ, η_list)
end

@testset "loglikelihood complex ODE (NN/SoftTree/Spline, multi-RE)" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length = 6))

    model = @Model begin
        @helpers begin
            sat(u) = u / (1 + abs(u))
        end

        @fixedEffects begin
            a = RealNumber(0.15)
            σ = RealNumber(0.35)
            ζ = NNParameters(chain; function_name = :NN1, calculate_se = false)
            Γ = SoftTreeParameters(2, 2; function_name = :ST1, calculate_se = false)
            sp = SplineParameters(
                knots; function_name = :SP1, degree = 2, calculate_se = false
            )
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on = [:ID, :SITE])
            w = DynamicCovariate(; interpolation = LinearInterpolation)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column = :SITE)
        end

        @preDifferentialEquation begin
            pre = sat(NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1]) +
                SP1(x.Age / 100, sp) + η_id
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2 + pre + w(t) + η_site
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(log1p(x1(t)^2), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2],
        SITE = [:A, :A, :A, :B, :B, :B],
        t = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        Age = [30.0, 30.0, 30.0, 35.0, 35.0, 35.0],
        BMI = [20.0, 20.0, 20.0, 22.0, 22.0, 22.0],
        w = [0.2, 0.35, 0.5, 0.1, 0.25, 0.4],
        y = [1.0, 1.05, 1.1, 0.9, 0.95, 1.0]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η_list = [
        ComponentArray((η_id = 0.1, η_site = -0.1)),
        ComponentArray((η_id = -0.1, η_site = 0.2)),
    ]

    ll = loglikelihood(dm, θ, η_list)
end

# Shared softplus scalar-RE model for the two ForwardDiff testsets below.
softplus_fd_dm = let
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(softplus(a + η), σ)
        end
    end
    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 1.1])
    DataModel(model, df; primary_id = :ID, time_col = :t)
end

@testset "loglikelihood ForwardDiff (fixed effects)" begin
    dm = softplus_fd_dm
    θ = get_θ0_untransformed(get_model(dm).fixed.fixed)
    η_list = [ComponentArray((η = 0.1,))]

    g = ForwardDiff.gradient(x -> loglikelihood(dm, x, η_list), θ)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood ForwardDiff (random effects)" begin
    dm = softplus_fd_dm
    θ = get_θ0_untransformed(get_model(dm).fixed.fixed)
    η0 = ComponentArray((η = 0.1,))

    g = ForwardDiff.gradient(η -> loglikelihood(dm, θ, [η]), η0)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood ODE ForwardDiff (fixed effects)" begin
    dm = fx_ode_dm()   # two individuals, so η_list has two entries
    θ = get_θ0_untransformed(get_model(dm).fixed.fixed)
    η_list = [ComponentArray((η = 0.1,)), ComponentArray((η = -0.1,))]

    g = ForwardDiff.gradient(x -> loglikelihood(dm, x, η_list), θ)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood ODE ForwardDiff (random effects)" begin
    dm = fx_ode_dm()
    θ = get_θ0_untransformed(get_model(dm).fixed.fixed)
    η0 = ComponentArray((η = 0.1,))

    g = ForwardDiff.gradient(η -> loglikelihood(dm, θ, [η, η]), η0)
    @test g isa ComponentArray
    @test all(isfinite, collect(g))
end

@testset "loglikelihood skips missing scalar observables (non-ODE regression)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.2)
            b = RealNumber(-0.3)
            σy = RealNumber(0.5)
            σz = RealNumber(0.7)
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            μ = a + b * t
            y ~ Normal(μ, σy)
            z ~ Normal(μ + 1.0, σz)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = Union{Missing, Float64}[1.1, missing, missing],
        z = Union{Missing, Float64}[2.2, 2.0, missing]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    ll = loglikelihood(dm, θ, ComponentArray())

    μ1 = 1.2
    μ2 = 0.9
    ll_expected = logpdf(Normal(μ1, 0.5), 1.1) +
        logpdf(Normal(μ1 + 1.0, 0.7), 2.2) +
        logpdf(Normal(μ2 + 1.0, 0.7), 2.0)
    @test ll ≈ ll_expected atol = 1.0e-12
end

@testset "loglikelihood skips missing scalar observables (ODE regression)" begin
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(0.0)
            σy = RealNumber(0.2)
            σz = RealNumber(0.3)
        end

        @covariates begin
            t = Covariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -k * x1
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σy)
            z ~ Normal(2.0 * x1(t), σz)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = Union{Missing, Float64}[1.1, missing, 0.95, missing],
        z = Union{Missing, Float64}[2.05, 1.9, missing, missing]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    ll = loglikelihood(dm, θ, ComponentArray())

    ll_expected = logpdf(Normal(1.0, 0.2), 1.1) +
        logpdf(Normal(2.0, 0.3), 2.05) +
        logpdf(Normal(2.0, 0.3), 1.9) +
        logpdf(Normal(1.0, 0.2), 0.95)
    @test ll ≈ ll_expected atol = 1.0e-12
end

@testset "loglikelihood non-ODE uses row-specific random effects for varying groups" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(0.2)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
        end

        @formulas begin
            y ~ Normal(a + η_year, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 2, 2],
        YEAR = [:A, :B, :B, :A, :C],
        t = [0.0, 1.0, 2.0, 0.0, 1.0],
        y = [0.05, 0.55, 0.35, -0.15, 0.2]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [
        ComponentArray((; η_year = [0.1, 0.4])),
        ComponentArray((; η_year = [0.1, 0.3])),
    ]

    ll = loglikelihood(dm, θ, η_list)
    ll_expected = logpdf(Normal(0.1, 0.2), 0.05) +
        logpdf(Normal(0.4, 0.2), 0.55) +
        logpdf(Normal(0.4, 0.2), 0.35) +
        logpdf(Normal(0.1, 0.2), -0.15) +
        logpdf(Normal(0.3, 0.2), 0.2)

    @test NoLimits._needs_rowwise_random_effects(dm, 1; obs_only = true)
    @test NoLimits._needs_rowwise_random_effects(dm, 2; obs_only = true)
    @test ll ≈ ll_expected atol = 1.0e-12
end

# A numeric error raised by the model itself (here `log` of a negative argument) must
# degrade to a -Inf likelihood, not kill the fit: the SAEM/MCEM E-step has no handler of
# its own, so an out-of-domain random-effect draw used to abort the whole run.
@testset "numeric error in the model degrades to -Inf" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(log(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.15, 0.25]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))

    ok = NoLimits.conditional_loglikelihood(dm, 1, θ, ComponentArray(η = 0.0))
    @test isfinite(ok)
    # a + η = 1 - 2 < 0 -> log throws inside the formula
    bad = NoLimits.conditional_loglikelihood(dm, 1, θ, ComponentArray(η = -2.0))
    @test bad == -Inf
end

# The sibling of the above for the RE prior: `_loglikelihood_individual`'s guard does not cover
# it, so before this guard an invertible-flow prior failing on a bad proposal killed the whole
# fit (a normalizing-flow SAEM fit died outright) instead of scoring the point -Inf.
@testset "numeric error in the RE prior degrades to -Inf" begin
    # `Roots.ConvergenceFailed` is what an invertible-flow prior raises when its bracketing
    # root-find gives up, and it is not a subtype of any other whitelisted error.
    @test NoLimits._is_numeric_error(Roots.ConvergenceFailed("failed"))
    @test NoLimits._is_numeric_error(DomainError(-1.0, "x"))
    @test NoLimits._is_numeric_error(ArgumentError("x"))
    @test !NoLimits._is_numeric_error(MethodError(sqrt, (nothing,)))

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            # `sqrt(a)` throws once the optimizer pushes `a` negative, inside the RE prior
            η = RandomEffect(Normal(0.0, sqrt(a)); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.15, 0.25]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    θ = get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm)))
    _, infos, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
    cache = NoLimits.build_likelihood_cache(dm; force_saveat = true)
    batch = infos[1]
    b = fill(0.0, NoLimits.get_batch_re_dim(batch))

    @test isfinite(NoLimits.re_logprior(dm, batch, θ, b; const_cache = cc, cache = cache))

    θ_bad = deepcopy(θ)
    θ_bad.a = -1.0                      # sqrt(-1) throws inside the RE distribution
    @test NoLimits.re_logprior(dm, batch, θ_bad, b; const_cache = cc, cache = cache) == -Inf
end

# ── Invariance oracles ───────────────────────────────────────────────────────
#
# Properties that hold whatever the fit's quality is, so the fixtures stay tiny
# and the tolerance sits on the INVARIANCE rather than on the estimate. These are
# the cheap CI stand-ins for oracles that until now only the external stress run
# (533 cells, hours on a cluster) checked. GHQuadrature's own serial-vs-threaded
# and permutation guards live in estimation_ghquadrature_tests.jl (#151).

const _INV_SER = NoLimits.EnsembleSerial()

@testset "objective is invariant to EnsembleSerial vs EnsembleThreads" begin
    # Issue #151: GHQuadrature accumulated batch marginals in completion order under
    # EnsembleThreads, so its objective moved with the thread schedule (relative gaps
    # up to 0.74). Nothing in the suite asserted this for any estimator.
    if Threads.nthreads() < 2
        @info "SKIPPED serial-vs-threaded invariance: needs Threads.nthreads() > 1, " *
            "got $(Threads.nthreads()). Run julia with `-t auto` to exercise it."
        @test true
    else
        # Serial arm = the shared fixture fit, so the method must mirror fixtures.jl.
        cases = (
            (
                "MLE", fx_mle(), fx_nore_dm(),
                NoLimits.MLE(; optim_kwargs = (maxiters = 3,)),
            ),
            (
                "MAP", fx_map(), fx_nore_prior_dm(),
                NoLimits.MAP(; optim_kwargs = (maxiters = 3,)),
            ),
            (
                "Laplace", fx_laplace(), fx_re_dm(),
                NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)),
            ),
            (
                "FOCEI", fx_focei(), fx_re_dm(),
                NoLimits.FOCEI(;
                    multistart_n = 1, multistart_k = 1,
                    optim_kwargs = (maxiters = 3,)
                ),
            ),
            (
                "Pooled", fx_pooled(), fx_re_dm(),
                NoLimits.Pooled(; optim_kwargs = (maxiters = 3,)),
            ),
        )
        # rtol: MLE/MAP/FOCEI/Pooled came out bit-identical over repeats, but the
        # threaded Laplace intermittently drifts (worst observed 9e-10 relative) --
        # the thread schedule perturbs the EB modes and the outer optimizer walks a
        # marginally different path from there. #151-class breakage is O(0.1).
        for (name, res_serial, dm, method) in cases
            @testset "$name" begin
                res_threaded = fit_model(dm, method; serialization = EnsembleThreads())
                @test isapprox(
                    get_objective(res_serial), get_objective(res_threaded);
                    rtol = 1.0e-6
                )
            end
        end
    end
end

@testset "marginal-likelihood accessor agrees with the fit objective" begin
    # Issue #98: the GHQ fit integrated against the mode-centred measure while the
    # accessor still used the prior-centred one, whose signed Smolyak sum drifts with
    # the level and can turn negative. Pinning the two together catches that directly.
    res_g = fx_ghq()                       # GHQuadrature(level = 2), no penalty
    ml2 = get_marginal_likelihood(res_g; level = 2, serialization = _INV_SER)
    @test isapprox(ml2, -get_objective(res_g); rtol = 1.0e-10)
    # AGHQ is exact for this linear-Gaussian fixture at any level; the prior-centred
    # rule of #98 drifted away as the level rose.
    @test isapprox(
        get_marginal_likelihood(res_g; level = 3, serialization = _INV_SER),
        ml2; rtol = 1.0e-8
    )

    # Laplace: the accessor re-solves the EB modes instead of reusing the fit's, and
    # -½logdet(-H(b*)) is not stationary in b*, so the gap tracks the mode tolerance
    # (measured 1.9e-10 relative at -O0). A #98-shaped measure mismatch is O(1).
    res_l = fx_laplace()
    θ_l = NoLimits.get_params(res_l; scale = :untransformed)
    @test isapprox(
        NoLimits.laplace_marginal(fx_re_dm(), θ_l; serialization = _INV_SER),
        -get_objective(res_l); rtol = 1.0e-6
    )

    # Issue #171: RE levels pinned via `constants_re` are not integrated over, so their
    # prior density has to be added back. It was dropped, which made the accessor a
    # conditional likelihood in those levels (its η-derivative lost the -η prior term).
    # With every level pinned nothing is left to integrate, so the accessor must equal
    # the complete-data joint at the same values.
    res_c = fx_constre_laplace()      # symbol IDs, so RE levels can be pinned
    dm_c = fx_constre_dm()
    θ_c = NoLimits.get_params(res_c; scale = :untransformed)
    re_tab = get_random_effects(dm_c, res_c).η
    cre = (;
        η = NamedTuple(
            Symbol(id) => v
                for (id, v) in zip(re_tab[!, :ID], re_tab[!, :η_1])
        ),
    )
    joint = sum(
        complete_data_loglikelihood(dm_c, i, θ_c, (; η = re_tab[i, :η_1]))
            for i in 1:length(get_individuals(dm_c))
    )
    @test isapprox(
        get_marginal_likelihood(dm_c, res_c; constants_re = cre, serialization = _INV_SER),
        joint; rtol = 1.0e-8
    )
end

@testset "objectives are finite at the fitted estimate" begin
    # 32 GHQuadrature cells of the external stress run reported `Inf` objectives while
    # CI stayed green, because nothing here asserted finiteness on a good fixture.
    for (name, res) in (
            ("MLE", fx_mle()), ("MAP", fx_map()), ("VI", fx_vi()),
            ("Pooled", fx_pooled()), ("Laplace", fx_laplace()), ("FOCEI", fx_focei()),
            ("GHQuadrature", fx_ghq()), ("SAEM", fx_saem()), ("MCEM", fx_mcem()),
        )
        @testset "$name" begin
            @test isfinite(get_objective(res))
        end
    end
    for (name, res) in (
            ("Laplace", fx_laplace()), ("FOCEI", fx_focei()),
            ("GHQuadrature", fx_ghq()), ("SAEM", fx_saem()), ("MCEM", fx_mcem()),
        )
        @testset "$name likelihoods" begin
            @test isfinite(get_loglikelihood(res))
            @test isfinite(
                get_marginal_likelihood(
                    res; level = 2, serialization = _INV_SER
                )
            )
        end
    end
end

@testset "censored outcome distributions" begin
    # Issue #197: any third-party `Distribution` with a `logpdf` works as an outcome
    # without integration code -- `Distributions.censored` stands in here for
    # CensoredDistributions.jl, which exercises the same generic dispatch.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.6; scale = :log)
            ω = RealNumber(0.4; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            μ = a + η + 0.1 * t
            y ~ censored(Normal(μ, σ); upper = 2.0)
        end
    end
    df = DataFrame(
        ID = repeat(1:6, inner = 3), t = repeat([0.0, 1.0, 2.0], 6),
        y = [
            1.0, 1.4, 2.0, 0.7, 1.2, 1.9, 1.3, 2.0, 2.0, 0.9, 1.1, 1.6,
            1.5, 1.8, 2.0, 0.6, 1.0, 1.7,
        ]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    for (name, method) in (
            ("Laplace", NoLimits.Laplace(; optim_kwargs = (maxiters = 3,))),
            (
                "GHQuadrature",
                NoLimits.GHQuadrature(;
                    level = 2,
                    optim_kwargs = (maxiters = 3,)
                ),
            ),
            ("Pooled", NoLimits.Pooled(; optim_kwargs = (maxiters = 3,))),
        )
        @testset "$name" begin
            @test isfinite(get_objective(fit_model(dm, method)))
        end
    end

    # FOCEI dispatches on a fixed family whitelist for its Fisher-information
    # surrogate, so it rejects censored outcomes by design rather than silently.
    @test_throws ErrorException fit_model(dm, NoLimits.FOCEI())
end

# Invalid iteration/chain/sample counts used to construct fine and fail deep inside the
# fit (BoundsError, 0/0, or a silently substituted default) — reject them up front (#226).
@testset "estimator options reject non-positive counts" begin
    @test_throws ErrorException NoLimits.SAEM(; maxiters = 0)
    @test_throws ErrorException NoLimits.SAEM(; n_chains = 0)
    @test_throws ErrorException NoLimits.SAEM(; kappa = 0.0)
    @test_throws ErrorException NoLimits.SAEM(; consecutive_params = 0)
    @test_throws ErrorException NoLimits.SAEM(; t0 = -1)
    @test_throws ErrorException NoLimits.SAEM(; mcmc_steps = 0)
    @test_throws ErrorException NoLimits.MCEM(; maxiters = 0)
    @test_throws ErrorException NoLimits.MCEM(; consecutive_params = 0)
    @test_throws ErrorException NoLimits.MCEM(; sample_schedule = 0)
    @test_throws ErrorException NoLimits.MCEM(; sample_schedule = Int[])
    @test_throws ErrorException NoLimits.MCEM(; sample_schedule = [10, 0])
    @test_throws ErrorException NoLimits.EBEOptions(; multistart_n = 0)
    @test_throws ErrorException NoLimits.EBEOptions(; multistart_n = 5, multistart_k = 6)
    @test_throws ErrorException NoLimits.EBEOptions(; max_rounds = 0)
    @test_throws ArgumentError NoLimits.GHQuadrature(; level = 0)
    @test_throws ArgumentError NoLimits.GHQuadrature(; level = (η = 0,))
    # Valid settings still construct.
    @test NoLimits.SAEM(; maxiters = 3) isa NoLimits.SAEM
    @test NoLimits.MCEM(; sample_schedule = [5, 7]) isa NoLimits.MCEM
end

# Misspelled symbols used to silently select a different algorithm, and out-of-domain
# schedule/annealing values silently changed the estimator (#229).
@testset "estimator options reject invalid schedule settings" begin
    @test_throws ErrorException NoLimits.SAEM(; sa_schedule = :bad)
    @test_throws ErrorException NoLimits.SAEM(; sa_schedule = :custom)
    @test_throws ErrorException NoLimits.SAEM(; sa_phase2_kappa = NaN)
    @test_throws ErrorException NoLimits.SAEM(; sa_phase2_kappa = 1.0)
    @test_throws ErrorException NoLimits.SAEM(; sa_anneal_schedule = :exponetial)
    @test_throws ErrorException NoLimits.SAEM(; sa_anneal_alpha = 2.0)
    @test_throws ErrorException NoLimits.SAEM(; sa_anneal_iters = -5)
    @test_throws ErrorException NoLimits.SAEM(; small_n_chain_target = 0)
    @test_throws ErrorException NoLimits.SAEM(; anneal_min_sd = 0.0)
    @test_throws ErrorException NoLimits.SAEM(; var_lb_value = 0.0)
    @test_throws ErrorException NoLimits.MCMC(; turing_kwargs = (n_adapt = -1,))
    @test_throws ErrorException NoLimits.MCMC(; turing_kwargs = (n_samples = 0,))
    @test_throws ErrorException NoLimits.Laplace(; theta_tol = -1.0)
    # sa_anneal_alpha = 0 stays valid: it is the documented way to disable annealing.
    @test NoLimits.SAEM(; sa_anneal_alpha = 0.0) isa NoLimits.SAEM

    # A custom γ schedule is checked where it is used, not just at construction.
    bad = NoLimits.SAEM(; sa_schedule = :custom, sa_schedule_fn = (i, o) -> 2.0).saem
    @test_throws ErrorException NoLimits._saem_gamma_schedule(1, bad)
    good = NoLimits.SAEM(; sa_schedule = :custom, sa_schedule_fn = (i, o) -> 0.25).saem
    @test NoLimits._saem_gamma_schedule(1, good) == 0.25
end

# Symbol-valued options also accept strings, so the Python/R bindings need no
# per-function conversion (#255). Invalid strings must fail exactly like invalid symbols.
@testset "symbol options accept strings (#255)" begin
    @test NoLimits._as_symbol("abc") === :abc
    @test NoLimits._as_symbol(:abc) === :abc
    @test NoLimits._as_symbol(nothing) === nothing

    s_str = NoLimits.SAEM(;
        sa_schedule = "two_phase", sa_anneal_schedule = "linear",
        update_schedule = "all", builtin_stats = "auto", resid_var_param = "σ",
        ebe_multistart_sampling = "random", ebe_rescue_multistart_sampling = "random"
    ).saem
    s_sym = NoLimits.SAEM(;
        sa_schedule = :two_phase, sa_anneal_schedule = :linear,
        update_schedule = :all, builtin_stats = :auto, resid_var_param = :σ,
        ebe_multistart_sampling = :random, ebe_rescue_multistart_sampling = :random
    ).saem
    @test s_str.sa_schedule === s_sym.sa_schedule
    @test s_str.sa_anneal_schedule === s_sym.sa_anneal_schedule
    @test s_str.update_schedule === s_sym.update_schedule
    @test s_str.builtin_stats === s_sym.builtin_stats
    @test s_str.resid_var_param === s_sym.resid_var_param
    @test s_str.ebe_multistart_sampling === s_sym.ebe_multistart_sampling
    @test s_str.ebe_rescue.sampling === s_sym.ebe_rescue.sampling
    @test_throws ErrorException NoLimits.SAEM(; sa_schedule = "bad")
    @test_throws ErrorException NoLimits.SAEM(; sa_anneal_schedule = "exponetial")

    l_str = NoLimits.Laplace(; multistart_sampling = "random", nan_recovery = "error")
    l_sym = NoLimits.Laplace(; multistart_sampling = :random, nan_recovery = :error)
    @test l_str.multistart.sampling === l_sym.multistart.sampling
    @test l_str.nan_recovery === l_sym.nan_recovery

    for (T, kw) in (
            (NoLimits.FOCEI, :multistart_sampling), (NoLimits.GHQuadrature, :multistart_sampling),
        )
        @test getfield(T(; kw => "random").multistart, :sampling) ===
            getfield(T(; kw => :random).multistart, :sampling)
    end

    m_str = NoLimits.MCEM(; ebe_multistart_sampling = "random", update_schedule = "all")
    m_sym = NoLimits.MCEM(; ebe_multistart_sampling = :random, update_schedule = :all)
    @test m_str.ebe.sampling === m_sym.ebe.sampling
    @test m_str.update_schedule === m_sym.update_schedule
    @test NoLimits.MCEM_IS(; proposal = "prior").proposal ===
        NoLimits.MCEM_IS(; proposal = :prior).proposal

    @test NoLimits.EBEOptions(; sampling = "random").sampling ===
        NoLimits.EBEOptions(; sampling = :random).sampling
    @test NoLimits.MCIntegrator(; mode = "prior").mode ===
        NoLimits.MCIntegrator(; mode = :prior).mode
    @test_throws ErrorException NoLimits.MCIntegrator(; mode = "turnig")

    ms_str = NoLimits.Multistart(; sampling = "lhs", screening = "ebe")
    ms_sym = NoLimits.Multistart(; sampling = :lhs, screening = :ebe)
    @test (ms_str.sampling, ms_str.screening) === (ms_sym.sampling, ms_sym.screening)
    for T in (NoLimits.Pooled, NoLimits.PooledMap)
        @test T(; refreeze_check = "refit").refreeze_check ===
            T(; refreeze_check = :refit).refreeze_check
        @test_throws ErrorException T(; refreeze_check = "warm")
    end

    dm = fx_nore_dm()
    @test NoLimits.get_params(dm; scale = "untransformed") ==
        NoLimits.get_params(dm; scale = :untransformed)
    @test_throws ErrorException NoLimits.get_params(dm; scale = "untransfomed")
    res = fx_mle()
    @test NoLimits.get_params(res; scale = "transformed") ==
        NoLimits.get_params(res; scale = :transformed)
    @test_throws ErrorException NoLimits.get_params(res; scale = "transfomed")

    m_str2 = set_solver_config(fx_nore_model(); saveat_mode = "auto", closed_form = "off")
    m_sym2 = set_solver_config(fx_nore_model(); saveat_mode = :auto, closed_form = :off)
    @test get_solver_config(m_str2).saveat_mode === get_solver_config(m_sym2).saveat_mode
    @test get_solver_config(m_str2).closed_form === get_solver_config(m_sym2).closed_form
end

@testset "boundary validation (#240 Group 2)" begin
    # ── Normalizing planar flow constructor and scalar density pair (#235.12-15) ──
    @test_throws ArgumentError NormalizingPlanarFlow(0, 2)
    @test_throws ArgumentError NormalizingPlanarFlow(2, -1)
    @test_throws ArgumentError NormalizingPlanarFlow(
        2, 2; base_dist = MvNormal(zeros(3), I)
    )
    flow = NormalizingPlanarFlow(1, 2)
    @test pdf(flow, 0.3) ≈ exp(logpdf(flow, 0.3))
    @test_throws ErrorException NPFParameter(2, 2; base_dist = MvNormal(zeros(3), I))

    # ── Shared GHQ (dimension, level) validation (#235.17-18, 23-24) ─────────────
    for f in (
            NoLimits.build_sparse_grid, NoLimits.get_sparse_grid,
            NoLimits.n_ghq_points, NoLimits.ghq_points_bound,
        )
        @test_throws ArgumentError f(0, 2)
        @test_throws ArgumentError f(2, 0)
        @test_throws ArgumentError f(-1, -1)
    end
    @test_throws ArgumentError NoLimits.get_anisotropic_grid(Int[], Int[])
    @test_throws ArgumentError NoLimits.get_anisotropic_grid([1, 2], [1])

    # Any `Integer` level normalizes to `Int`, so it keeps the isotropic path (#235.21-22).
    @test_throws ArgumentError GHQuadrature(level = 0)
    @test_throws ArgumentError GHQuadrature(level = 2.5)
    @test_throws ArgumentError GHQuadrature(level = Int[])
    @test_throws ArgumentError GHQuadrature(level = (; η = 0))
    @test GHQuadrature(level = Int8(3)).level === 3
    @test GHQuadrature(level = UInt(3)).level === 3
    @test GHQuadrature(level = Int8[2, 3]).level == Int[2, 3]
    @test NoLimits.n_ghq_points(Int8(2), UInt(3)) == NoLimits.n_ghq_points(2, 3)
    @test NoLimits.get_sparse_grid(Int8(2), UInt(3)).nodes ==
        NoLimits.get_sparse_grid(2, 3).nodes

    # ── Plot grid sizing (#235.19-20) ───────────────────────────────────────────
    @test_throws ErrorException NoLimits.calculate_plot_size(0, 2)
    @test_throws ErrorException NoLimits.calculate_plot_size(-3, 2)
    @test NoLimits.calculate_plot_size(4, 2) isa Tuple{Int, Int}

    # ── MCMC result boundary (#235.25-26) ───────────────────────────────────────
    chain = NoLimits.MCMCChains.Chains(randn(4, 1, 1), [:a])
    dm_re = fx_re_dm()
    @test_throws ArgumentError NoLimits.build_fit_result(
        dm_re, MLE(), chain; sampler = nothing, n_samples = 0
    )
    @test_throws ArgumentError NoLimits.build_fit_result(
        dm_re, MLE(), chain; sampler = nothing, n_samples = 10, n_adapt = -1
    )
    @test_throws ArgumentError NoLimits.build_fit_result(
        dm_re, MLE(), chain; sampler = nothing, n_samples = 10, n_adapt = 4
    )

    # ── Non-finite values behind a broad eltype, and EVID (#235.29-30) ──────────
    _df(y) = DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = y)
    @test_throws ErrorException DataModel(
        fx_re_model(), _df(Any[0.1, NaN, 0.2, 0.3]); primary_id = :ID, time_col = :t
    )
    @test_throws ErrorException DataModel(
        fx_re_model(), _df(Any[0.1, Inf, 0.2, 0.3]); primary_id = :ID, time_col = :t
    )
    @test DataModel(
        fx_re_model(), _df(Any[0.1, 0.2, 0.3, 0.4]); primary_id = :ID, time_col = :t
    ) isa DataModel

    ev_df(evid) = DataFrame(
        ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.3, 0.4],
        EVID = evid, AMT = zeros(4), RATE = zeros(4), CMT = ones(Int, 4)
    )
    ev_dm(evid) = DataModel(
        fx_re_model(), ev_df(evid); primary_id = :ID, time_col = :t, evid_col = :EVID
    )
    @test ev_dm([0, 4, 0, 0]) isa DataModel     # nonzero event codes stay legal
    @test_throws ErrorException ev_dm([0.0, NaN, 0.0, 0.0])
    @test_throws ErrorException ev_dm([0.0, Inf, 0.0, 0.0])
    @test_throws ErrorException ev_dm(["a", "b", "a", "a"])

    # ── Event-only individuals no longer index an empty observation set (#237.29) ─
    df_ev = copy(fx_re_df())
    n = nrow(df_ev)
    df_ev.EVID = [id == 6 ? 1 : 0 for id in df_ev.ID]
    df_ev.AMT = zeros(n)
    df_ev.RATE = zeros(n)
    df_ev.CMT = ones(Int, n)
    dm_ev = DataModel(
        fx_re_model(), df_ev; primary_id = :ID, time_col = :t, evid_col = :EVID
    )
    @test isempty(NoLimits.get_obs_rows(NoLimits.get_row_groups(dm_ev))[6])
    @test NoLimits._individual_id(dm_ev, 6) == 6
    η_ev = get_random_effects(dm_ev, fx_laplace(), :η)
    @test length(η_ev) == length(get_individuals(dm_ev))
    @test all(isfinite, η_ev)

    # ── GHQ log-likelihood honours `serialization` (#237.30) ────────────────────
    # `serialization` is load-bearing at that call site: it decides whether the cache is
    # a single object or one per chunk, and the branch must reduce either to the same
    # number. Before the fix the keyword was dropped and the cache was always serial.
    @test NoLimits.build_ll_cache(
        dm_re; serialization = NoLimits.EnsembleThreads(), nthreads = 3,
        force_saveat = true
    ) isa AbstractVector
    res_g2 = fx_ghq()
    ll_ser = get_loglikelihood(dm_re, res_g2; serialization = NoLimits.EnsembleSerial())
    ll_thr = get_loglikelihood(dm_re, res_g2; serialization = NoLimits.EnsembleThreads())
    @test isfinite(ll_ser)
    @test ll_thr ≈ ll_ser
end

@testset "Poisson fast path at the λ=0 boundary (#249)" begin
    # 0 * log(0) used to be NaN, which the accumulator turned into -Inf for a case
    # Distributions scores at 0.0.
    @test NoLimits._fast_logpdf(Poisson(0.0), 0.0) == logpdf(Poisson(0.0), 0)
    @test NoLimits._fast_logpdf(Poisson(0.0), 0.0) == 0.0
    @test NoLimits._fast_logpdf(Poisson(0.0), 1.0) == -Inf
    for λ in (0.0, 1.0e-12, 0.5, 3.0), y in (0.0, 1.0, 4.0)
        @test NoLimits._fast_logpdf(Poisson(λ), y) ≈ logpdf(Poisson(λ), Int(y)) atol = 1.0e-10
    end
    # d/dλ logpdf(Poisson(λ), 0) = -1 at λ = 0; the NaN branch poisoned the Dual.
    g = ForwardDiff.derivative(λ -> NoLimits._fast_logpdf(Poisson(λ), 0.0), 0.0)
    @test g == -1.0
    @test ForwardDiff.derivative(λ -> NoLimits._fast_logpdf(Poisson(λ), 2.0), 1.5) ≈ 2 / 1.5 - 1
end

# A dispatch-visible ensemble type: `_ensemble_nthreads` is the one place the
# `serialization` object is dispatched on, so a spy can record that the exact object
# passed by the caller reached the cache builder (#244).
struct _SpyEnsemble <: SciMLBase.EnsembleAlgorithm end
const _SPY_CALLS = Ref(0)
NoLimits._ensemble_nthreads(::_SpyEnsemble) = (_SPY_CALLS[] += 1; Threads.maxthreadid())

@testset "GHQ get_loglikelihood dispatches the supplied ensemble (#244)" begin
    dm = fx_re_dm()
    res = fx_ghq()
    _SPY_CALLS[] = 0
    ll_spy = get_loglikelihood(dm, res; serialization = _SpyEnsemble())
    @test _SPY_CALLS[] == 1
    @test isfinite(ll_spy)
    @test ll_spy ≈ get_loglikelihood(dm, res; serialization = _INV_SER)
end

@testset "get_loglikelihood_quadrature accepts any Integer level (#243)" begin
    dm = fx_re_dm()
    res = fx_ghq()
    ll = get_loglikelihood_quadrature(dm, res; level = 2, serialization = _INV_SER)
    @test get_loglikelihood_quadrature(
        dm, res; level = Int8(2), serialization = _INV_SER
    ) == ll
    @test get_loglikelihood_quadrature(
        dm, res; level = UInt(2), serialization = _INV_SER
    ) == ll
    @test_throws ArgumentError get_loglikelihood_quadrature(
        dm, res; level = 0, serialization = _INV_SER
    )
end
