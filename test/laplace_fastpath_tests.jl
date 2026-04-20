using Test
using DataFrames
using NoLimits
using Distributions
using Random
using ComponentArrays
using Optimization
using OptimizationOptimJL
using LineSearches

# ─────────────────────────────────────────────────────────────────────────────
# Merged from: laplace_fastpath_baseline_tests.jl, laplace_fastpath_config_tests.jl
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: helpers
# ══════════════════════════════════════════════════════════════════════════════

function _lfp_make_df(kind::Symbol; n_id::Int=12, n_obs::Int=5, seed::Int=123)
    rng = MersenneTwister(seed)
    n = n_id * n_obs
    ID = repeat(1:n_id, inner=n_obs)
    t = repeat(collect(0.0:(n_obs - 1)), n_id)
    z = randn(rng, n)
    η = 0.6 .* randn(rng, n_id)

    if kind == :gaussian
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            lin = 0.2 + 0.7 * z[i] + η[ID[i]]
            y[i] = lin + 0.3 * randn(rng)
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :lognormal
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            lin = -0.1 + 0.5 * z[i] + η[ID[i]]
            y[i] = exp(lin + 0.25 * randn(rng))
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :bernoulli
        y = Vector{Int}(undef, n)
        for i in eachindex(y)
            lin = 0.1 + 0.6 * z[i] + η[ID[i]]
            p = inv(1 + exp(-lin))
            y[i] = rand(rng) < p ? 1 : 0
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :poisson
        y = Vector{Int}(undef, n)
        for i in eachindex(y)
            lin = -0.2 + 0.4 * z[i] + 0.7 * η[ID[i]]
            y[i] = rand(rng, Poisson(exp(lin)))
        end
        return DataFrame(ID=ID, t=t, z=z, y=y)
    elseif kind == :ode_offset
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            x1t = exp(-0.3 * t[i])
            y[i] = x1t + η[ID[i]] + 0.2 * randn(rng)
        end
        return DataFrame(ID=ID, t=t, y=y)
    elseif kind == :ode_eta
        y = Vector{Float64}(undef, n)
        for i in eachindex(y)
            x1t = exp(-(0.3 + 0.2 * η[ID[i]]) * t[i])
            y[i] = x1t + 0.2 * randn(rng)
        end
        return DataFrame(ID=ID, t=t, y=y)
    else
        error("Unknown fixture kind: $(kind)")
    end
end

function _lfp_model(kind::Symbol)
    if kind == :gaussian
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.5, scale=:log)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η, σ)
            end
        end
    elseif kind == :lognormal
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                σ = RealNumber(0.4, scale=:log)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                y ~ LogNormal(a + b * z + η, σ)
            end
        end
    elseif kind == :bernoulli
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                p = logistic(a + b * z + η)
                y ~ Bernoulli(p)
            end
        end
    elseif kind == :poisson
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.0)
                b = RealNumber(0.4)
                τ = RealNumber(0.7, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @formulas begin
                λ = exp(a + b * z + η)
                y ~ Poisson(λ)
            end
        end
    elseif kind == :ode_offset
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log)
                σ = RealNumber(0.3, scale=:log)
                τ = RealNumber(0.6, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -k * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t) + η, σ)
            end
        end
    elseif kind == :ode_eta
        return @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                k = RealNumber(0.3, scale=:log)
                σ = RealNumber(0.3, scale=:log)
                τ = RealNumber(0.6, scale=:log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, τ); column=:ID)
            end
            @DifferentialEquation begin
                D(x1) ~ -(k + η) * x1
            end
            @initialDE begin
                x1 = 1.0
            end
            @formulas begin
                y ~ Normal(x1(t), σ)
            end
        end
    else
        error("Unknown model kind: $(kind)")
    end
end

function _lfp_setup_objgrad(dm::DataModel)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θ0 = get_θ0_untransformed(dm.model.fixed.fixed)
    n_batches = length(batch_infos)
    Tθ = eltype(θ0)
    bstar_cache = NoLimits._LaplaceBStarCache([Vector{Tθ}() for _ in 1:n_batches], falses(n_batches))
    grad_cache = NoLimits._LaplaceGradCache([Vector{Tθ}() for _ in 1:n_batches],
                                            fill(Tθ(NaN), n_batches),
                                            [Vector{Tθ}() for _ in 1:n_batches],
                                            falses(n_batches))
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(Tθ, n_batches)
    ebe_cache = NoLimits._LaplaceCache(nothing, bstar_cache, grad_cache, ad_cache, hess_cache)
    return (; batch_infos, const_cache, ll_cache, θ0, ebe_cache)
end

function _lfp_eval_objgrad(dm::DataModel; seed::Int=44)
    s = _lfp_setup_objgrad(dm)
    inner = NoLimits.LaplaceInnerOptions(
        OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking(maxstep=1.0)),
        (maxiters=2,),
        Optimization.AutoForwardDiff(),
        1e-6
    )
    hess = NoLimits.LaplaceHessianOptions(1e-6, 6, 10.0, true, 1e-6, true, false, 8)
    cache_opts = NoLimits.LaplaceCacheOptions(0.0)
    ms = NoLimits.LaplaceMultistartOptions(0, 0, 1e-6, 1, :lhs)
    rng = MersenneTwister(seed)
    obj, grad, _ = NoLimits._laplace_objective_and_grad(dm,
                                                         s.batch_infos,
                                                         s.θ0,
                                                         s.const_cache,
                                                         s.ll_cache,
                                                         s.ebe_cache;
                                                         inner=inner,
                                                         hessian=hess,
                                                         cache_opts=cache_opts,
                                                         multistart=ms,
                                                         rng=rng,
                                                         serialization=EnsembleSerial())
    return obj, collect(grad)
end

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG: helpers
# ══════════════════════════════════════════════════════════════════════════════

function _lfpc_model_normal(; with_priors::Bool=false)
    with_priors || return _lfp_model(:gaussian)
    return @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.7))
            τ = RealNumber(0.6, scale=:log, prior=LogNormal(0.0, 0.7))
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + b * z + η, σ)
        end
    end
end

function _lfpc_model_normal_two_re(; with_priors::Bool=false)
    if with_priors
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.1, prior=Normal(0.0, 1.0))
                b = RealNumber(0.2, prior=Normal(0.0, 1.0))
                σ = RealNumber(0.4, scale=:log, prior=LogNormal(0.0, 0.7))
                τ0 = RealNumber(0.6, scale=:log, prior=LogNormal(0.0, 0.7))
                τ1 = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.7))
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η0 + z * η1, σ)
            end
        end
    else
        return @Model begin
            @covariates begin
                t = Covariate()
                z = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.1)
                b = RealNumber(0.2)
                σ = RealNumber(0.4, scale=:log)
                τ0 = RealNumber(0.6, scale=:log)
                τ1 = RealNumber(0.5, scale=:log)
            end
            @randomEffects begin
                η0 = RandomEffect(Normal(0.0, τ0); column=:ID)
                η1 = RandomEffect(Normal(0.0, τ1); column=:ID)
            end
            @formulas begin
                y ~ Normal(a + b * z + η0 + z * η1, σ)
            end
        end
    end
end

# Alias used by Step4 test.
_lfpc_model(; with_priors::Bool=false) = _lfpc_model_normal(; with_priors=with_priors)

_lfpc_model_lognormal() = _lfp_model(:lognormal)

_lfpc_model_bernoulli() = _lfp_model(:bernoulli)

_lfpc_model_poisson() = _lfp_model(:poisson)

_lfpc_model_ode_offset() = _lfp_model(:ode_offset)
_lfpc_model_ode_eta()    = _lfp_model(:ode_eta)

function _lfpc_model_re_lognormal()
    return @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
            μη = RealNumber(0.0)
            τη = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(μη, τη); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + b * z + η, σ)
        end
    end
end

function _lfpc_model_re_exponential()
    return @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
            θ = RealNumber(0.8, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Exponential(θ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + b * z + η, σ)
        end
    end
end

function _lfpc_model_mixed_outcomes()
    return @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
            τ = RealNumber(0.6, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y1 ~ Normal(a + b * z + η, σ)
            λ = exp(a + b * z + η^2)
            y2 ~ Poisson(λ)
        end
    end
end

function _lfpc_df_normal()
    return DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.2, -0.1, 0.1, 0.0, -0.3, 0.4],
        y = [0.3, 0.0, 0.1, -0.1, -0.2, 0.2]
    )
end

function _lfpc_df_lognormal()
    return DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.2, -0.1, 0.1, 0.0, -0.3, 0.4],
        y = [1.2, 1.0, 1.1, 0.9, 0.8, 1.3]
    )
end

function _lfpc_df_bernoulli()
    return DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.2, -0.1, 0.1, 0.0, -0.3, 0.4],
        y = [1, 0, 1, 0, 0, 1]
    )
end

function _lfpc_df_poisson()
    return DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.2, -0.1, 0.1, 0.0, -0.3, 0.4],
        y = [1, 2, 0, 1, 3, 1]
    )
end

function _lfpc_df_ode()
    return DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [1.1, 0.7, 0.9, 0.5, 1.0, 0.6]
    )
end

function _lfpc_df_mixed()
    return DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.2, -0.1, 0.1, 0.0, -0.3, 0.4],
        y1 = [0.3, 0.0, 0.1, -0.1, -0.2, 0.2],
        y2 = [1, 2, 0, 1, 3, 1]
    )
end

# Keep compatibility with Step 1 test naming.
_lfpc_df() = _lfpc_df_normal()

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE: test sets
# ══════════════════════════════════════════════════════════════════════════════

@testset "Laplace fastpath baseline fixtures are deterministic" begin
    for kind in (:gaussian, :lognormal, :bernoulli, :poisson, :ode_offset, :ode_eta)
        df1 = _lfp_make_df(kind; seed=123)
        df2 = _lfp_make_df(kind; seed=123)
        @test names(df1) == names(df2)
        for col in names(df1)
            @test all(df1[!, col] .== df2[!, col])
        end
    end
end

@testset "Laplace objective+gradient baseline reproducibility" begin
    for kind in (:gaussian, :lognormal, :bernoulli, :poisson)
        model = _lfp_model(kind)
        df = _lfp_make_df(kind; n_id=4, seed=321)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        obj1, g1 = _lfp_eval_objgrad(dm; seed=999)
        obj2, g2 = _lfp_eval_objgrad(dm; seed=999)
        @test isapprox(obj1, obj2; atol=1e-10, rtol=1e-10)
        @test length(g1) == length(g2)
        @test isapprox(g1, g2; atol=1e-8, rtol=1e-8)
    end
end

@testset "ODE fixture baseline (offset only vs RE in ODE) both run" begin
    for kind in (:ode_offset, :ode_eta)
        model = _lfp_model(kind)
        model = set_solver_config(model; saveat_mode=:saveat)
        df = _lfp_make_df(kind; n_id=4, seed=222)
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=2, multistart_k=2))
        @test res.summary.converged isa Bool
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG: test sets
# ══════════════════════════════════════════════════════════════════════════════

@testset "Laplace fastpath mode options" begin
    @test NoLimits.Laplace().fastpath.mode == :auto
    @test NoLimits.LaplaceMAP().fastpath.mode == :auto
    @test NoLimits.Laplace(; fastpath_mode=:off).fastpath.mode == :off
    @test NoLimits.LaplaceMAP(; fastpath_mode=:off).fastpath.mode == :off
    @test_throws ErrorException NoLimits.Laplace(; fastpath_mode=:invalid_mode)
    @test_throws ErrorException NoLimits.LaplaceMAP(; fastpath_mode=:invalid_mode)
end

@testset "Laplace fastpath start info is per outcome" begin
    dm = DataModel(_lfpc_model_normal(), _lfpc_df_normal(); primary_id=:ID, time_col=:t)
    info_auto = NoLimits._laplace_fastpath_start_info(dm, :auto)
    @test info_auto.mode == :auto
    @test info_auto.active == true
    @test info_auto.backend == :newton_inner
    @test info_auto.summary_reason == :newton_inner_backend_enabled
    @test length(info_auto.outcomes) == 1
    @test info_auto.outcomes[1].outcome == :y
    @test info_auto.outcomes[1].eligible == true
    @test info_auto.outcomes[1].reasons == (:eligible_normal_linear_re,)

    info_off = NoLimits._laplace_fastpath_start_info(dm, :off)
    @test info_off.mode == :off
    @test info_off.active == false
    @test info_off.backend == :none
    @test info_off.summary_reason == :mode_off
    @test length(info_off.outcomes) == 1
    @test info_off.outcomes[1].outcome == :y
    @test info_off.outcomes[1].eligible == false
    @test info_off.outcomes[1].reasons == (:mode_off,)

    @test_throws ErrorException NoLimits._laplace_fastpath_start_info(dm, :invalid_mode)
end

@testset "Laplace fastpath auto eligibility covers all target families" begin
    cases = (
        (name=:normal, model=_lfpc_model_normal(), df=_lfpc_df_normal(), expected=true, reason=:eligible_normal_linear_re, active=true, summary=:newton_inner_backend_enabled),
        (name=:lognormal, model=_lfpc_model_lognormal(), df=_lfpc_df_lognormal(), expected=true, reason=:eligible_lognormal_linear_re, active=true, summary=:newton_inner_backend_enabled),
        (name=:bernoulli, model=_lfpc_model_bernoulli(), df=_lfpc_df_bernoulli(), expected=true, reason=:eligible_bernoulli_logistic_linear_re, active=true, summary=:newton_inner_backend_enabled),
        (name=:poisson, model=_lfpc_model_poisson(), df=_lfpc_df_poisson(), expected=true, reason=:eligible_poisson_log_linear_re, active=true, summary=:newton_inner_backend_enabled),
        (name=:ode_offset, model=set_solver_config(_lfpc_model_ode_offset(); saveat_mode=:saveat), df=_lfpc_df_ode(), expected=true, reason=:eligible_normal_linear_re, active=true, summary=:newton_inner_backend_enabled_with_ode_polish),
        (name=:ode_eta, model=set_solver_config(_lfpc_model_ode_eta(); saveat_mode=:saveat), df=_lfpc_df_ode(), expected=false, reason=:random_effect_in_de_dynamics, active=false, summary=:no_eligible_outcomes),
        (name=:re_lognormal, model=_lfpc_model_re_lognormal(), df=_lfpc_df_normal(), expected=true, reason=:eligible_normal_linear_re, active=true, summary=:newton_inner_backend_enabled),
        (name=:re_exponential, model=_lfpc_model_re_exponential(), df=_lfpc_df_normal(), expected=false, reason=:unsupported_random_effect_distribution, active=false, summary=:no_eligible_outcomes),
    )

    for c in cases
        dm = DataModel(c.model, c.df; primary_id=:ID, time_col=:t)
        info = NoLimits._laplace_fastpath_start_info(dm, :auto)
        @test length(info.outcomes) == 1
        out = info.outcomes[1]
        @test out.outcome == :y
        @test out.eligible == c.expected
        @test c.reason in out.reasons
        @test info.summary_reason == c.summary
        @test info.active == c.active
        if c.active
            @test info.backend == :newton_inner
        else
            @test info.backend == :none
        end
    end
end

@testset "Laplace fastpath enables dense backend for multi-RE eligible model" begin
    dm = DataModel(_lfpc_model_normal_two_re(), _lfpc_df_normal(); primary_id=:ID, time_col=:t)
    info = NoLimits._laplace_fastpath_start_info(dm, :auto)
    @test info.mode == :auto
    @test info.active == true
    @test info.backend == :newton_inner
    @test info.summary_reason == :newton_inner_backend_enabled
    @test length(info.outcomes) == 1
    @test info.outcomes[1].eligible == true
    @test info.outcomes[1].reasons == (:eligible_normal_linear_re,)
end

@testset "Laplace fastpath eligibility is per outcome in mixed models" begin
    dm = DataModel(_lfpc_model_mixed_outcomes(), _lfpc_df_mixed(); primary_id=:ID, time_col=:t)
    info = NoLimits._laplace_fastpath_start_info(dm, :auto)
    @test length(info.outcomes) == 2

    by_outcome = Dict(o.outcome => o for o in info.outcomes)
    @test haskey(by_outcome, :y1)
    @test haskey(by_outcome, :y2)

    @test by_outcome[:y1].eligible
    @test :eligible_normal_linear_re in by_outcome[:y1].reasons
    @test !by_outcome[:y2].eligible
    @test :poisson_requires_exp_affine in by_outcome[:y2].reasons
    @test info.summary_reason == :partial_eligibility_backend_requires_all_outcomes
    @test !info.active
    @test info.backend == :none

    info_off = NoLimits._laplace_fastpath_start_info(dm, :off)
    @test info_off.summary_reason == :mode_off
    @test all(!o.eligible for o in info_off.outcomes)
    @test all(o.reasons == (:mode_off,) for o in info_off.outcomes)
end

@testset "Step4 no numerical change: fastpath auto equals off" begin
    dm_l = DataModel(_lfpc_model(), _lfpc_df(); primary_id=:ID, time_col=:t)
    dm_m = DataModel(_lfpc_model(; with_priors=true), _lfpc_df(); primary_id=:ID, time_col=:t)
    dm_l2 = DataModel(_lfpc_model_normal_two_re(), _lfpc_df(); primary_id=:ID, time_col=:t)
    dm_ode = DataModel(set_solver_config(_lfpc_model_ode_offset(); saveat_mode=:saveat), _lfpc_df_ode(); primary_id=:ID, time_col=:t)

    lap_auto = NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
    lap_off = NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:off)
    map_auto = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
    map_off = NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:off)
    lap_auto_dense = NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
    lap_off_dense = NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:off)
    lap_auto_ode = NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:auto)
    lap_off_ode = NoLimits.Laplace(; optim_kwargs=(maxiters=2,), inner_kwargs=(maxiters=2,), multistart_n=0, multistart_k=0, fastpath_mode=:off)

    res_l_auto = fit_model(dm_l, lap_auto; rng=MersenneTwister(22))
    res_l_off = fit_model(dm_l, lap_off; rng=MersenneTwister(22))
    @test isapprox(get_objective(res_l_auto), get_objective(res_l_off); atol=1e-10, rtol=1e-10)

    res_m_auto = fit_model(dm_m, map_auto; rng=MersenneTwister(22))
    res_m_off = fit_model(dm_m, map_off; rng=MersenneTwister(22))
    @test isapprox(get_objective(res_m_auto), get_objective(res_m_off); atol=1e-10, rtol=1e-10)

    res_l2_auto = fit_model(dm_l2, lap_auto_dense; rng=MersenneTwister(33))
    res_l2_off = fit_model(dm_l2, lap_off_dense; rng=MersenneTwister(33))
    @test isapprox(get_objective(res_l2_auto), get_objective(res_l2_off); atol=1e-8, rtol=1e-8)

    res_ode_auto = fit_model(dm_ode, lap_auto_ode; rng=MersenneTwister(44))
    res_ode_off = fit_model(dm_ode, lap_off_ode; rng=MersenneTwister(44))
    @test isapprox(get_objective(res_ode_auto), get_objective(res_ode_off); atol=1e-8, rtol=1e-8)
end
