using Test
using DataFrames
using DataInterpolations
using NoLimits
using JLD2
using OptimizationNLopt
using FiniteDifferences
using LineSearches
using OptimizationBBO
using Optimization
using OptimizationOptimJL
import Optimisers
using Distributions
using ComponentArrays
using LinearAlgebra
using Random
using SciMLBase
using OrdinaryDiffEq

# Second half of the consolidated Laplace tests (split from
# estimation_laplace_tests.jl for CI shard balance; the two files run in
# different shards and are self-contained).

# Fresh Laplace EBE cache scaffold (shared by the Hutchinson tests).
function _make_laplace_ebe_cache(T::Type, n_batches::Int)
    bstar_cache = NoLimits._LaplaceBStarCache(
        [Vector{T}() for _ in 1:n_batches], falses(n_batches)
    )
    grad_cache = NoLimits._LaplaceGradCache(
        [Vector{T}() for _ in 1:n_batches],
        fill(T(NaN), n_batches),
        [Vector{T}() for _ in 1:n_batches],
        falses(n_batches)
    )
    ad_cache = NoLimits._init_laplace_ad_cache(n_batches)
    hess_cache = NoLimits._init_laplace_hess_cache(T, n_batches)
    return NoLimits._LaplaceCache(
        nothing, bstar_cache, grad_cache, ad_cache, hess_cache
    )
end
@testset "Laplace serial == threaded with dynamic covariates (#115)" begin
    # Dynamic-covariate interpolants and a DE signal read per-individual state that the
    # threaded path must not share across tasks.
    if Threads.nthreads() < 2
        @info "skipped: needs Threads.nthreads() > 1"
        @test true
        return
    end
    model = @Model begin
        @fixedEffects begin
            k = RealNumber(0.5; scale = :log)
            α = RealNumber(0.2)
            ω = RealNumber(0.4; scale = :log)
            σ = RealNumber(0.3; scale = :log)
            σ2 = RealNumber(0.1; scale = :log)
        end
        @covariates begin
            t = Covariate()
            w = DynamicCovariate(; interpolation = LinearInterpolation)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @DifferentialEquation begin
            s(t) = x1 / (1 + x1)
            D(x1) ~ -k * x1 + α * w(t)
        end
        @initialDE begin
            x1 = exp(η)
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
            y2 ~ Normal(s(t), σ2)
        end
    end
    rng = MersenneTwister(4)
    rows = NamedTuple[]
    for i in 1:8
        η = 0.3 * randn(rng)
        for tt in 0.0:0.5:2.0
            x = exp(η - 0.4 * tt)
            obs = tt > 0.0
            push!(
                rows,
                (;
                    ID = "id_$i", t = tt, w = 0.5 + rand(rng),
                    y = obs ? x + 0.2 * randn(rng) : missing,
                    y2 = obs ? x / (1 + x) + 0.05 * randn(rng) : missing,
                )
            )
        end
    end
    dm = DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
    method = NoLimits.Laplace(; optim_kwargs = (maxiters = 3,))
    obj(ser) = NoLimits.get_objective(
        fit_model(dm, method; serialization = ser, rng = MersenneTwister(99))
    )
    o_serial = obj(EnsembleSerial())
    @test isapprox(obj(EnsembleThreads()), o_serial; rtol = 1.0e-10)
end

@testset "Laplace fit with BlackBoxOptim requires bounds" begin
    # Bespoke: needs free params with no finite bounds so BBO errors without lb/ub.
    bbo_model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    dm = DataModel(
        bbo_model,
        DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]);
        primary_id = :ID, time_col = :t
    )
    @test_throws ErrorException fit_model(
        dm,
        NoLimits.Laplace(;
            optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), optim_kwargs = (maxiters = 2,)
        )
    )
    lb, ub = default_bounds_from_start(dm; margin = 1.0)
    res = fit_model(
        dm,
        NoLimits.Laplace(;
            optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(), optim_kwargs = (maxiters = 2,), lb = lb, ub = ub
        )
    )
    @test res.summary.converged isa Bool
end

@testset "Laplace multistart options" begin
    lap = NoLimits.Laplace()
    @test lap.multistart.n == 50 && lap.multistart.k == 10 &&
        lap.multistart.sampling == :lhs
    @test fit_model(
        fx_re_dm(),
        NoLimits.Laplace(;
            optim_kwargs = (maxiters = 2,),
            multistart_n = 2, multistart_k = 2, multistart_grad_tol = 0.0
        ),
        rng = MersenneTwister(1)
    ) isa FitResult
end

@testset "Laplace objective cache only reuses valid gradients" begin
    θ = ComponentArray((a = 0.1, σ = 0.2))
    axs = getaxes(θ)
    cache = NoLimits._LaplaceObjCache{Float64, ComponentArray}(
        nothing, Inf,
        ComponentArray(zeros(Float64, length(θ)), axs), false
    )
    NoLimits._laplace_obj_cache_set_obj!(cache, θ, 1.0)
    @test NoLimits._laplace_obj_cache_lookup(cache, θ, 1.0e9) === nothing
    grad = ComponentArray([3.0, 4.0], axs)
    NoLimits._laplace_obj_cache_set_obj_grad!(cache, θ, 2.0, grad)
    hit = NoLimits._laplace_obj_cache_lookup(cache, θ, 0.0)
    @test hit !== nothing && hit[1] == 2.0 && collect(hit[2]) == collect(grad)
end

@testset "Laplace threaded cache fallback preserves ODE options" begin
    dm = fx_ode_dm()
    ll_cache = build_ll_cache(dm; ode_kwargs = (abstol = 1.0e-8, reltol = 1.0e-7))
    threaded = NoLimits._laplace_thread_caches(dm, ll_cache, 2)
    @test length(threaded) == 2
    @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
    @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
end

@testset "reestimate_ebes" begin
    dm = fx_re_dm()
    n = length(get_individuals(dm))
    res_new = reestimate_ebes(fx_laplace())
    re = get_random_effects(res_new)
    @test re isa NamedTuple && haskey(re, :η) && nrow(re.η) == n
    res_nostore = fit_model(
        dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)); store_data_model = false
    )
    @test nrow(get_random_effects(dm, reestimate_ebes(dm, res_nostore)).η) == n
    @test nrow(get_random_effects(reestimate_ebes(fx_laplace(); individuals = [1, 2])).η) ==
        n
    path = tempname() * ".jld2"
    save_fit(path, fx_laplace())
    @test nrow(get_random_effects(dm, reestimate_ebes(dm, load_fit(path; dm = dm))).η) == n
    re_saem = get_random_effects(reestimate_ebes(fx_saem()))
    @test re_saem isa NamedTuple && haskey(re_saem, :η) && nrow(re_saem.η) == n
end

@testset "reestimate_ebes mcmc sampling" begin
    res_new = reestimate_ebes(
        fx_laplace(); ebe_multistart_sampling = :mcmc,
        ebe_multistart_n = 5, ebe_mcmc_n_adapt = 2
    )
    re = get_random_effects(res_new)
    @test re isa NamedTuple && haskey(re, :η) &&
        nrow(re.η) == length(get_individuals(fx_re_dm()))
end

@testset "Laplace with NormalizingPlanarFlow custom base_dist" begin
    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.3, 0.25]
    )
    function make_npf_model(base)
        @Model begin
            @fixedEffects begin
                a = RealNumber(0.1)
                σ = RealNumber(0.3, scale = :log)
                ψ = NPFParameter(1, 2; seed = 1, calculate_se = false, base_dist = base)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η[1], σ)
            end
        end
    end
    res_default = fit_model(
        DataModel(make_npf_model(nothing), df; primary_id = :ID, time_col = :t),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,))
    )
    @test res_default isa FitResult
    res_mvn = fit_model(
        DataModel(
            make_npf_model(MvNormal([0.5], [2.0;;])), df; primary_id = :ID, time_col = :t
        ),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,))
    )
    @test res_mvn isa FitResult
    res_tdist = fit_model(
        DataModel(
            make_npf_model(MvTDist(5, zeros(1), ones(1, 1))),
            df; primary_id = :ID, time_col = :t
        ),
        NoLimits.Laplace(; optim_kwargs = (maxiters = 2,))
    )
    @test res_tdist isa FitResult
    @test NoLimits.get_objective(res_default) != NoLimits.get_objective(res_tdist)
end

@testset "Laplace penalty enters objective AND gradient" begin
    dm = fx_re_dm()
    res_unpen = fit_model(dm, NoLimits.Laplace())
    res_pen = fit_model(dm, NoLimits.Laplace(); penalty = (a = 1.0e6,))
    a_unpen = NoLimits.get_params(res_unpen; scale = :untransformed).a
    a_pen = NoLimits.get_params(res_pen; scale = :untransformed).a
    # The ridge penalty w·a² with a huge weight must pull â to ≈ 0. With the
    # historical bug the reported gradient lacked the penalty term, so the
    # optimizer stalled at (or walked toward) the unpenalized optimum.
    @test abs(a_pen) < 0.05
    @test abs(a_pen) < abs(a_unpen)
end

# ── Hutchinson logdet-gradient option (folded from estimation_hutchinson) ────

@testset "Hutchinson logdet gradient approximates trace" begin
    Random.seed!(1234)
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    pairing, batch_infos, const_cache = NoLimits._build_re_batch_infos(
        dm, NamedTuple()
    )
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(model.fixed.fixed)
    ebe_cache = _make_laplace_ebe_cache(eltype(θu), length(batch_infos))

    info = batch_infos[1]
    b = NoLimits._laplace_default_b0(dm, info, θu, const_cache, ll_cache)

    res_exact = NoLimits._laplace_grad_batch(
        dm, info, θu, b, const_cache, ll_cache, ebe_cache.ad_cache, 1;
        use_trace_logdet_grad = true,
        use_hutchinson = false
    )
    res_hutch = NoLimits._laplace_grad_batch(
        dm, info, θu, b, const_cache, ll_cache, ebe_cache.ad_cache, 1;
        use_trace_logdet_grad = true,
        use_hutchinson = true,
        hutchinson_n = 16
    )

    denom = max(norm(res_exact.grad), eps())
    rel_err = norm(res_hutch.grad - res_exact.grad) / denom
    @test rel_err < 0.6
end

@testset "Hutchinson gradients are driven by passed rng" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η1 = RandomEffect(Normal(0.0, 1.0); column = :ID)
            η2 = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η1 + η2, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.3, 0.2, 0.05, -0.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(model.fixed.fixed)
    method = NoLimits.Laplace(; use_hutchinson = true, hutchinson_n = 1)

    function eval_grad(global_seed::Int, rng_seed::Int)
        Random.seed!(global_seed)
        ebe_cache = _make_laplace_ebe_cache(eltype(θu), length(batch_infos))
        _, g, _ = NoLimits._laplace_objective_and_grad(
            dm, batch_infos, θu, const_cache, ll_cache, ebe_cache;
            inner = method.inner,
            hessian = method.hessian,
            cache_opts = method.cache,
            multistart = method.multistart,
            rng = MersenneTwister(rng_seed)
        )
        return collect(g)
    end

    g1 = eval_grad(1, 123)
    g2 = eval_grad(2, 123)
    g3 = eval_grad(1, 999)

    @test isapprox(g1, g2; atol = 1.0e-12, rtol = 1.0e-12)
    @test maximum(abs.(g1 .- g3)) > 1.0e-6
end

# ── NewtonInner inner optimizer (folded from estimation_newton_inner) ────────
# NewtonInner is an OPT-IN inner-EBE optimizer: the default (LBFGS via
# Optimization.jl) is unchanged. These tests check that (1) the option produces
# the same fits as the default within inner-solver tolerance, (2) the
# `max_dim` fallback reproduces the default path exactly, and (3) the inner
# solutions themselves agree at matching gradient tolerances.

function _newton_test_dm()
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.8)
            b = RealNumber(0.3)
            ω = RealNumber(0.5, scale = :log)
            σ = RealNumber(0.4, scale = :log)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            lin = a + b * x.Age + η
            y ~ Normal(lin, σ)
        end
    end
    rng = Xoshiro(11)
    rows = NamedTuple[]
    for i in 1:40
        age = 0.5 + 0.05 * (i % 20)
        ηi = 0.4 * randn(rng)
        for j in 1:6
            t = (j - 1) * 0.5
            y = 0.8 + 0.3 * age + ηi + 0.4 * randn(rng)
            push!(rows, (ID = i, t = t, Age = age, y = y))
        end
    end
    return DataModel(model, DataFrame(rows); primary_id = :ID, time_col = :t)
end

@testset "NewtonInner inner solver (opt-in)" begin
    dm = _newton_test_dm()

    @testset "inner solve agrees with the default optimizer" begin
        llc = NoLimits.build_ll_cache(dm; force_saveat = true)
        _, binfos, ccache = NoLimits._build_re_batch_infos(dm, NamedTuple())
        θ = NoLimits.get_θ0_untransformed(get_model(dm).fixed.fixed)
        adc = NoLimits._init_laplace_ad_cache(length(binfos))
        for bi in (1, 5, 17)
            info = binfos[bi]
            b0 = zeros(info.n_b)
            sol_def = NoLimits._laplace_solve_batch!(dm, info, θ, ccache, llc, adc, bi, b0)
            sol_new = NoLimits._laplace_solve_batch!(
                dm, info, θ, ccache, llc, adc, bi, b0;
                optimizer = NewtonInner()
            )
            @test sol_new isa NoLimits._NewtonSol
            @test sol_new.converged
            @test NoLimits._laplace_sol_grad_norm(sol_new) <= 1.0e-8
            @test collect(sol_new.u) ≈ collect(sol_def.u) atol = 1.0e-6
            @test NoLimits._laplace_sol_logf(sol_new) ≈ NoLimits._laplace_sol_logf(sol_def) atol = 1.0e-8
        end
    end

    @testset "Laplace fit matches default within tolerance" begin
        res_def = fit_model(
            dm, NoLimits.Laplace(optim_kwargs = (maxiters = 40,));
            serialization = EnsembleSerial(), rng = Xoshiro(3)
        )
        res_new = fit_model(
            dm,
            NoLimits.Laplace(
                optim_kwargs = (maxiters = 40,),
                inner_optimizer = NewtonInner()
            );
            serialization = EnsembleSerial(), rng = Xoshiro(3)
        )
        @test isfinite(get_objective(res_new))
        @test get_objective(res_new) ≈ get_objective(res_def) rtol = 1.0e-6 atol = 1.0e-6
        # Qualified: MCMCChains (loaded by earlier files in the same batch
        # subprocess) also exports a `get_params`, making the bare name ambiguous.
        @test collect(
            NoLimits.get_params(
                res_new;
                scale = :transformed
            )
        ) ≈
            collect(NoLimits.get_params(res_def; scale = :transformed)) rtol = 1.0e-3 atol = 1.0e-3
        eta_def = get_random_effects(dm, res_def, :η)
        eta_new = get_random_effects(dm, res_new, :η)
        @test eta_new ≈ eta_def atol = 1.0e-4
    end

    @testset "FOCEI fit matches default within tolerance" begin
        res_def = fit_model(
            dm, NoLimits.FOCEI(optim_kwargs = (maxiters = 40,));
            serialization = EnsembleSerial(), rng = Xoshiro(3)
        )
        res_new = fit_model(
            dm,
            NoLimits.FOCEI(
                optim_kwargs = (maxiters = 40,),
                inner_optimizer = NewtonInner()
            );
            serialization = EnsembleSerial(), rng = Xoshiro(3)
        )
        @test isfinite(get_objective(res_new))
        @test get_objective(res_new) ≈ get_objective(res_def) rtol = 1.0e-6 atol = 1.0e-6
    end

    @testset "max_dim fallback reproduces the default path" begin
        res_def = fit_model(
            dm, NoLimits.Laplace(optim_kwargs = (maxiters = 15,));
            serialization = EnsembleSerial(), rng = Xoshiro(3)
        )
        res_fb = fit_model(
            dm,
            NoLimits.Laplace(
                optim_kwargs = (maxiters = 15,),
                inner_optimizer = NewtonInner(max_dim = 0)
            );
            serialization = EnsembleSerial(), rng = Xoshiro(3)
        )
        @test get_objective(res_fb) ≈ get_objective(res_def) rtol = 1.0e-10 atol = 1.0e-10
    end
end

# The outer derivative-free optimizers size each coordinate's first step by that
# coordinate's own value, so a correlated Ω starting at [1 ε; ε 1] leaves the second
# log-Cholesky diagonal (≈ -ε²/2) effectively frozen. Preconditioning gives every
# coordinate a first step of at least 1, so the covariance block can actually move.
@testset "preconditioning frees a near-zero transformed coordinate" begin
    model = @Model begin
        @fixedEffects begin
            tcl = RealNumber(log(0.05))
            tv = RealNumber(log(1.0))
            Ω = RealPSDMatrix([1.0 0.01; 0.01 1.0], scale = :cholesky)
            σ = RealNumber(0.2, scale = :log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end

        @formulas begin
            cp = exp(tcl + η[1]) / exp(tv + η[2])
            y ~ Normal(cp, σ)
        end
    end

    rng = Xoshiro(7)
    ids = repeat(1:12; inner = 4)
    df = DataFrame(
        ID = ids, t = repeat([0.5, 1.0, 2.0, 4.0], 12),
        y = 0.05 .* exp.(0.4 .* randn(rng, 48))
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    # BOBYQA is pinned, not defaulted: this test exercises a DERIVATIVE-FREE pathology - such an
    # optimizer sizes each coordinate's first step by that coordinate's own value, so a
    # coordinate starting near zero never moves, and preconditioning is what frees it. The
    # default optimizer is gradient-based and does not size steps that way, so leaving the
    # optimizer implicit would silently stop testing the thing this testset is named after.
    bob = NLopt.LN_BOBYQA()
    on = fit_model(dm, NoLimits.Laplace(; optimizer = bob); rng = Xoshiro(1))
    off = fit_model(
        dm, NoLimits.Laplace(; optimizer = bob, precondition = false); rng = Xoshiro(1)
    )
    Ω_on = NoLimits.get_params(on; scale = :untransformed).Ω
    Ω_off = NoLimits.get_params(off; scale = :untransformed).Ω

    # Preconditioning must never do worse, and it must move Ω[2,2] off its start.
    @test NoLimits.get_objective(on) <= NoLimits.get_objective(off) + 1.0e-8
    @test abs(Ω_on[2, 2] - 1.0) > abs(Ω_off[2, 2] - 1.0)

    # The gradient-based default does not size steps by |x0|, so preconditioning neither frees
    # nor freezes a coordinate there - it must simply not disturb the fit. Asserted two-sided on
    # purpose: which of two optimizer parameterizations lands infinitesimally lower is arbitrary
    # (they agree to ~1e-8 relative here), so a one-sided inequality would fail on a coin flip.
    on_d = fit_model(dm, NoLimits.Laplace(); rng = Xoshiro(1))
    off_d = fit_model(dm, NoLimits.Laplace(; precondition = false); rng = Xoshiro(1))
    @test isfinite(NoLimits.get_objective(on_d))
    @test NoLimits.get_objective(on_d) ≈ NoLimits.get_objective(off_d) rtol = 1.0e-4
end

# Preconditioning rescales the outer variable, so the gradient handed to the optimizer
# must carry the per-coordinate factor: G_z[i] = s[i] * G_θ[i]. Both fits below start at
# the same θ0 (z = 0 maps to θ0), so the two gradients are taken at the identical point
# and a factor missing on any single coordinate shows up as a ratio != s[i].
@testset "preconditioned gradient carries the per-coordinate factor" begin
    model = @Model begin
        @fixedEffects begin
            tcl = RealNumber(log(0.05))
            tv = RealNumber(log(1.0))
            Ω = RealPSDMatrix([1.0 0.01; 0.01 1.0], scale = :cholesky)
            σ = RealNumber(0.2, scale = :log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(zeros(2), Ω); column = :ID)
        end

        @formulas begin
            cp = exp(tcl + η[1]) / exp(tv + η[2])
            y ~ Normal(cp, σ)
        end
    end

    rng = Xoshiro(11)
    df = DataFrame(
        ID = repeat(1:15; inner = 4), t = repeat([0.5, 1.0, 2.0, 4.0], 15),
        y = 0.05 .* exp.(0.3 .* randn(rng, 60))
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    lay_pc = NoLimits.free_parameter_layout(NoLimits.get_fixed(NoLimits.get_model(dm)))
    s = NoLimits._precondition_scale(
        NoLimits.get_model(dm), lay_pc.free_names, lay_pc.θ0_free_t
    )

    first_gradient = pc -> begin
        res = fit_model(
            dm,
            NoLimits.Laplace(;
                precondition = pc,
                optimizer = OptimizationOptimJL.LBFGS(
                    linesearch = LineSearches.BackTracking()
                ),
                optim_kwargs = (;
                    maxiters = 1, store_trace = true,
                    extended_trace = true,
                )
            );
            rng = Xoshiro(1)
        )
        collect(NoLimits.get_raw(res).trace[1].metadata["g(x)"])
    end

    g_off = first_gradient(false)
    g_on = first_gradient(true)
    @test length(g_on) == length(s)
    @test all(isapprox(g_on[i] / g_off[i], s[i]; rtol = 1.0e-6) for i in eachindex(g_off))
end

@testset "AD-incompatible and bounded-support RE distributions (#247)" begin
    # NoncentralT/F/Chisq/Beta and StudentizedRange reach Rmath through
    # StatsFuns.RFunctions, which converts its argument to Float64. That is an upstream
    # ForwardDiff incompatibility, so the marginal estimators reject them up front
    # instead of failing with a MethodError deep in the AD path.
    m_nct = @Model begin
        @fixedEffects begin
            a = RealNumber(0.4)
            σ = RealNumber(0.35; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Distributions.NoncentralT(5.0, 0.5); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(
        ID = repeat(1:4, inner = 3), t = repeat(0.0:2.0, 4), y = collect(1.0:12.0)
    )
    dm_nct = DataModel(m_nct, df; primary_id = :ID, time_col = :t)
    for method in (NoLimits.Laplace(), NoLimits.FOCEI(), NoLimits.GHQuadrature())
        @test_throws ArgumentError fit_model(dm_nct, method)
    end
    # The parse-time distribution symbol must survive module qualification, otherwise
    # every `Distributions.X(...)` random effect is seen as `:unknown`.
    @test NoLimits.get_re_types(NoLimits.get_random(m_nct)).η === :NoncentralT

    # A bounded-support RE prior can put every empirical Bayes mode outside its support,
    # so the Laplace objective is the finite infeasibility wall at every θ. That is not a
    # converged fit, whatever the optimizer's return code says.
    m_unif = @Model begin
        @fixedEffects begin
            a = RealNumber(0.4)
            σ = RealNumber(0.35; scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Uniform(-1.0, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df_u = DataFrame(
        ID = repeat(1:4, inner = 3), t = repeat(0.0:2.0, 4),
        y = repeat([10.0, -10.0, 10.0, -10.0], inner = 3)
    )
    dm_u = DataModel(m_unif, df_u; primary_id = :ID, time_col = :t)
    res_u = fit_model(dm_u, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))
    @test NoLimits.get_objective(res_u) >= 1.0e10
    @test NoLimits.get_converged(res_u) === false
    @test res_u.summary.notes isa AbstractString
end

# ── Mini-batching over RE batches (#281) ─────────────────────────────────────

# Records the (nbatches, iter, rng) the schedule is called with, so the tests can
# assert it advances exactly once per optimizer iteration.
struct _MBRec
    nb::Vector{Int}
    it::Vector{Int}
    rngs::Vector{Any}
end
function (r::_MBRec)(nbatches::Int, iter::Int, rng)
    push!(r.nb, nbatches)
    push!(r.it, iter)
    push!(r.rngs, rng)
    return [1, 2]
end

struct _MBFixed
    sel::Vector{Int}
end
(f::_MBFixed)(nbatches::Int, iter::Int, rng) = f.sel

struct _MBAlternate end
(::_MBAlternate)(nbatches::Int, iter::Int, rng) = [mod1(iter, nbatches)]

@testset "Laplace/FOCEI/GHQ mini-batching constructors (#281)" begin
    for ctor in (NoLimits.Laplace, NoLimits.FOCEI, NoLimits.GHQuadrature)
        @test ctor().update_schedule === :all
        @test ctor(update_schedule = :all).update_schedule === :all
        @test ctor(update_schedule = 2).update_schedule == 2
        @test ctor(update_schedule = _MBFixed([1])).update_schedule isa _MBFixed
        @test_throws ErrorException ctor(update_schedule = 0)
        @test_throws ErrorException ctor(update_schedule = "bogus")
        @test_throws ErrorException ctor(update_schedule = 1.5)

        @test ctor().optimizer isa OptimizationOptimJL.LBFGS
        @test ctor(update_schedule = 2).optimizer isa Optimisers.AbstractRule

        @test_logs (:warn, r"stochastic") ctor(
            update_schedule = 2, optimizer = OptimizationOptimJL.LBFGS()
        )
        adam = Optimisers.Adam(0.05)
        @test_logs ctor(update_schedule = 2, optimizer = adam)
        @test ctor(update_schedule = 2, optimizer = adam).optimizer === adam
        @test_logs ctor(update_schedule = :all, optimizer = OptimizationOptimJL.LBFGS())
    end
end

@testset "Minibatch state machinery (#281)" begin
    @test NoLimits._minibatch_state(:all, 6, MersenneTwister(1)) === nothing
    @test NoLimits._minibatch_current!(nothing) === nothing
    @test NoLimits._minibatch_active(nothing) === nothing

    st = NoLimits._minibatch_state(2, 6, MersenneTwister(1))
    NoLimits._minibatch_current!(st)
    @test length(st.selected) == 2
    @test allunique(st.selected)
    @test issorted(st.selected)
    @test all(i -> 1 <= i <= 6, st.selected)
    @test st.scale == 3.0
    @test NoLimits._minibatch_active(st) == Set(st.selected)

    # A minibatch larger than the batch count degenerates to all batches, scale 1.
    st_all = NoLimits._minibatch_state(99, 6, MersenneTwister(1))
    NoLimits._minibatch_current!(st_all)
    @test st_all.selected == collect(1:6)
    @test st_all.scale == 1.0
end

@testset "Projected Optimisers rule respects the box (#281)" begin
    x = [0.0, 0.0]
    rule = NoLimits._ProjectedRule(Optimisers.Descent(1.0), [-1.0, -1.0], [0.5, 0.5])
    opt_state = Optimisers.setup(rule, x)
    opt_state, x = Optimisers.update!(opt_state, x, [-10.0, 10.0])
    @test x == [0.5, -1.0]

    # `:all` + a non-Optimisers optimizer is passed through untouched.
    lbfgs = OptimizationOptimJL.LBFGS()
    o, ub_flag = NoLimits._bounded_optimizer(lbfgs, true, [-1.0], [1.0])
    @test o === lbfgs
    @test ub_flag === true
end

@testset "Laplace mini-batching (#281)" begin
    dm = fx_re_dm()
    ser = SciMLBase.EnsembleSerial()

    # ── 1. `:all` is bit-identical to the pre-#281 default path ──────────────
    r_def = fit_model(
        dm, NoLimits.Laplace(optim_kwargs = (; maxiters = 5)); serialization = ser
    )
    r_all = fit_model(
        dm, NoLimits.Laplace(update_schedule = :all, optim_kwargs = (; maxiters = 5));
        serialization = ser
    )
    @test NoLimits.get_objective(r_def) == NoLimits.get_objective(r_all)
    @test NoLimits.get_params(r_def; scale = :transformed) ==
        NoLimits.get_params(r_all; scale = :transformed)

    # ── 2. The schedule advances once per OPTIMIZER iteration ────────────────
    rec = _MBRec(Int[], Int[], Any[])
    rng = MersenneTwister(1)
    res_rec = fit_model(
        dm,
        NoLimits.Laplace(
            update_schedule = rec, optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 4)
        );
        rng = rng, serialization = ser
    )
    @test rec.it == collect(1:4)
    @test all(==(6), rec.nb)
    @test all(x -> x === rng, rec.rngs)
    @test isfinite(NoLimits.get_objective(res_rec))

    # ── 3. Integer schedule runs and reports a finite full-data objective ────
    res_int = fit_model(
        dm,
        NoLimits.Laplace(
            update_schedule = 2, optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 3)
        );
        rng = MersenneTwister(7), serialization = ser
    )
    @test isfinite(NoLimits.get_objective(res_int))

    # ── 4. Reproducible given the same rng ───────────────────────────────────
    res_int2 = fit_model(
        dm,
        NoLimits.Laplace(
            update_schedule = 2, optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 3)
        );
        rng = MersenneTwister(7), serialization = ser
    )
    @test NoLimits.get_params(res_int; scale = :transformed) ==
        NoLimits.get_params(res_int2; scale = :transformed)

    # ── 5. Scaling: nbatches / |selected| makes the estimate unbiased ────────
    _, batch_infos, const_cache = NoLimits._build_re_batch_infos(dm, NamedTuple())
    ll_cache = build_ll_cache(dm)
    θu = get_θ0_untransformed(dm)
    method = NoLimits.Laplace()
    n_b = length(batch_infos)
    @test n_b == 6

    function mb_obj(sched; cache = nothing)
        st = sched === nothing ? nothing :
            NoLimits._minibatch_current!(
                NoLimits._minibatch_state(sched, n_b, MersenneTwister(3))
            )
        ebe = cache === nothing ?
            _make_laplace_ebe_cache(eltype(θu), n_b) : cache
        return NoLimits._laplace_objective_only(
            dm, batch_infos, θu, const_cache, ll_cache, ebe;
            inner = method.inner, hessian = method.hessian,
            cache_opts = method.cache, multistart = method.multistart,
            rng = MersenneTwister(11), serialization = ser, minibatch = st
        )
    end

    full = mb_obj(nothing)
    singles = [mb_obj(_MBFixed([i])) for i in 1:n_b]
    @test isapprox(sum(singles) / n_b, full; rtol = 1.0e-10)
    @test isapprox(
        mb_obj(_MBFixed([1, 3])), (singles[1] + singles[3]) / 2; rtol = 1.0e-10
    )

    # ── 6. No stale per-batch memo across a schedule advance ─────────────────
    shared = _make_laplace_ebe_cache(eltype(θu), n_b)
    invalidate! = NoLimits._LaplaceMinibatchInvalidate(nothing, shared)
    v1 = mb_obj(_MBFixed([1]); cache = shared)
    invalidate!()
    v2 = mb_obj(_MBFixed([2]); cache = shared)
    @test v1 != v2
    @test isapprox(v1, singles[1]; rtol = 1.0e-10)
    @test isapprox(v2, singles[2]; rtol = 1.0e-10)

    # ── 7. Penalty is added once and is NOT scaled by nbatches/|selected| ────
    pen = (; a = 100.0)
    res_pen = fit_model(
        dm,
        NoLimits.Laplace(
            update_schedule = 2, optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 3)
        );
        penalty = pen, rng = MersenneTwister(5), serialization = ser
    )
    θhat = NoLimits.get_params(res_pen; scale = :untransformed)
    data_part = NoLimits._laplace_objective_only(
        dm, batch_infos, θhat, const_cache, ll_cache,
        _make_laplace_ebe_cache(eltype(θhat), n_b);
        inner = method.inner, hessian = method.hessian,
        cache_opts = method.cache, multistart = method.multistart,
        rng = MersenneTwister(11), serialization = ser
    )
    @test isapprox(
        NoLimits.get_objective(res_pen),
        data_part + NoLimits._penalty_value(θhat, pen);
        rtol = 1.0e-8
    )

    # ── 8. Alternating single-batch schedule exercises cache invalidation ────
    res_alt = fit_model(
        dm,
        NoLimits.Laplace(
            update_schedule = _MBAlternate(), optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 3)
        );
        rng = MersenneTwister(2), serialization = ser
    )
    @test isfinite(NoLimits.get_objective(res_alt))

    # ── 9. Adam under active model bounds uses the projected rule ────────────
    res_proj = fit_model(
        dm,
        NoLimits.Laplace(
            update_schedule = 2, optimizer = Optimisers.Adam(1.0),
            optim_kwargs = (; maxiters = 5),
            lb = (; a = -10.0, σ = -10.0, ω = -10.0),
            ub = (; a = 0.25, σ = 10.0, ω = 10.0)
        );
        rng = MersenneTwister(4), serialization = ser
    )
    θt = NoLimits.get_params(res_proj; scale = :transformed)
    @test all(collect(θt) .>= [-10.0, -10.0, -10.0] .- 1.0e-12)
    @test all(collect(θt) .<= [0.25, 10.0, 10.0] .+ 1.0e-12)
    @test isapprox(θt.a, 0.25; atol = 1.0e-8)
end

@testset "FOCEI mini-batching (#281)" begin
    dm = fx_re_dm()
    ser = SciMLBase.EnsembleSerial()

    rec = _MBRec(Int[], Int[], Any[])
    rng = MersenneTwister(1)
    res = fit_model(
        dm,
        NoLimits.FOCEI(
            update_schedule = rec, optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 4)
        );
        rng = rng, serialization = ser
    )
    @test rec.it == collect(1:4)
    @test all(==(6), rec.nb)
    @test all(x -> x === rng, rec.rngs)
    @test isfinite(NoLimits.get_objective(res))

    mk() = fit_model(
        dm,
        NoLimits.FOCEI(
            update_schedule = 2, optimizer = Optimisers.Adam(0.05),
            optim_kwargs = (; maxiters = 3)
        );
        rng = MersenneTwister(7), serialization = ser
    )
    @test NoLimits.get_params(mk(); scale = :transformed) ==
        NoLimits.get_params(mk(); scale = :transformed)
end

# Optimisers rules only call the fused AD objective, so the Float64 "last feasible
# objective" wall used to stay empty and a good fit was reported as infeasible (#281).
@testset "Laplace with an Optimisers rule is not reported infeasible (#281)" begin
    dm = fx_re_dm()
    logs, res = Test.collect_test_logs(min_level = Base.CoreLogging.Warn) do
        fit_model(
            dm,
            NoLimits.Laplace(
                optimizer = Optimisers.Adam(0.05), optim_kwargs = (; maxiters = 3)
            );
            serialization = SciMLBase.EnsembleSerial()
        )
    end
    @test !any(l -> occursin("never became finite", string(l.message)), logs)
    obj = NoLimits.get_objective(res)
    @test isfinite(obj)
    @test obj < 1.0e9
    @test NoLimits.get_summary(res).notes == NamedTuple()
end
