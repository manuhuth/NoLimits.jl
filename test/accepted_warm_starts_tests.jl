using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Optimization
using OptimizationOptimJL
using Random
using SciMLBase

# Two inner modes: y ~ N(η², 0.1) with y = 4 gives η ≈ ±2. With η ~ N(μ, 1) and μ > 0 the
# positive mode is the better one; μ < 0 makes the negative mode the better one.
function _aws_dm()
    model = @Model begin
        @fixedEffects begin
            μ = RealNumber(0.5)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(μ, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(η^2, 0.1)
        end
    end
    df = DataFrame(ID = [1, 2], t = [0.0, 0.0], y = [4.0, 4.0])
    return DataModel(model, df; primary_id = :ID, time_col = :t)
end

function _aws_setup(dm, warm::Bool)
    method = NoLimits.Laplace(; multistart_n = 0, multistart_k = 0)
    inner = NoLimits._resolve_inner_options(method.inner, dm)
    ms = NoLimits._resolve_multistart_options(method.multistart, inner)
    _, binfos, cc = NoLimits._build_re_batch_infos(dm, NamedTuple())
    llc = build_ll_cache(dm; force_saveat = true)
    cache = NoLimits._init_laplace_eval_cache(length(binfos), Float64)
    warm && (cache.warm = [Float64[] for _ in binfos])
    kw = (;
        inner = inner, hessian = method.hessian, cache_opts = method.cache, multistart = ms,
        rng = Random.Xoshiro(1), serialization = SciMLBase.EnsembleSerial(),
    )
    obj(μ) = NoLimits._laplace_objective_only(
        dm, binfos, ComponentArray(μ = μ), cc, llc, cache; kw...
    )
    return (; obj, cache, binfos, cc, llc, inner)
end

@testset "AcceptedWarmStarts" begin
    dm = _aws_dm()

    @testset "defaults unchanged" begin
        @test NoLimits.Laplace().warm_starts === nothing
        @test NoLimits.FOCEI().warm_starts === nothing
        @test NoLimits._init_laplace_eval_cache(2, Float64).warm === nothing
        @test_throws ErrorException NoLimits.Laplace(;
            warm_starts = AcceptedWarmStarts(), update_schedule = :minibatch
        )
        @test_throws ErrorException AcceptedWarmStarts(; multistart_every = 0)
        @test !NoLimits._supports_refresh(OptimizationOptimJL.LBFGS())
        # callback protocol after a multistart that changed the modes
        st = (; iter = 2, u = [0.0])
        cb(refresh_ok, user) = NoLimits._AcceptedWarmCallback(_ -> true, refresh_ok, user, "Laplace")
        @test cb(true, nothing)(st, 0.0) === :refresh
        @test cb(true, (s, f) -> true)(st, 0.0) === true            # the user's halt wins
        @test (@test_logs (:warn, r"stale") cb(false, nothing)(st, 0.0)) === false
        @test NoLimits._AcceptedWarmCallback(_ -> false, true, nothing, "L")(st, 0.0) === false
    end

    @testset "value at θ does not depend on earlier trial points" begin
        # warm starts committed once at θ0 = 0.5 (positive mode); the trial point θ_far = -3
        # pulls the modes to the negative branch.
        s1 = _aws_setup(dm, true)
        s1.obj(0.5)
        NoLimits._commit_warm!(s1.cache)
        s1.obj(-3.0)
        v_hist = s1.obj(0.4)
        s2 = _aws_setup(dm, true)
        s2.obj(0.5)
        NoLimits._commit_warm!(s2.cache)
        v_direct = s2.obj(0.4)
        @test v_hist == v_direct
        @test all(b -> b[1] > 0, s1.cache.bstar_cache.b_star)
        # default rule: warm start from the last evaluated θ, so the history matters here
        d1 = _aws_setup(dm, false)
        d1.obj(0.5)
        d1.obj(-3.0)
        d2 = _aws_setup(dm, false)
        d2.obj(0.5)
        @test d1.obj(0.4) != d2.obj(0.4)
    end

    @testset "multistart finds the better mode" begin
        s = _aws_setup(dm, true)
        s.obj(-3.0)               # modes on the negative branch
        NoLimits._commit_warm!(s.cache)
        s.obj(0.5)                # warm-started: stays on the negative (worse) branch
        @test all(b -> b[1] < 0, s.cache.bstar_cache.b_star)
        v_before = s.obj(0.5)
        ws = AcceptedWarmStarts(; multistart_every = 1)
        changed = NoLimits._laplace_accepted_multistart!(
            s.cache, dm, s.binfos, ComponentArray(μ = 0.5), s.cc, s.llc, s.inner, ws
        )
        @test changed
        @test all(b -> b[1] > 0, s.cache.bstar_cache.b_star)
        NoLimits._commit_warm!(s.cache)
        s.cache.θ_cache = nothing
        @test s.obj(0.5) < v_before
        # nothing to improve: no change
        @test !NoLimits._laplace_accepted_multistart!(
            s.cache, dm, s.binfos, ComponentArray(μ = 0.5), s.cc, s.llc, s.inner, ws
        )
    end

    @testset "legacy options: initial multistart, no refresh, per-individual seed" begin
        ws = AcceptedWarmStarts()
        @test !ws.initial_multistart && ws.refresh
        st = (; iter = 2, u = [0.0])
        # refresh = false: stale values are intended, no :refresh and no warning
        cb = NoLimits._AcceptedWarmCallback(_ -> true, false, nothing, "FOCEI", false)
        @test (@test_logs cb(st, 0.0)) === false
        # seed = seed + individual index (data order) for one-individual batches
        s = _aws_setup(dm, true)
        ws7 = AcceptedWarmStarts(; seed = 7)
        for (bi, info) in enumerate(s.binfos)
            @test rand(NoLimits._aws_rng(ws7, info, bi)) == rand(Random.Xoshiro(7 + only(info.inds)))
        end
        # initial multistart: with μ = 0 the default start η = 0 is a stationary point of
        # the log-joint, the inner solve stays there; the multistart finds a mode at |η| ≈ 2
        s0 = _aws_setup(dm, true)
        s0.obj(0.0)
        @test all(b -> abs(b[1]) < 1.0e-6, s0.cache.bstar_cache.b_star)
        wsi = AcceptedWarmStarts(; initial_multistart = true)
        @test NoLimits._laplace_accepted_multistart!(
            s0.cache, dm, s0.binfos, ComponentArray(μ = 0.0), s0.cc, s0.llc, s0.inner, wsi;
            from_default = true
        )
        @test all(b -> abs(abs(b[1]) - 2) < 0.1, s0.cache.bstar_cache.b_star)
        # in a fit from μ = 0 the modes stay at the saddle η = 0 (no finite Laplace value)
        # unless the initial multistart moves them
        for ims in (false, true)
            logs, _ = Test.collect_test_logs() do
                fit_model(
                    dm, NoLimits.Laplace(;
                        warm_starts = AcceptedWarmStarts(; initial_multistart = ims, refresh = false),
                        optim_kwargs = (; maxiters = 2)
                    );
                    theta_0_untransformed = ComponentArray(μ = 0.0),
                    serialization = SciMLBase.EnsembleSerial()
                )
            end
            @test any(l -> occursin("never became finite", string(l.message)), logs) == !ims
        end
    end

    @testset "fit with AcceptedWarmStarts" begin
        calls = Ref(0)
        user_cb = (state, f) -> (calls[] += 1; false)
        for M in (NoLimits.Laplace, NoLimits.FOCEI)
            calls[] = 0
            res = fit_model(
                dm, M(;
                    warm_starts = AcceptedWarmStarts(; multistart_every = 2),
                    optim_kwargs = (; maxiters = 20, callback = user_cb)
                );
                serialization = SciMLBase.EnsembleSerial()
            )
            @test isfinite(NoLimits.get_objective(res))
            @test calls[] > 0                       # the user's callback still runs
        end
    end
end
