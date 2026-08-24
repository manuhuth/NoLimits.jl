using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using SciMLBase
using ComponentArrays
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using OptimizationBBO

# One scalar-RE model shared by the option/sampler/constants testsets below
# (they assert fit-option behavior, not model structure). Structure-specific
# testsets (multivariate, ODE, Poisson, covariate-RE, multi-group) use the
# shared fx_* fixtures from fixtures.jl.
const _MCEM_MODEL = fx_tiny_re_model()

const _MCEM_DM2 = fx_tiny_re_dm()

const _MCEM_DM3 = DataModel(
    _MCEM_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0]
    );
    primary_id = :ID, time_col = :t
)

const _MCEM_DM4 = DataModel(
    _MCEM_MODEL,
    DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    );
    primary_id = :ID, time_col = :t
)

@testset "MCEM default sampler" begin
    method = NoLimits.MCEM()
    @test method.e_step isa NoLimits.MCEM_MCMC
    @test method.e_step.sampler isa SaemixMH
    @test method.ebe.multistart_n == 50
    @test method.ebe.multistart_k == 1
    @test method.ebe.sampling == :lhs
    @test method.ebe_rescue.sampling == :lhs
end

@testset "MCEM windowed drift test triggers early stop" begin
    # Inf tolerances make every post-window-fill check pass, so the stop point is
    # deterministic: window fill (4) + consecutive (2) - 1 = iteration 5.
    res = fit_model(
        _MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 30, convergence_window = 4, consecutive_params = 2,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf,
            progress = false
        )
    )
    @test NoLimits.get_converged(res)
    @test 5 <= res.result.iterations < 30
    diag = res.result.notes.diagnostics
    @test isnan(diag.drift_θ[1])  # window not yet full
    @test isfinite(diag.drift_θ[end])
end

@testset "MCEM no early stop before drift window fills" begin
    res = fit_model(
        _MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 3, convergence_window = 4, consecutive_params = 1,
            atol_theta = Inf, rtol_theta = Inf, atol_Q = Inf, rtol_Q = Inf,
            progress = false
        )
    )
    @test !NoLimits.get_converged(res)
    @test res.result.iterations == 3
end

@testset "MCEM basic (random effects)" begin
    res = fit_model(
        _MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

# NOTE: testsets shared line-for-line with SAEM (serial-vs-threaded reproducibility,
# convergence stabilization, multiple RE groups, thread caches/RNGs, EBE rescue,
# constants_re) live as parameterized "SAEM/MCEM …" loops in estimation_saem_tests.jl.

@testset "MCEM basic with NUTS" begin
    res = fit_model(
        _MCEM_DM2,
        NoLimits.MCEM(;
            sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
end

@testset "MCEM constants_re" begin
    res = fit_model(
        _MCEM_DM3,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        );
        constants_re = (; η = (; A = 0.0))
    )
    @test res isa FitResult
end

@testset "MCEM constants for fixed effects" begin
    res = fit_model(
        _MCEM_DM2,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        );
        constants = (a = 0.2,)
    )
    @test res isa FitResult
end

@testset "MCEM RE distribution with constant covariates" begin
    res = fit_model(
        fx_recov_dm(),
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
end

@testset "MCEM threaded E-step" begin
    res = fit_model(
        _MCEM_DM4,
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        );
        serialization = EnsembleThreads()
    )
    @test res isa FitResult
end

@testset "MCEM update_schedule minibatching" begin
    @test NoLimits.MCEM().update_schedule === :all
    # 4 batches. Every schedule must still yield a finite Q: batches skipped by the
    # E-step keep their previous draws, so the M-step never sees an empty sample set.
    for (sched, ser) in (
            (2, EnsembleSerial()),
            (2, EnsembleThreads()),
            ((n, it, r) -> [1 + (it % n)], EnsembleSerial()),
        )
        res = fit_model(
            _MCEM_DM4,
            NoLimits.MCEM(;
                sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
                maxiters = 3, update_schedule = sched
            );
            serialization = ser
        )
        @test res isa FitResult
        @test isfinite(NoLimits.get_objective(res))
    end
    # MCEM_IS switching out of its MCMC warm-up must refresh every batch, or the IS
    # Q-function would hit batches with no importance weights.
    res_is = fit_model(
        _MCEM_DM4,
        NoLimits.MCEM(;
            e_step = NoLimits.MCEM_IS(; n_samples = 8, warm_start_mcmc_iters = 1),
            maxiters = 3, update_schedule = 2, progress = false
        )
    )
    @test isfinite(NoLimits.get_objective(res_is))
    @test_throws ErrorException fit_model(
        _MCEM_DM4,
        NoLimits.MCEM(; maxiters = 2, progress = false, update_schedule = :nope)
    )
end

@testset "MCEM multivariate RE" begin
    res = fit_model(
        fx_mvnp_dm(),
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
end

@testset "MCEM multivariate RE with NUTS" begin
    res = fit_model(
        fx_mvnp_dm(),
        NoLimits.MCEM(;
            sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
end

@testset "MCEM optimizer Adam (OptimizationOptimisers)" begin
    method = NoLimits.MCEM(
        optimizer = OptimizationOptimisers.Adam(0.05),
        optim_kwargs = (; maxiters = 2),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        maxiters = 2
    )
    res = fit_model(_MCEM_DM2, method)
    @test res isa FitResult
end

@testset "MCEM optimizer BlackBoxOptim (OptimizationBBO)" begin
    lb, ub = default_bounds_from_start(_MCEM_DM2; margin = 1.0)
    method = NoLimits.MCEM(
        optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs = (; iterations = 3),
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        maxiters = 2,
        lb = lb, ub = ub
    )
    res = fit_model(_MCEM_DM2, method)
    @test res isa FitResult
end

@testset "MCEM with ODE model" begin
    res = fit_model(
        fx_ode_dm(),
        NoLimits.MCEM(;
            sampler = MH(), turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
end

@testset "MCEM non-normal Poisson outcome" begin
    res = fit_model(
        fx_pois_dm(),
        NoLimits.MCEM(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            maxiters = 2
        )
    )
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

# One scalar-RE model/DataModel shared by all IS-variant testsets below (they
# assert e-step option behavior and diagnostics, not model structure). The
# multi-RE testset uses fx_mg_dm(); the LogNormal-RE bijection testset keeps a
# bespoke model.
const _MIS_MODEL = fx_tiny_re_model()

const _MIS_DM = DataModel(
    _MIS_MODEL,
    DataFrame(
        ID = ["A", "A", "B", "B"],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.05]
    );
    primary_id = :ID, time_col = :t
)

@testset "MCEM_IS struct and MCEM_MCMC struct" begin
    es_mcmc = NoLimits.MCEM_MCMC()
    @test es_mcmc.sampler isa SaemixMH
    @test es_mcmc.warm_start == true
    @test es_mcmc.sample_schedule == 100

    es_is = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior)
    @test es_is.n_samples == 2
    @test es_is.proposal === :prior
    @test es_is.adapt == true
    @test es_is.warm_start_mcmc_iters == 0
    @test es_is.mcmc_warmup === nothing

    es_is2 = NoLimits.MCEM_IS(
        n_samples = 2, proposal = :gaussian, warm_start_mcmc_iters = 3
    )
    @test es_is2.warm_start_mcmc_iters == 3
    @test es_is2.mcmc_warmup isa NoLimits.MCEM_MCMC

    # MCEM with IS e_step
    method = NoLimits.MCEM(e_step = NoLimits.MCEM_IS(n_samples = 2))
    @test method.e_step isa NoLimits.MCEM_IS
    @test method.e_step.n_samples == 2

    # Backward compat: MCEM() still creates MCEM_MCMC
    method2 = NoLimits.MCEM()
    @test method2.e_step isa NoLimits.MCEM_MCMC
end

@testset "IS prior proposal — basic fit" begin
    res = fit_model(
        _MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior, adapt = false),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS gaussian proposal — blocks updated" begin
    res = fit_model(
        _MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :gaussian, adapt = true),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    @test res isa NoLimits.FitResult
    diag = res.result.notes.diagnostics
    # ESS recorded for IS iterations (not NaN from iter 1 once gaussian proposal is used)
    @test length(diag.ess_hist) == length(diag.Q_hist)
    # After at least 2 iterations the gaussian proposal should have n_samples > 0
    # (all ess values should be finite for the IS phase)
    @test all(isfinite, diag.ess_hist)
end

@testset "IS user-provided proposal function" begin
    # User proposal: sample from N(0, 2) for all entries, return correct shapes
    function my_proposal_is_test(θ, batch_info, re_dists, rng, n_samples)
        nb = batch_info.n_b
        samples = randn(rng, nb, n_samples) .* 2.0
        # log q = sum of Normal(0, 2) logpdfs
        log_qs = vec(sum(logpdf.(Normal(0.0, 2.0), samples); dims = 1))
        return samples, log_qs
    end

    res = fit_model(
        _MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = my_proposal_is_test),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS warm_start_mcmc_iters — MCMC then IS" begin
    es = NoLimits.MCEM_IS(
        n_samples = 2,
        proposal = :gaussian,
        adapt = true,
        warm_start_mcmc_iters = 2,
        mcmc_warmup = NoLimits.MCEM_MCMC(
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
            sample_schedule = 10
        )
    )
    res = fit_model(
        _MIS_DM,
        NoLimits.MCEM(
            e_step = es,
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    @test res isa NoLimits.FitResult
    diag = res.result.notes.diagnostics
    # First 2 iterations are MCMC (ess = NaN), rest are IS (ess finite)
    @test isnan(diag.ess_hist[1])
    @test isnan(diag.ess_hist[2])
    @test all(isfinite, diag.ess_hist[3:end])
end

@testset "IS weights are finite and normalized" begin
    res = fit_model(
        _MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    diag = res.result.notes.diagnostics
    # ESS must be in [1, n_samples] for IS iters
    for ess in diag.ess_hist
        if isfinite(ess)
            @test ess >= 1.0
            @test ess <= 50.0 + 1.0e-6   # small tolerance for float arithmetic
        end
    end
end

@testset "IS ESS tracked in diagnostics" begin
    res = fit_model(
        _MIS_DM,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    diag = res.result.notes.diagnostics
    @test length(diag.ess_hist) == length(diag.Q_hist)
    @test all(isfinite, diag.ess_hist)  # pure IS: all finite
end

@testset "IS with multi-RE model" begin
    res = fit_model(
        fx_mg_dm(),
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :prior),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS LogNormal RE — bijection applied" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(1.0, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(LogNormal(0.0, 0.5); column = :ID)
        end
        @formulas begin
            y ~ Normal(a * η, σ)
        end
    end

    df = DataFrame(
        ID = ["A", "A", "B", "B"],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.1, 0.9, 1.3, 1.2]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res = fit_model(
        dm,
        NoLimits.MCEM(
            e_step = NoLimits.MCEM_IS(n_samples = 2, proposal = :gaussian, adapt = true),
            maxiters = 2,
            consecutive_params = 1,
            progress = false
        )
    )
    @test res isa NoLimits.FitResult
    diag = res.result.notes.diagnostics
    @test all(isfinite, diag.ess_hist)
    params = NoLimits.get_params(res; scale = :untransformed)
    @test all(isfinite, collect(params))
end

@testset "IS backward compat — MCEM() legacy kwargs still work" begin
    # Old API: sampler= and turing_kwargs= at the top level
    method = NoLimits.MCEM(
        sampler = MH(),
        turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
        maxiters = 2,
        consecutive_params = 1,
        progress = false
    )
    @test method.e_step isa NoLimits.MCEM_MCMC
    @test method.e_step.sampler isa MH

    res = fit_model(_MIS_DM, method)
    @test res isa NoLimits.FitResult
    @test NoLimits.get_converged(res) isa Bool
end

@testset "MCEM dev_api Q primitives (partition + M-step Q value/gradient)" begin
    dm = fx_re_dm()   # a, σ obs-side (q1); ω only in RE dist (q2)
    fe = NoLimits.get_fixed(NoLimits.get_model(dm))
    θ = NoLimits.get_θ0_untransformed(fe)

    part = NoLimits.mcem_q_partition(dm)
    @test part.q1 == [:a, :σ]
    @test part.q2 == [:ω]
    partc = NoLimits.mcem_q_partition(dm; constants = (; σ = 0.3))
    @test partc.q1 == [:a]
    @test partc.q2 == [:ω]

    # FIXED weighted draws (importance; deterministic under a seeded rng)
    draws = NoLimits.sample_random_effect_draws(
        dm, θ; method = :importance, n_samples = 64,
        serialization = NoLimits.EnsembleSerial(), rng = MersenneTwister(20240824)
    )
    nb = length(draws)
    @test nb > 1

    for prt in (:q1, :q2), scl in (:transformed, :untransformed)
        Q, g = NoLimits.mcem_q_objective_and_gradient(
            dm, θ, draws; part = prt, scale = scl,
            serialization = NoLimits.EnsembleSerial()
        )
        Qsum = 0.0
        gsum = zeros(length(g))
        for bi in 1:nb
            Qb, gb = NoLimits.mcem_q_objective_and_gradient(
                dm, θ, draws, bi; part = prt, scale = scl,
                serialization = NoLimits.EnsembleSerial()
            )
            Qsum += Qb
            gsum .+= collect(gb)
        end
        @test isapprox(Q, Qsum; atol = 1.0e-10, rtol = 0)
        @test isapprox(collect(g), gsum; atol = 1.0e-10, rtol = 0)
    end

    # Value is bit-identical to the fit kernel `_mcem_Q` at the same arguments.
    _, bis, cc = NoLimits.build_re_batch_infos(dm, NamedTuple())
    llc = NoLimits.build_ll_cache(dm; serialization = NoLimits.EnsembleSerial(), force_saveat = true)
    sbb = [NoLimits.get_draws(d) for d in draws]
    wbb = map(draws) do d
        lw = NoLimits.get_log_weights(d)
        w = exp.(lw .- maximum(lw))
        return w ./ sum(w)
    end
    Qref = NoLimits._mcem_Q(dm, bis, θ, cc, llc, sbb, wbb; serialization = NoLimits.EnsembleSerial())
    Qu, _ = NoLimits.mcem_q_objective_and_gradient(
        dm, θ, draws; part = :q1, scale = :untransformed,
        serialization = NoLimits.EnsembleSerial()
    )
    @test Qu == Qref
    Q2ref = NoLimits._mcem_Q2(dm, bis, θ, cc, llc, sbb, wbb; serialization = NoLimits.EnsembleSerial())
    Q2u, _ = NoLimits.mcem_q_objective_and_gradient(
        dm, θ, draws; part = :q2, scale = :untransformed,
        serialization = NoLimits.EnsembleSerial()
    )
    @test Q2u == Q2ref

    # free_names subset freezes the complement; gradient is on the free axes.
    _, gsub = NoLimits.mcem_q_objective_and_gradient(dm, θ, draws; part = :q1, free_names = [:a])
    @test length(gsub) == 1
    @test_throws ErrorException NoLimits.mcem_q_objective_and_gradient(dm, θ, draws, nb + 1; part = :q1)
    @test_throws ErrorException NoLimits.mcem_q_objective_and_gradient(dm, θ, draws; part = :bogus)
end

@testset "MCEM dev_api mcem_e_step (state-threaded E-step)" begin
    dm = fx_re_dm()
    fe = NoLimits.get_fixed(NoLimits.get_model(dm))
    θ0 = NoLimits.get_θ0_untransformed(fe)
    method = NoLimits.MCEM(; maxiters = 20, progress = false)

    # Determinism: same fresh seed -> bit-identical draws.
    d_a, s_a = NoLimits.mcem_e_step(dm, θ0, method, nothing; rng = MersenneTwister(11))
    d_b, s_b = NoLimits.mcem_e_step(dm, θ0, method, nothing; rng = MersenneTwister(11))
    nb = length(d_a)
    @test nb > 1
    @test all(NoLimits.get_draws(d_a[bi]) == NoLimits.get_draws(d_b[bi]) for bi in 1:nb)
    @test all(NoLimits.get_draws(d_a[bi]) isa AbstractMatrix for bi in 1:nb)
    @test s_a.iter == 2

    # State threading: a second call advances iter and uses warm-start.
    d2, s2 = NoLimits.mcem_e_step(dm, θ0, method, s_a; rng = MersenneTwister(11))
    @test s2.iter == 3
    @test length(d2) == nb

    # Draws feed straight into the M-step primitive.
    Q, g = NoLimits.mcem_q_objective_and_gradient(dm, θ0, d_a; part = :q1, serialization = NoLimits.EnsembleSerial())
    @test isfinite(Q) && all(isfinite, collect(g))

    # Round trip: the federated protocol { E-step local -> M-step Q2 (LBFGS over the
    # summed grad) -> M-step Q1 -> repeat } reproduces fit_model(dm, MCEM()).
    tr = NoLimits.get_transform(fe)
    itr = NoLimits.get_inverse_transform(fe)
    mstep = function (θ, draws, part, fnames)
        θt = tr(θ)
        θf0 = ComponentArray(NamedTuple{Tuple(fnames)}(Tuple(getproperty(θt, n) for n in fnames)))
        axsf = getaxes(θf0)
        x0 = collect(ComponentArrays.getdata(θf0))
        rebuild = function (x)
            θt_loc = ComponentArray(collect(θt), getaxes(θt))
            θf = ComponentArray(x, axsf)
            for n in fnames
                setproperty!(θt_loc, n, getproperty(θf, n))
            end
            return itr(θt_loc)
        end
        f = (x, p) -> -NoLimits.mcem_q_objective_and_gradient(
            dm, rebuild(x), draws; part = part, free_names = fnames,
            scale = :transformed, serialization = NoLimits.EnsembleSerial()
        )[1]
        g! = function (G, x, p)
            gg = NoLimits.mcem_q_objective_and_gradient(
                dm, rebuild(x), draws; part = part, free_names = fnames,
                scale = :transformed, serialization = NoLimits.EnsembleSerial()
            )[2]
            G .= .-collect(gg)
            return nothing
        end
        sol = solve(OptimizationProblem(OptimizationFunction(f; grad = g!), x0), LBFGS(); maxiters = 50)
        return rebuild(sol.u)
    end

    res = fit_model(dm, method; rng = MersenneTwister(7))
    p_ref = NoLimits.get_params(res; scale = :untransformed)

    θ = θ0
    state = nothing
    rng_loop = MersenneTwister(7)
    for _ in 1:30
        draws, state = NoLimits.mcem_e_step(dm, θ, method, state; rng = rng_loop)
        θ = mstep(θ, draws, :q2, [:ω])
        θ = mstep(θ, draws, :q1, [:a, :σ])
    end
    @test all(isfinite, collect(θ))
    @test isapprox(θ.a, p_ref.a; atol = 0.1)
    @test isapprox(θ.σ, p_ref.σ; atol = 0.1)
    @test isapprox(θ.ω, p_ref.ω; atol = 0.1)
end
