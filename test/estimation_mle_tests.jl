using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using Optimization
using OptimizationBBO
using OptimizationOptimisers
using OptimizationOptimJL

# Shared MLE/MAP testsets: models carry priors (MLE ignores them, MAP requires them).
const MLE_MAP_METHODS = (("MLE", NoLimits.MLE()), ("MAP", NoLimits.MAP()))

# Shared exp-mean model; one @Model serves several testsets (compile-bound suite).
const MLE_EXP_MODEL = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        a = RealNumber(0.1; prior = Normal(0.0, 1.0))
        σ = RealNumber(0.5, scale = :log, prior = LogNormal(0.0, 0.5))
    end

    @formulas begin
        y ~ Normal(exp(a), σ)
    end
end

function _mle_dm_basic()
    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )
    return DataModel(MLE_EXP_MODEL, df; primary_id = :ID, time_col = :t)
end

@testset "MLE non-ODE" begin
    res = fx_mle()                        # shared no-RE MLE fit
    @test res isa FitResult
    @test NoLimits.get_params(res; scale = :untransformed) isa ComponentArray

    # #310: a finite but overflowing observation makes the objective constant, so the fit
    # returns the starting values; it must say so and name the individual.
    df_bad = copy(fx_nore_df())
    df_bad.y[8] = 1.0e200
    dm_bad = DataModel(fx_nore_model(), df_bad; primary_id = :ID, time_col = :t)
    @test_logs (:warn, r"never became finite.*ID\): 4") match_mode = :any fit_model(
        dm_bad, NoLimits.MLE(); serialization = NoLimits.EnsembleSerial()
    )
end

let
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; prior = Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1^2
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
        y = [1.0, 1.05]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t)

    for (label, method) in MLE_MAP_METHODS
        @testset "$label ODE" begin
            res = fit_model(dm, method)

            @test res isa FitResult
        end
    end
end

@testset "MLE ODE with parameterized initial state" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            ka = RealNumber(1.0, prior = Normal(1.0, 0.5))
            ke = RealNumber(0.1, prior = Normal(0.1, 0.05))
            V = RealNumber(20.0, prior = Normal(20.0, 5.0))
            D = RealNumber(320.0, prior = Normal(320.0, 50.0))
            σ = RealNumber(1.0, scale = :log, prior = LogNormal(0.0, 0.5))
        end

        @DifferentialEquation begin
            D(A) ~ -ka * A
            D(C) ~ (ka * A) / V - ke * C
        end

        @initialDE begin
            A = D
            C = 0.0
        end

        @formulas begin
            y ~ Normal(C(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1.0, 1.05, 0.98, 1.02]
    )

    model_saveat = set_solver_config(model; saveat_mode = :saveat)
    dm = DataModel(model_saveat, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE())

    @test res isa FitResult
end

@testset "MLE rejects random effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            y ~ Normal(exp(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm, NoLimits.MLE())
end

let
    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(MLE_EXP_MODEL, df; primary_id = :ID, time_col = :t)

    for (label, method) in MLE_MAP_METHODS
        @testset "$label requires a free fixed effect" begin
            @test_throws ErrorException fit_model(
                dm, method; constants = (a = 0.2, σ = 0.3)
            )
        end
    end
end

let
    model = @Model begin
        @fixedEffects begin
            β = RealVector([0.2, -0.1], prior = MvNormal(zeros(2), LinearAlgebra.I))
            σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
        end

        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @formulas begin
            μ = exp(β[1] * z + β[2] * z^2)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        z = [0.1, 0.2, 0.15, 0.3],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    for (label, method) in MLE_MAP_METHODS
        @testset "$label fixed vector parameters" begin
            res = fit_model(dm, method)

            @test res isa FitResult
        end
    end
end

let
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; prior = Normal(0.0, 1.0))
            σ = RealNumber(0.5; lower = 0.3, scale = :identity, prior = LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    for (label, method) in MLE_MAP_METHODS
        @testset "$label respects bounds (σ lower bound)" begin
            res = fit_model(dm, method)

            θu = NoLimits.get_params(res; scale = :untransformed)
            @test θu.σ >= 0.3
        end
    end
end

let
    dm = _mle_dm_basic()

    for (label, method) in MLE_MAP_METHODS
        @testset "$label constants" begin
            res = fit_model(dm, method; constants = (a = 0.0,))

            θu = NoLimits.get_params(res; scale = :untransformed)
            @test θu.a == 0.0
        end
    end

    # #169: invalid overrides report themselves at call time, not as a raw DomainError /
    # FieldError from deep inside the objective.
    @testset "override validation" begin
        @test_throws ErrorException fit_model(dm, NoLimits.MLE(); constants = (σ = -1.0,))
        @test_throws ErrorException fit_model(dm, NoLimits.MLE(); penalty = (bb = 1.0,))
        # #165: σ = 0 has zero LogNormal prior density; the objective's Inf short-circuit
        # would hand the optimizer a zero gradient and it would report convergence there.
        # It also sits exactly on the boundary of the :log transform, which #249 now
        # rejects first with an ArgumentError.
        @test_throws ArgumentError fit_model(
            dm, NoLimits.MAP();
            theta_0_untransformed = ComponentArray(a = 0.1, σ = 0.0)
        )
    end
end

@testset "MLE penalties" begin
    dm = _mle_dm_basic()
    res_no_penalty = fit_model(dm, NoLimits.MLE())
    res = fit_model(dm, NoLimits.MLE(); penalty = (a = 100.0,))

    θu0 = NoLimits.get_params(res_no_penalty; scale = :untransformed)
    θu = NoLimits.get_params(res; scale = :untransformed)
    @test abs(θu.a) ≤ abs(θu0.a)
end

@testset "MLE penalty mimics Normal prior" begin
    model_prior = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; prior = Normal(0.0, 1.0))
        end

        @formulas begin
            y ~ Normal(exp(a), 2.0)
        end
    end

    model_penalty = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            #σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(exp(a), 2.0)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm_prior = DataModel(model_prior, df; primary_id = :ID, time_col = :t)
    dm_penalty = DataModel(model_penalty, df; primary_id = :ID, time_col = :t)
    res_map = fit_model(dm_prior, NoLimits.MAP())
    res_pen = fit_model(dm_penalty, NoLimits.MLE(); penalty = (a = 0.5,))

    a_map = NoLimits.get_params(res_map; scale = :untransformed).a
    a_pen = NoLimits.get_params(res_pen; scale = :untransformed).a
    @test isapprox(a_map, a_pen; rtol = 1.0e-4, atol = 1.0e-4)
end

@testset "MLE uses optim_kwargs" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(optim_kwargs = (; iterations = 1))
    res = fit_model(dm, method)

    @test res isa FitResult
    stats = res.result.solution.stats
    @test hasproperty(stats, :iterations)
    @test stats.iterations <= 1
end

@testset "MLE accepts lb-only user bounds" begin
    dm = _mle_dm_basic()
    lb = ComponentArray((; a = -2.0, σ = -3.0))
    res = fit_model(dm, NoLimits.MLE(lb = lb))
    @test res isa FitResult
end

@testset "MLE accepts ub-only user bounds" begin
    dm = _mle_dm_basic()
    ub = ComponentArray((; a = 2.0, σ = 2.0))
    res = fit_model(dm, NoLimits.MLE(ub = ub))
    @test res isa FitResult
end

@testset "MLE BBO requires finite bounds on both sides" begin
    dm = _mle_dm_basic()
    lb = ComponentArray((; a = -2.0, σ = -3.0))
    method = NoLimits.MLE(
        optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        lb = lb, optim_kwargs = (; iterations = 5)
    )
    err = try
        fit_model(dm, method)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("finite lower and upper bounds", sprint(showerror, err))
end

@testset "MLE optimizer BFGS (Optim)" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(optimizer = Optim.BFGS(), optim_kwargs = (;))
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "MLE optimizer NelderMead (Optim)" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(optimizer = Optim.NelderMead(), optim_kwargs = (;))
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "MLE optimizer Adam (OptimizationOptimisers)" begin
    dm = _mle_dm_basic()
    method = NoLimits.MLE(
        optimizer = OptimizationOptimisers.Adam(0.05), optim_kwargs = (; maxiters = 2)
    )
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "MLE optimizer BlackBoxOptim (OptimizationBBO)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1; lower = -2.0, upper = 2.0, scale = :identity)
            σ = RealNumber(0.5; lower = 0.1, upper = 2.0, scale = :identity)
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    method = NoLimits.MLE(
        optimizer = OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
        optim_kwargs = (; iterations = 5)
    )
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "MLE non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
        end

        @formulas begin
            λ = exp(a + b * z)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.5, 1.0, 1.5],
        y = [1, 1, 2, 3]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE())

    @test res isa FitResult
    θu = NoLimits.get_params(res; scale = :untransformed)
end

let
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0, prior = Uniform(0.1, 1.0))
        end

        @formulas begin
            y ~ Poisson(a)
        end
    end

    df = DataFrame(
        ID = [1],
        t = [0.0],
        y = [1.0]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    @testset "MLE handles +Inf objective in AD path" begin
        res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))

        @test res isa FitResult
        @test !isfinite(NoLimits.get_objective(res))
    end

    # #165: a = 0 lies outside the Uniform(0.1, 1.0) prior support, so the MAP objective is
    # +Inf with zero AD partials there - the optimizer used to read that as a converged
    # optimum. It must be refused instead of "fitted".
    @testset "MAP rejects a start outside the prior support" begin
        @test_throws ErrorException fit_model(
            dm, NoLimits.MAP(; optim_kwargs = (maxiters = 2,))
        )
    end
end

# Shared by the two validation testsets below; per-fit kwargs never mutate the model.
const MLE_VALIDATION_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        s = RealNumber(0.5, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a, s)
    end
end

@testset "fit_model starting-value validation" begin
    model = MLE_VALIDATION_MODEL
    df = DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [1.0, 1.1, 0.9, 1.0])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    # A plain NamedTuple start is accepted, like `constants`/`penalty`.
    res = fit_model(dm, NoLimits.MLE(); theta_0_untransformed = (a = 1.2, s = 0.4))
    @test res isa FitResult
    @test_throws ErrorException fit_model(
        dm, NoLimits.MLE();
        theta_0_untransformed = (a = 1.2, nope = 0.4)
    )
    @test_throws ErrorException fit_model(
        dm, NoLimits.MLE();
        theta_0_untransformed = ComponentArray(a = NaN, s = 0.3)
    )
    @test_throws ErrorException fit_model(
        dm, NoLimits.MLE();
        theta_0_untransformed = ComponentArray(a = Inf, s = 0.3)
    )

    # A non-finite objective is never reported as converged (#208/#209/#214/#215).
    summ = NoLimits.FitSummary(Inf, true, NoLimits.get_params(res), nothing)
    @test summ.converged == false
end

@testset "fit_model option validation" begin
    model = MLE_VALIDATION_MODEL
    dm = DataModel(
        model, DataFrame(ID = [1, 2], t = [0.0, 0.0], y = [0.1, 0.2]);
        primary_id = :ID, time_col = :t
    )
    K = (maxiters = 1,)
    fit(m; kw...) = fit_model(dm, m; kw...)

    # lb/ub are checked at the shared bounds resolver, so every estimator inherits it.
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K, ub = [0.4]))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K, ub = [NaN, NaN]))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K, lb = Float64[]))
    @test_throws ErrorException fit(
        NoLimits.MLE(
            optim_kwargs = K, lb = [0.3, 0.3], ub = [0.2, 0.2]
        )
    )
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); constants = [1.0, 2.0])
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); penalty = (a = "x",))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); penalty = (a = NaN,))
    @test_throws ErrorException fit(
        NoLimits.MLE(optim_kwargs = K); extra_objective = (x, y) -> 0.0
    )
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); extra_objective = 1)
    # Unknown / out-of-domain parameter overrides are named, on both entry points.
    @test_throws ErrorException fit(
        NoLimits.MLE(optim_kwargs = K);
        theta_0_untransformed = ComponentArray(a = 0.2, s = 0.3, x = 9.0)
    )
    @test_throws ErrorException simulate_data(
        dm; theta_untransformed = ComponentArray(a = 0.2, s = -1.0)
    )
    @test_throws ErrorException simulate_data(
        dm; theta_untransformed = ComponentArray(a = NaN, s = 0.3)
    )

    # #311: ParticleSwarm samples particles uniformly inside the box (an infinite edge
    # gives all-NaN estimates with a finite objective) and IPNewton is barrier-only, so
    # both are rejected instead of silently returning junk or throwing from inside Optim.
    @test_throws ErrorException fit(
        NoLimits.MLE(
            optimizer = Optim.ParticleSwarm(), optim_kwargs = (; maxiters = 5),
            lb = [-2.0, -2.0], ub = [Inf, Inf]
        )
    )
    @test_throws ErrorException fit(
        NoLimits.MLE(optimizer = Optim.IPNewton(), optim_kwargs = K)
    )
    # #311: a start outside the box used to surface as `Initial x[(1,)]=0.0 is outside of
    # [...]` in preconditioned z-coordinates, naming no number the user typed.
    @test_logs (:warn, r"clamped into the box") match_mode = :any fit(
        NoLimits.MLE(optim_kwargs = K, lb = [-1.0, -1.0], ub = [1.0, 1.0]);
        theta_0_untransformed = (a = 5.0, s = 0.5)
    )

    @test fit(NoLimits.MLE(optim_kwargs = K, lb = [-5.0, -5.0], ub = [5.0, 5.0])) isa
        FitResult
    @test fit(NoLimits.MLE(optim_kwargs = K); penalty = (a = 10.0,)) isa FitResult
end

# ── Mini-batching over individuals (#281) ────────────────────────────────────
import Optimisers
using Random: MersenneTwister

# Fixed selection, and a recorder for the (nbatches, iter, rng) contract.
struct _MLEFixedSched
    sel::Vector{Int}
end
(s::_MLEFixedSched)(nbatches::Int, iter::Int, rng) = s.sel

mutable struct _MLESchedRecorder
    nb::Vector{Int}
    it::Vector{Int}
    rngs::Vector{Any}
end
_MLESchedRecorder() = _MLESchedRecorder(Int[], Int[], Any[])
function (r::_MLESchedRecorder)(nbatches::Int, iter::Int, rng)
    push!(r.nb, nbatches)
    push!(r.it, iter)
    push!(r.rngs, rng)
    return [1, 2]
end

@testset "MLE mini-batching (#281)" begin
    NL = NoLimits
    dm = fx_nore_dm()
    n_ind = length(NL.get_individuals(dm))
    @test n_ind == 4

    @testset "constructor" begin
        for T in (NL.MLE, NL.MAP)
            @test T().update_schedule === :all
            @test T(update_schedule = :all).update_schedule === :all
            @test T(update_schedule = "all").update_schedule === :all
            @test T(update_schedule = 2).update_schedule == 2
            f = (n, it, r) -> [1]
            @test T(update_schedule = f).update_schedule === f
            @test_throws ErrorException T(update_schedule = 0)
            @test_throws ErrorException T(update_schedule = "bogus")
            @test_throws ErrorException T(update_schedule = 1.5)

            @test T().optimizer isa OptimizationOptimJL.LBFGS
            @test T(update_schedule = 2).optimizer isa Optimisers.AbstractRule
            @test_logs (:warn, r"stochastic") T(
                update_schedule = 2, optimizer = OptimizationOptimJL.LBFGS()
            )
            adam = Optimisers.Adam(0.05)
            @test (@test_logs T(update_schedule = 2, optimizer = adam)).optimizer === adam
            lb = OptimizationOptimJL.LBFGS()
            @test (@test_logs T(update_schedule = :all, optimizer = lb)).optimizer === lb
        end
    end

    @testset ":all is unchanged" begin
        r1 = fit_model(
            dm, NL.MLE(; optim_kwargs = (maxiters = 3,));
            serialization = NoLimits.EnsembleSerial()
        )
        r2 = fit_model(
            dm, NL.MLE(; update_schedule = :all, optim_kwargs = (maxiters = 3,));
            serialization = NoLimits.EnsembleSerial()
        )
        @test get_objective(r1) == get_objective(r2)
        @test NL.get_params(r1; scale = :transformed) ==
            NL.get_params(r2; scale = :transformed)
    end

    @testset "_loglikelihood_indices" begin
        θ = NL.get_θ0_untransformed(dm)
        for ser in (NoLimits.EnsembleSerial(), NoLimits.EnsembleThreads())
            full = NL.loglikelihood(dm, θ, ComponentArray(); serialization = ser)
            @test NL._loglikelihood_indices(
                dm, θ, ComponentArray(), 1:n_ind; serialization = ser
            ) == full
            part = NL._loglikelihood_indices(
                dm, θ, ComponentArray(), [2, 4]; serialization = ser
            )
            singles = sum(
                NL._loglikelihood_indices(
                        dm, θ, ComponentArray(), [i]; serialization = ser
                    ) for i in (2, 4)
            )
            @test part ≈ singles
        end
    end

    @testset "one draw per optimizer iteration" begin
        rec = _MLESchedRecorder()
        rng = MersenneTwister(1)
        fit_model(
            dm,
            NL.MLE(;
                update_schedule = rec, optimizer = Optimisers.Adam(0.05),
                optim_kwargs = (maxiters = 4,)
            );
            serialization = NoLimits.EnsembleSerial(), rng = rng
        )
        @test rec.it == collect(1:4)
        @test all(==(n_ind), rec.nb)
        @test all(r -> r === rng, rec.rngs)
    end

    @testset "final objective is the full-data objective" begin
        penalty = (; a = 100.0)
        extra = θ -> 0.5 * θ.a^2
        res = fit_model(
            dm,
            NL.MLE(;
                update_schedule = _MLEFixedSched([1, 3]),
                optimizer = Optimisers.Adam(0.05), optim_kwargs = (maxiters = 5,)
            );
            penalty = penalty, extra_objective = extra,
            serialization = NoLimits.EnsembleSerial(), rng = MersenneTwister(2)
        )
        θ̂ = NL.get_params(res; scale = :untransformed)
        expected = -NL.loglikelihood(
            dm, θ̂, ComponentArray(); serialization = NoLimits.EnsembleSerial()
        ) + NL._penalty_value(θ̂, penalty) + extra(θ̂)
        @test get_objective(res) ≈ expected

        # The optimizer itself saw the scaled subset objective, with the global terms
        # (penalty, extra_objective) added once and unscaled. Adam(0.0) freezes θ at θ0.
        res0 = fit_model(
            dm,
            NL.MLE(;
                update_schedule = _MLEFixedSched([1, 3]),
                optimizer = Optimisers.Adam(0.0), optim_kwargs = (maxiters = 1,)
            );
            penalty = penalty, extra_objective = extra,
            serialization = NoLimits.EnsembleSerial(), rng = MersenneTwister(2)
        )
        θ00 = NL.get_θ0_untransformed(dm)
        expected_sub = (n_ind / 2) * (
            -NL._loglikelihood_indices(
                dm, θ00, ComponentArray(), [1, 3];
                serialization = NoLimits.EnsembleSerial()
            )
        ) + NL._penalty_value(θ00, penalty) + extra(θ00)
        @test NL.get_result(res0).solution.objective ≈ expected_sub rtol = 1.0e-12

        # The scaled subset objective is a different (unbiased) quantity.
        θ0 = NL.get_θ0_untransformed(dm)
        sub = (n_ind / 2) * NL._loglikelihood_indices(
            dm, θ0, ComponentArray(), [1, 3];
            serialization = NoLimits.EnsembleSerial()
        )
        @test sub != NL.loglikelihood(
            dm, θ0, ComponentArray(); serialization = NoLimits.EnsembleSerial()
        )
    end

    @testset "reproducible given the same rng" begin
        mk() = fit_model(
            dm,
            NL.MLE(;
                update_schedule = 2, optimizer = Optimisers.Adam(0.05),
                optim_kwargs = (maxiters = 3,)
            );
            serialization = NoLimits.EnsembleSerial(), rng = MersenneTwister(7)
        )
        @test NL.get_params(mk(); scale = :transformed) ==
            NL.get_params(mk(); scale = :transformed)
    end

    @testset "bounds are projected for Optimisers rules" begin
        θ0_t = NL.get_θ0_transformed(dm)
        lo = (; a = θ0_t.a - 0.01, b = θ0_t.b - 0.01, σ = θ0_t.σ - 0.01)
        hi = (; a = θ0_t.a + 0.01, b = θ0_t.b + 0.01, σ = θ0_t.σ + 0.01)
        res = fit_model(
            dm,
            NL.MLE(;
                update_schedule = 2, lb = lo, ub = hi,
                optimizer = Optimisers.Adam(1.0), optim_kwargs = (maxiters = 5,)
            );
            serialization = NoLimits.EnsembleSerial(), rng = MersenneTwister(3)
        )
        θ̂t = NL.get_params(res; scale = :transformed)
        names = (:a, :b, :σ)
        @test all(
            getproperty(lo, k) - 1.0e-12 <= θ̂t[k] <= getproperty(hi, k) + 1.0e-12
                for k in names
        )
        @test any(
            min(
                    abs(θ̂t[k] - getproperty(lo, k)), abs(θ̂t[k] - getproperty(hi, k))
                ) <= 1.0e-8 for k in names
        )

        # The bounded LBFGS path is untouched.
        res_all = fit_model(
            dm, NL.MLE(; lb = lo, ub = hi, optim_kwargs = (maxiters = 3,));
            serialization = NoLimits.EnsembleSerial()
        )
        @test res_all isa FitResult
    end
end

@testset "Rmath-backed outcome is rejected under ForwardDiff (#326)" begin
    # NoncentralT's logpdf goes through the Rmath C library, which cannot take duals.
    # Before the probe this failed with a bare `MethodError: Float64(::Dual)` from
    # inside the optimizer.
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ NoncentralT(5.0, a)
        end
    end
    dm = DataModel(
        model, DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [0.2, 0.4]);
        primary_id = :ID, time_col = :t
    )
    err = try
        fit_model(
            dm, NoLimits.MLE(; optim_kwargs = (maxiters = 1,));
            serialization = NoLimits.EnsembleSerial()
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not ForwardDiff-differentiable", err.msg)
    @test occursin("AutoFiniteDiff", err.msg)

    res = fit_model(
        dm,
        NoLimits.MLE(;
            adtype = Optimization.AutoFiniteDiff(), optim_kwargs = (maxiters = 1,)
        );
        serialization = NoLimits.EnsembleSerial()
    )
    @test res isa FitResult
    @test isfinite(get_objective(res))
end
