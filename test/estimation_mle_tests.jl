using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using OptimizationBBO
using OptimizationOptimisers
using OptimizationOptimJL

# Shared MLE/MAP testsets: models carry priors (MLE ignores them, MAP requires them).
const MLE_MAP_METHODS = (("MLE", NoLimits.MLE()), ("MAP", NoLimits.MAP()))

@testset "MLE non-ODE" begin
    res = fx_mle()                        # shared no-RE MLE fit
    @test res isa FitResult
    @test NoLimits.get_params(res; scale = :untransformed) isa ComponentArray
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
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior = Normal(0.0, 1.0))
            σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    for (label, method) in MLE_MAP_METHODS
        @testset "$label requires a free fixed effect" begin
            @test_throws ErrorException fit_model(
                dm, method; constants = (a = 0.2, σ = 0.3))
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
    model = @Model begin
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

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [0.1, 0.12, 0.11]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

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
        @test_throws ErrorException fit_model(dm, NoLimits.MAP();
            theta_0_untransformed = ComponentArray(a = 0.1, σ = 0.0))
    end
end

@testset "MLE penalties" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale = :log)
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
    @test isapprox(a_map, a_pen; rtol = 1e-4, atol = 1e-4)
end

@testset "MLE uses optim_kwargs" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale = :log)
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
    method = NoLimits.MLE(optim_kwargs = (; iterations = 1))
    res = fit_model(dm, method)

    @test res isa FitResult
    stats = res.result.solution.stats
    @test hasproperty(stats, :iterations)
    @test stats.iterations <= 1
end

function _mle_dm_basic()
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.5, scale = :log)
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
    return dm
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
        lb = lb, optim_kwargs = (; iterations = 5))
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
    method = NoLimits.MLE(optimizer = BFGS(), optim_kwargs = (;))
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
        optimizer = OptimizationOptimisers.Adam(0.05), optim_kwargs = (; maxiters = 2))
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
        optim_kwargs = (; iterations = 5))
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
            dm, NoLimits.MAP(; optim_kwargs = (maxiters = 2,)))
    end
end

@testset "fit_model starting-value validation" begin
    model = @Model begin
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
    df = DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [1.0, 1.1, 0.9, 1.0])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    # A plain NamedTuple start is accepted, like `constants`/`penalty`.
    res = fit_model(dm, NoLimits.MLE(); theta_0_untransformed = (a = 1.2, s = 0.4))
    @test res isa FitResult
    @test_throws ErrorException fit_model(dm, NoLimits.MLE();
        theta_0_untransformed = (a = 1.2, nope = 0.4))
    @test_throws ErrorException fit_model(dm, NoLimits.MLE();
        theta_0_untransformed = ComponentArray(a = NaN, s = 0.3))
    @test_throws ErrorException fit_model(dm, NoLimits.MLE();
        theta_0_untransformed = ComponentArray(a = Inf, s = 0.3))

    # A non-finite objective is never reported as converged (#208/#209/#214/#215).
    summ = NoLimits.FitSummary(Inf, true, NoLimits.get_params(res), nothing)
    @test summ.converged == false
end

@testset "fit_model option validation" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            s = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, s)
        end
    end
    dm = DataModel(model, DataFrame(ID = [1, 2], t = [0.0, 0.0], y = [0.1, 0.2]);
        primary_id = :ID, time_col = :t)
    K = (maxiters = 1,)
    fit(m; kw...) = fit_model(dm, m; kw...)

    # lb/ub are checked at the shared bounds resolver, so every estimator inherits it.
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K, ub = [0.4]))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K, ub = [NaN, NaN]))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K, lb = Float64[]))
    @test_throws ErrorException fit(NoLimits.MLE(
        optim_kwargs = K, lb = [0.3, 0.3], ub = [0.2, 0.2]))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); constants = [1.0, 2.0])
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); penalty = (a = "x",))
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); penalty = (a = NaN,))
    @test_throws ErrorException fit(
        NoLimits.MLE(optim_kwargs = K); extra_objective = (x, y) -> 0.0)
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K); extra_objective = 1)
    # Unknown / out-of-domain parameter overrides are named, on both entry points.
    @test_throws ErrorException fit(NoLimits.MLE(optim_kwargs = K);
        theta_0_untransformed = ComponentArray(a = 0.2, s = 0.3, x = 9.0))
    @test_throws ErrorException simulate_data(
        dm; theta_untransformed = ComponentArray(a = 0.2, s = -1.0))
    @test_throws ErrorException simulate_data(
        dm; theta_untransformed = ComponentArray(a = NaN, s = 0.3))

    @test fit(NoLimits.MLE(optim_kwargs = K, lb = [-5.0, -5.0], ub = [5.0, 5.0])) isa
          FitResult
    @test fit(NoLimits.MLE(optim_kwargs = K); penalty = (a = 10.0,)) isa FitResult
end
