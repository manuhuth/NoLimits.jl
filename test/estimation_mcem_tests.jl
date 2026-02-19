using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random
using SciMLBase
using OptimizationOptimisers
using OptimizationBBO

@testset "MCEM default sampler" begin
    method = NoLimits.MCEM()
    @test method.mcmc.sampler isa NUTS
    @test method.ebe.multistart_n == 50
    @test method.ebe.multistart_k == 10
    @test method.ebe.sampling == :lhs
    @test method.ebe_rescue.sampling == :lhs
end

@testset "MCEM basic (random effects)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
                             maxiters=2))
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

@testset "MCEM serial vs threaded is reproducible" begin
    Threads.nthreads() < 2 && return

    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.MCEM(; sampler=MH(),
                                   turing_kwargs=(n_samples=8, n_adapt=0, progress=false, verbose=false),
                                   maxiters=2,
                                   progress=false)
    res_serial = fit_model(dm, method; serialization=EnsembleSerial(), rng=MersenneTwister(123))
    res_threads = fit_model(dm, method; serialization=EnsembleThreads(), rng=MersenneTwister(123))
    @test res_serial.summary.objective == res_threads.summary.objective
    @test collect(NoLimits.get_params(res_serial, scale=:untransformed)) ==
          collect(NoLimits.get_params(res_threads, scale=:untransformed))
end

@testset "MCEM basic with NUTS" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=10, n_adapt=5, progress=false),
                             maxiters=2))
    @test res isa FitResult
end

@testset "MCEM convergence requires both parameter and Q stabilization" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
        maxiters=2,
        consecutive_params=1,
        atol_theta=Inf,
        rtol_theta=Inf,
        atol_Q=0.0,
        rtol_Q=0.0
    ))
    # If stopping used only parameter tolerance, this would stop after 1 iteration.
    @test res.result.iterations == 2
end

@testset "MCEM multiple RE groups" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        SITE = [:X, :X, :X, :X, :Y, :Y, :Y, :Y],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
                             maxiters=2))
    @test res isa FitResult
    re = NoLimits.get_random_effects(dm, res)
    @test !isempty(re)
end

@testset "MCEM constants_re" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
                             maxiters=2);
                    constants_re=(; η=(; A=0.0,)))
    @test res isa FitResult
end

@testset "MCEM constants for fixed effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
                             maxiters=2);
                    constants=(a=0.2,))
    @test res isa FitResult
end

@testset "MCEM RE distribution with constant covariates" begin
    model = @Model begin
        @fixedEffects begin
            μ0 = RealNumber(0.0)
            β = RealNumber(0.5)
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariate(constant_on=:ID)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ0 + β * x, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        x = [1.0, 1.0, 2.0, 2.0],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
                             maxiters=2))
    @test res isa FitResult
end

@testset "MCEM threaded E-step" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.0, -0.05, 0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
                             maxiters=2);
                    serialization=EnsembleThreads())
    @test res isa FitResult
end

@testset "MCEM multivariate RE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], I(2)); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=10, n_adapt=0, progress=false),
                             maxiters=2))
    @test res isa FitResult
end

@testset "MCEM multivariate RE with NUTS" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], I(2)); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=NUTS(5, 0.3), turing_kwargs=(n_samples=8, n_adapt=4, progress=false),
                             maxiters=2))
    @test res isa FitResult
end

@testset "MCEM optimizer Adam (OptimizationOptimisers)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.MCEM(optimizer=OptimizationOptimisers.Adam(0.05),
                  optim_kwargs=(; maxiters=3),
                  sampler=MH(),
                  turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
                  maxiters=2)
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "MCEM optimizer BlackBoxOptim (OptimizationBBO)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    lb, ub = default_bounds_from_start(dm; margin=1.0)
    method = NoLimits.MCEM(optimizer=OptimizationBBO.BBO_adaptive_de_rand_1_bin_radiuslimited(),
                  optim_kwargs=(; iterations=3),
                  sampler=MH(),
                  turing_kwargs=(n_samples=6, n_adapt=0, progress=false),
                  maxiters=2,
                  lb=lb, ub=ub)
    res = fit_model(dm, method)
    @test res isa FitResult
end

@testset "MCEM with ODE model" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            k = RealNumber(0.2)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @initialDE begin
            x1 = exp(a + η)
        end

        @DifferentialEquation begin
            D(x1) ~ -exp(k) * x1
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 0.8, 1.05, 0.85]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(), turing_kwargs=(n_samples=6, n_adapt=0, progress=false),
                             maxiters=2))
    @test res isa FitResult
end

@testset "MCEM threaded helper cache preserves ODE options" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + η
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [0.9, 0.7]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    ll_cache = build_ll_cache(dm; ode_kwargs=(abstol=1e-8, reltol=1e-7))
    threaded = NoLimits._mcem_thread_caches(dm, ll_cache, 2)
    @test length(threaded) == 2
    @test all(c -> c.ode_args == ll_cache.ode_args, threaded)
    @test all(c -> c.ode_kwargs == ll_cache.ode_kwargs, threaded)
end

@testset "MCEM thread RNGs are reproducible from passed rng" begin
    r1 = NoLimits._mcem_thread_rngs(MersenneTwister(42), 3)
    r2 = NoLimits._mcem_thread_rngs(MersenneTwister(42), 3)
    r3 = NoLimits._mcem_thread_rngs(MersenneTwister(99), 3)
    s1 = [rand(r, Float64) for r in r1]
    s2 = [rand(r, Float64) for r in r2]
    s3 = [rand(r, Float64) for r in r3]
    @test s1 == s2
    @test s1 != s3
end

@testset "MCEM non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.7); column=:ID)
        end

        @formulas begin
            λ = exp(a + b * z + η)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.2, 0.4, 0.5, 0.8, 1.0],
        y = [1, 1, 2, 2, 3, 4]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCEM(; sampler=MH(),
                                              turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
                                              maxiters=2))
    @test res isa FitResult
    @test NoLimits.get_converged(res) isa Bool
end

@testset "MCEM final EBE rescue options are configurable" begin
    method = NoLimits.MCEM(;
        ebe_rescue_on_high_grad=false,
        ebe_rescue_multistart_n=77,
        ebe_rescue_multistart_k=11,
        ebe_rescue_max_rounds=9,
        ebe_rescue_grad_tol=1e-5
    )
    @test method.ebe_rescue.enabled == false
    @test method.ebe_rescue.multistart_n == 77
    @test method.ebe_rescue.multistart_k == 11
    @test method.ebe_rescue.max_rounds == 9
    @test method.ebe_rescue.grad_tol == 1e-5
end

@testset "MCEM get_random_effects recomputes EB modes with rescue options" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.1, 0.2, 0.0, -0.1])
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    method = NoLimits.MCEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=8, n_adapt=0, progress=false),
        maxiters=1,
        ebe_rescue_on_high_grad=true,
        ebe_rescue_multistart_n=12,
        ebe_rescue_multistart_k=4,
        ebe_rescue_max_rounds=2,
        ebe_rescue_grad_tol=1e-7
    )
    res = fit_model(dm, method; store_eb_modes=false)
    re = NoLimits.get_random_effects(dm, res)
    @test haskey(re, :η)
    @test nrow(re.η) == length(unique(df.ID))
end
