# Benchmark loglikelihood evaluation and allocations.
using NoLimits
using DataFrames
using ComponentArrays
using OrdinaryDiffEq

function setup_ll_nonode()
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(softplus(a + η), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model.fixed.fixed)
    η_list = [ComponentArray((η = 0.1,)), ComponentArray((η = -0.1,))]
    return (dm, θ, η_list)
end

function setup_ll_ode()
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
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
        y = [1.0, 0.9]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    θ = get_θ0_untransformed(model_saveat.fixed.fixed)
    η_list = [ComponentArray((η = 0.1,))]
    return (dm, θ, η_list)
end

function run_loglik_benchmarks(; nsteps=1_000)
    dm, θ, η_list = setup_ll_nonode()
    dm_ode, θ_ode, η_list_ode = setup_ll_ode()

    # Warmup (build caches / compile)
    solver_nonode = get_solver_config(dm.model)
    solver_ode = get_solver_config(dm_ode.model)
    cache_nonode = build_ll_cache(dm; nthreads=Threads.maxthreadid())
    cache_ode = build_ll_cache(dm_ode; nthreads=Threads.maxthreadid())
    loglikelihood(dm, θ, η_list; cache=cache_nonode)
    loglikelihood(dm_ode, θ_ode, η_list_ode; cache=cache_ode)
    GC.gc()

    alloc_nonode = @allocated begin
        for _ in 1:nsteps
            loglikelihood(dm, θ, η_list; cache=cache_nonode)
        end
    end
    elapsed_nonode = @elapsed begin
        for _ in 1:nsteps
            loglikelihood(dm, θ, η_list; cache=cache_nonode)
        end
    end

    alloc_ode = @allocated begin
        for _ in 1:nsteps
            loglikelihood(dm_ode, θ_ode, η_list_ode; cache=cache_ode)
        end
    end
    elapsed_ode = @elapsed begin
        for _ in 1:nsteps
            loglikelihood(dm_ode, θ_ode, η_list_ode; cache=cache_ode)
        end
    end

    @info "loglikelihood benchmark (non-ODE)" nsteps elapsed_sec=elapsed_nonode alloc_bytes=alloc_nonode alloc_per_call=alloc_nonode / nsteps
    @info "loglikelihood benchmark (ODE)" nsteps elapsed_sec=elapsed_ode alloc_bytes=alloc_ode alloc_per_call=alloc_ode / nsteps
    return (nonode=(elapsed=elapsed_nonode, alloc=alloc_nonode),
            ode=(elapsed=elapsed_ode, alloc=alloc_ode))
end
