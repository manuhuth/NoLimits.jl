using NoLimits
using DataFrames
using Distributions
using Turing

function _setup_fit_dm()
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(softplus(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1, 1],
        t = [0.0, 1.0, 2.0, 3.0],
        y = [1.0, 1.05, 0.98, 1.02]
    )

    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _setup_fit_dm_large(; n_ids=100, n_time=5)
    model = @Model begin
        @helpers begin
            softplus(u) = log1p(exp(u))
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(softplus(a), σ)
        end
    end

    IDs = repeat(1:n_ids, inner=n_time)
    t = repeat(collect(0.0:1.0:(n_time - 1)), n_ids)
    y = 1.0 .+ 0.01 .* t .+ 0.05 .* randn(length(t))

    df = DataFrame(ID=IDs, t=t, y=y)
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function run_estimation_fit_benchmarks(; n_mle=3, n_map=3, n_mcmc=2, large=false)
    dm = large ? _setup_fit_dm_large() : _setup_fit_dm()

    # Warmup
    fit_model(dm, MLE())
    fit_model(dm, MAP())
    fit_model(dm, MCMC(; sampler=MH(), turing_kwargs=(n_samples=10, n_adapt=0, progress=false)))
    GC.gc()

    alloc_mle = @allocated begin
        for _ in 1:n_mle
            fit_model(dm, MLE())
        end
    end
    elapsed_mle = @elapsed begin
        for _ in 1:n_mle
            fit_model(dm, MLE())
        end
    end

    alloc_map = @allocated begin
        for _ in 1:n_map
            fit_model(dm, MAP())
        end
    end
    elapsed_map = @elapsed begin
        for _ in 1:n_map
            fit_model(dm, MAP())
        end
    end

    alloc_mcmc = @allocated begin
        for _ in 1:n_mcmc
            fit_model(dm, MCMC(; sampler=MH(), turing_kwargs=(n_samples=200, n_adapt=0, progress=false)))
        end
    end
    elapsed_mcmc = @elapsed begin
        for _ in 1:n_mcmc
            fit_model(dm, MCMC(; sampler=MH(), turing_kwargs=(n_samples=200, n_adapt=0, progress=false)))
        end
    end

    @info "fit_model benchmark (MLE)" n=n_mle elapsed_sec=elapsed_mle alloc_bytes=alloc_mle alloc_per_call=alloc_mle / n_mle
    @info "fit_model benchmark (MAP)" n=n_map elapsed_sec=elapsed_map alloc_bytes=alloc_map alloc_per_call=alloc_map / n_map
    @info "fit_model benchmark (MCMC MH)" n=n_mcmc elapsed_sec=elapsed_mcmc alloc_bytes=alloc_mcmc alloc_per_call=alloc_mcmc / n_mcmc
    return (mle=(elapsed=elapsed_mle, alloc=alloc_mle),
            map=(elapsed=elapsed_map, alloc=alloc_map),
            mcmc=(elapsed=elapsed_mcmc, alloc=alloc_mcmc))
end

run_estimation_fit_benchmarks(; n_mle=3, n_map=3, n_mcmc=2, large=true)