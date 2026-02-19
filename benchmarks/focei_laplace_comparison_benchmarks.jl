# Benchmark FOCEI vs Laplace and FOCEI info modes on the same NLME problem.
using NoLimits
using DataFrames
using Distributions
using Random

function _setup_focei_laplace_dm(; n_ids=120, n_time=4, seed=1234)
    Random.seed!(seed)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
        end

        @covariates begin
            t = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            μ = a + b * t + η
            y ~ Normal(μ, σ)
        end
    end

    ids = repeat(1:n_ids, inner=n_time)
    tvals = repeat(collect(0.0:(n_time - 1)), n_ids)
    η_true = rand(Normal(0.0, 1.0), n_ids)
    y = similar(tvals, Float64)
    for i in eachindex(ids)
        id = ids[i]
        μ = 0.5 + 0.2 * tvals[i] + η_true[id]
        y[i] = rand(Normal(μ, 0.3))
    end

    df = DataFrame(ID=ids, t=tvals, y=y)
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

@inline function _param_max_abs_diff(θa, θb)
    va = collect(θa)
    vb = collect(θb)
    length(va) == length(vb) || return Inf
    return maximum(abs.(va .- vb))
end

function _fit_bench(dm::DataModel, method; n=3)
    # Warmup compile path.
    NoLimits.fit_model(dm, method)
    GC.gc()

    alloc = @allocated begin
        for _ in 1:n
            NoLimits.fit_model(dm, method)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:n
            NoLimits.fit_model(dm, method)
        end
    end

    # Keep one representative result for method-comparison metrics.
    res = NoLimits.fit_model(dm, method)
    return (elapsed=elapsed, alloc=alloc, res=res)
end

function _result_metrics(res)
    θ = NoLimits.get_params(res; scale=:untransformed)
    ll = try
        NoLimits.get_loglikelihood(res)
    catch
        NaN
    end
    return (objective=NoLimits.get_objective(res),
            converged=NoLimits.get_converged(res),
            loglikelihood=ll,
            params=θ,
            notes=NoLimits.get_notes(res))
end

function _focei_fallback_summary(notes)
    total = hasproperty(notes, :focei_fallback_total) ? getproperty(notes, :focei_fallback_total) : missing
    info_logdet = hasproperty(notes, :focei_fallback_info_logdet) ? getproperty(notes, :focei_fallback_info_logdet) : missing
    mode_hessian = hasproperty(notes, :focei_fallback_mode_hessian) ? getproperty(notes, :focei_fallback_mode_hessian) : missing
    at_solution = hasproperty(notes, :focei_fallback_at_solution) ? getproperty(notes, :focei_fallback_at_solution) : missing
    return (total=total, info_logdet=info_logdet, mode_hessian=mode_hessian, at_solution=at_solution)
end

function run_focei_laplace_comparison_benchmarks(; n=3,
                                                  n_ids=120,
                                                  n_time=4,
                                                  seed=1234,
                                                  maxiters=8,
                                                  inner_maxiters=30,
                                                  mode_sensitivity=:exact_hessian)
    dm = _setup_focei_laplace_dm(; n_ids=n_ids, n_time=n_time, seed=seed)

    laplace = NoLimits.Laplace(; optim_kwargs=(maxiters=maxiters,),
                                       inner_kwargs=(maxiters=inner_maxiters,),
                                       multistart_n=0,
                                       multistart_k=0)
    focei_opg = NoLimits.FOCEI(; optim_kwargs=(maxiters=maxiters,),
                                       inner_kwargs=(maxiters=inner_maxiters,),
                                       multistart_n=0,
                                       multistart_k=0,
                                       info_jitter=1e-4,
                                       info_max_tries=10,
                                       info_mode=:opg,
                                       mode_sensitivity=mode_sensitivity)
    focei_fisher_common = NoLimits.FOCEI(; optim_kwargs=(maxiters=maxiters,),
                                                 inner_kwargs=(maxiters=inner_maxiters,),
                                                 multistart_n=0,
                                                 multistart_k=0,
                                                 info_jitter=1e-4,
                                                 info_max_tries=10,
                                                 info_mode=:fisher_common,
                                                 mode_sensitivity=mode_sensitivity)

    laplace_bench = _fit_bench(dm, laplace; n=n)
    focei_opg_bench = _fit_bench(dm, focei_opg; n=n)
    focei_fisher_bench = _fit_bench(dm, focei_fisher_common; n=n)

    laplace_metrics = _result_metrics(laplace_bench.res)
    focei_opg_metrics = _result_metrics(focei_opg_bench.res)
    focei_fisher_metrics = _result_metrics(focei_fisher_bench.res)

    cmp_laplace_vs_focei = (
        objective_abs_diff=abs(laplace_metrics.objective - focei_opg_metrics.objective),
        loglikelihood_abs_diff=abs(laplace_metrics.loglikelihood - focei_opg_metrics.loglikelihood),
        params_max_abs_diff=_param_max_abs_diff(laplace_metrics.params, focei_opg_metrics.params),
        laplace_converged=laplace_metrics.converged,
        focei_converged=focei_opg_metrics.converged
    )
    cmp_focei_modes = (
        objective_abs_diff=abs(focei_opg_metrics.objective - focei_fisher_metrics.objective),
        loglikelihood_abs_diff=abs(focei_opg_metrics.loglikelihood - focei_fisher_metrics.loglikelihood),
        params_max_abs_diff=_param_max_abs_diff(focei_opg_metrics.params, focei_fisher_metrics.params),
        focei_opg_converged=focei_opg_metrics.converged,
        focei_fisher_common_converged=focei_fisher_metrics.converged
    )

    @info "Fit benchmark (Laplace)" n elapsed_sec=laplace_bench.elapsed alloc_bytes=laplace_bench.alloc alloc_per_call=laplace_bench.alloc / n
    @info "Fit benchmark (FOCEI :opg)" n mode_sensitivity elapsed_sec=focei_opg_bench.elapsed alloc_bytes=focei_opg_bench.alloc alloc_per_call=focei_opg_bench.alloc / n
    @info "Fit benchmark (FOCEI :fisher_common)" n mode_sensitivity elapsed_sec=focei_fisher_bench.elapsed alloc_bytes=focei_fisher_bench.alloc alloc_per_call=focei_fisher_bench.alloc / n
    @info "Result comparison (Laplace vs FOCEI :opg)" cmp_laplace_vs_focei
    @info "Result comparison (FOCEI :opg vs :fisher_common)" cmp_focei_modes
    @info "FOCEI fallback summary (:opg)" _focei_fallback_summary(focei_opg_metrics.notes)
    @info "FOCEI fallback summary (:fisher_common)" _focei_fallback_summary(focei_fisher_metrics.notes)

    return (
        laplace=(elapsed=laplace_bench.elapsed, alloc=laplace_bench.alloc, metrics=laplace_metrics),
        focei_opg=(elapsed=focei_opg_bench.elapsed, alloc=focei_opg_bench.alloc, metrics=focei_opg_metrics),
        focei_fisher_common=(elapsed=focei_fisher_bench.elapsed, alloc=focei_fisher_bench.alloc, metrics=focei_fisher_metrics),
        compare_laplace_vs_focei=cmp_laplace_vs_focei,
        compare_focei_modes=cmp_focei_modes
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_focei_laplace_comparison_benchmarks()
end
