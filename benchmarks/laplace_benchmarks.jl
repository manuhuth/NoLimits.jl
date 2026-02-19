# Benchmark Laplace batch logf and EBE solve.
using NoLimits
using DataFrames
using ComponentArrays
using OptimizationOptimJL
using LineSearches
using LinearAlgebra

function _laplace_setup()
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(0.5)
            μ = RealVector([0.0, 0.0])
        end

        @randomEffects begin
            η_id = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
            η_site = RandomEffect(MvNormal(μ, LinearAlgebra.I(2)); column=:SITE)
        end

        @formulas begin
            lin = a + η_id[1] + η_site[2]
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = repeat(1:30, inner=2),
        SITE = repeat([:A, :B, :C], inner=20),
        t = repeat([0.0, 1.0], 30),
        y = randn(60)
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos = _build_laplace_batch_infos(dm, NamedTuple())
    info = batch_infos[1]
    cache = build_ll_cache(dm)
    θ = get_θ0_untransformed(model.fixed.fixed)
    return (dm, info, cache, θ)
end

function run_laplace_benchmarks(; nsteps=200)
    dm, info, cache, θ = _laplace_setup()
    b = zeros(info.n_b)
    constants_re = NamedTuple()

    # Warmup
    _laplace_logf_batch(dm, info, θ, b, constants_re, cache)
    _laplace_solve_batch!(dm, info, θ, constants_re, cache;
                          optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()))
    GC.gc()

    alloc_logf = @allocated begin
        for _ in 1:nsteps
            _laplace_logf_batch(dm, info, θ, b, constants_re, cache)
        end
    end
    elapsed_logf = @elapsed begin
        for _ in 1:nsteps
            _laplace_logf_batch(dm, info, θ, b, constants_re, cache)
        end
    end

    alloc_ebe = @allocated begin
        for _ in 1:nsteps
            _laplace_solve_batch!(dm, info, θ, constants_re, cache;
                                  optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()))
        end
    end
    elapsed_ebe = @elapsed begin
        for _ in 1:nsteps
            _laplace_solve_batch!(dm, info, θ, constants_re, cache;
                                  optimizer=OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking()))
        end
    end

    @info "Laplace logf benchmark" nsteps elapsed_sec=elapsed_logf alloc_bytes=alloc_logf alloc_per_call=alloc_logf / nsteps
    @info "Laplace EBE benchmark" nsteps elapsed_sec=elapsed_ebe alloc_bytes=alloc_ebe alloc_per_call=alloc_ebe / nsteps
    return (logf=(elapsed=elapsed_logf, alloc=alloc_logf),
            ebe=(elapsed=elapsed_ebe, alloc=alloc_ebe))
end

run_laplace_benchmarks()
